import gc
import numpy as np
import argparse
from config import Config, FUSION_PRETRAINING, FUSION_FINETUNING, FUSION_TESTING
import os, json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["fusion_pretraining", "fusion_finetuning", 'fusion_testing'], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--gpu", type=str, default=None, help="Override GPU index (e.g. '0' or '1')")
    parser.add_argument("--moda", type=str, default=None, choices=["B", "ScS", "W"], help="Override modality for single-modality runs")
    parser.add_argument("--start_fold", type=int, default=0, help="Start from this fold index (skip earlier folds)")
    args = parser.parse_args()
    if args.mode == "fusion_pretraining": 
        mode = FUSION_PRETRAINING 
    elif args.mode == "fusion_finetuning":
        mode = FUSION_FINETUNING
    else:
        mode = FUSION_TESTING
    
    config = Config(mode)
    if args.gpu is not None:
        config.gpu = args.gpu
    if args.moda is not None:
        config.moda = args.moda
        config.output_dir = config.output_dir + "_" + args.moda # type: ignore
        if hasattr(config, 'feature_dir') and config.feature_dir is not None:
            config.feature_dir = config.feature_dir + "_" + args.moda
        if hasattr(config, 'out_file_path'):
            config.out_file_path = os.path.join(config.output_dir,
                "output_test_" + config.site if config.site is not None else "output_test")
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, 'hyperparameter.json')
    if mode != FUSION_TESTING:
        with open(json_path,'w') as f:
            f.write(json.dumps(vars(config), ensure_ascii=False, indent=4, separators=(',', ':')))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    random.seed(config.seed)
    np.random.seed(config.seed)

    import torch
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.utils.data import WeightedRandomSampler
    from transformers import get_cosine_schedule_with_warmup
    from data_loader import (
        dataset_split,
        nested_pad_cross_validation,
        raw_collate_fn,
        DictDataset,
        process_json,
        dataset_sampling_weights,
    )
    from torch.utils.data import DataLoader
    from fusion_funcs import TrainingModel
    from losses import GeneralizedSupervisedNTXenLoss, FocalLoss, NTXenLoss, AdaSPLoss, LocalTokenAlignmentLoss, CLIPLoss, SoftCLIPLoss, SupervisedCLIPLoss, DirectTokenPatchContrastiveLoss
    from torch.nn import CrossEntropyLoss
    from models.img_txt_fusion import ImgTxtFusion

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    full_list, full_label, full_sen_attr, full_var, full_moda_indices, full_group = dataset_split( # type: ignore
                            config.train_csv_path, config.num_var, config.moda, config.var_idx, config.sen_attr_idx, config.group_idx, config.site, avail_moda=config.avail_moda)    

    for fold_id in range(args.start_fold, config.folds):
        print("\n-----------Fold: {}------------".format(fold_id))
        (
            train_list,
            val_list,
            test_list,
            train_label,
            val_label,
            test_label,
            train_var,
            val_var,
            test_var,
            train_sen_attr,
            val_sen_attr,
            test_sen_attr,
        ) = nested_pad_cross_validation(
            full_list=full_list,
            full_label=full_label,
            full_sen_attr=full_sen_attr,
            full_var=full_var,
            full_group=full_group,
            data_dir=config.data_dir,
            fold_id=fold_id,
            folds=config.folds,
            inner_fraction=config.inner_fraction,
            seed=config.split_seed,
        )
        if not config.use_var:
            train_var, val_var, test_var = None, None, None

        train_metadata_dict = process_json(train_list, config.metadata_path, config.knowledge_path, config.attr_list, config.knowledge_usage, config.label_dict, config.class_penalty, config.merge_texts)
        val_metadata_dict = process_json(val_list, config.metadata_path, config.knowledge_path, config.attr_list, config.knowledge_usage, config.label_dict, config.class_penalty, config.merge_texts)
        test_case_list = list(
            dict.fromkeys(name.split("/")[0] for name in test_list)
        )
        test_metadata_dict = process_json(test_case_list, config.metadata_path, config.knowledge_path, config.attr_list, config.knowledge_usage, config.label_dict, config.class_penalty, config.merge_texts)

        if config.sampling_mode is not None:
            train_weights = dataset_sampling_weights(config.sampling_mode, train_sen_attr, train_label, weight_power=config.weight_power)
            print(f"Using {config.sampling_mode} sampling strategy for training data.")
            print("Sample weights range: [{:.4f}, {:.4f}]".format(min(train_weights), max(train_weights)))
            sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataset_train = DictDataset(config.data_dir, train_list, train_label, train_metadata_dict, train_sen_attr, config.moda, config.moda_list,
                                    config.multi_moda, config.input_size, True, True, text_mode="text",
                                    image_mean=config.image_mean, image_std=config.image_std)
        dataset_val = DictDataset(config.data_dir, val_list, val_label, val_metadata_dict, val_sen_attr, config.moda, config.moda_list,
                                    config.multi_moda, config.input_size, True, False, text_mode="text",
                                    image_mean=config.image_mean, image_std=config.image_std)
        dataset_test = DictDataset(config.data_dir, test_list, test_label, test_metadata_dict, test_sen_attr, config.moda, config.moda_list,
                                    config.multi_moda, config.input_size, False, False, text_mode="text",
                                    image_mean=config.image_mean, image_std=config.image_std)
        
        loader_train = DataLoader(dataset_train, 
                                batch_size=config.batch_size,
                                shuffle=shuffle,
                                sampler=sampler,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        loader_val = DataLoader(dataset_val,
                                batch_size=config.batch_size,
                                shuffle=False,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        loader_test = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers
                                )
        print("Train dataset size: {}, Val dataset size: {}, Test dataset size: {}".format(
            len(dataset_train), len(dataset_val), len(dataset_test)))
        
        if config.visual_pretrained is not None:
            visual_pretrained = config.visual_pretrained + '_fold_{}.pth'.format(fold_id)
        else:
            visual_pretrained = None
        
        if config.mode == FUSION_PRETRAINING:
            net = ImgTxtFusion(config, config.vision_encoder, config.text_encoder, visual_pretrained)
        elif config.mode == FUSION_FINETUNING:
            net = ImgTxtFusion(config, config.vision_encoder, config.text_encoder, visual_pretrained)
        elif config.mode == FUSION_TESTING:
            net = ImgTxtFusion(config, config.vision_encoder, config.text_encoder, visual_pretrained)
        else:
            raise ValueError("Unknown mode: {}".format(config.mode))
        
        if config.img_cont_loss == "Supervised":
            img_cont_loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                                         kernel='rbf', sigma=config.sigma, return_logits=True)  # type: ignore
        elif config.img_cont_loss == "Unsupervised":
            img_cont_loss = NTXenLoss(temperature=config.temperature, return_logits=True)
        elif config.img_cont_loss == "AdaSP":
            img_cont_loss = AdaSPLoss(temp=config.temperature, loss_type=config.a_loss_type) # type: ignore
        else:
            raise ValueError("Unknown image contrastive loss: {}".format(config.img_cont_loss))

        if config.global_clip_loss == "CLIP":
            global_clip_loss = CLIPLoss(temperature=config.temperature)
        elif config.global_clip_loss == "Unsupervised":
            global_clip_loss = NTXenLoss(temperature=config.temperature, return_logits=False)
        else:
            raise ValueError("Unknown global contrastive loss: {}".format(config.global_clip_loss))
        
        if config.soft_clip_loss == "SoftCLIP":
            soft_clip_loss = SoftCLIPLoss(init_temp=config.temperature)
        elif config.soft_clip_loss == "SupervisedCLIP":
            soft_clip_loss = SupervisedCLIPLoss(num_classes=config.num_classes, init_temp=config.temperature)
        else:
            raise ValueError("Unknown soft contrastive loss: {}".format(config.soft_clip_loss))

        if config.local_clip_loss == "LocalCLIP":
            local_clip_loss = LocalTokenAlignmentLoss(temperature=config.temperature)
        elif config.local_clip_loss == "DirectTokenPatchContrastive":
            local_clip_loss = DirectTokenPatchContrastiveLoss(text_dim=config.d_model, image_dim=config.d_visual, temperature=config.temperature)
        else:
            raise ValueError("Unknown local contrastive loss: {}".format(config.local_clip_loss))

        if config.clas_loss == "CE":
            clas_loss = CrossEntropyLoss()
        elif config.clas_loss == "Focal":
            clas_loss = FocalLoss(gamma=config.gamma, alpha=config.alpha) # type: ignore
        else:
            raise ValueError("Unknown classification loss: {}".format(config.clas_loss))
        
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.scheduler == "cosine":
            num_warmup_steps = int(config.warm_epochs * len(loader_train))
            num_training_steps = int(config.nb_epochs * len(loader_train))
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        else:
            scheduler = None
        model = TrainingModel(fold_id, net, img_cont_loss, global_clip_loss, local_clip_loss, soft_clip_loss, clas_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler)

        if config.mode == FUSION_PRETRAINING:
            model.pretraining()
        elif config.mode == FUSION_FINETUNING:
            model.finetuning()
            model.test()
        else:
            model.test()
        
        torch.cuda.empty_cache()
        gc.collect()
