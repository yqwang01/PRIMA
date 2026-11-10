import numpy as np
import argparse
from config import Config, ENSEMBLE_TRAIN, ENSEMBLE_TEST
import os, json
import random



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ensemble_train", "ensemble_test"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    args = parser.parse_args()
    if args.mode == "ensemble_train": 
        mode = ENSEMBLE_TRAIN
    else:
        mode = ENSEMBLE_TEST

    config = Config(mode)
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, 'hyperparameter.json')
    if mode != ENSEMBLE_TEST:
        with open(json_path,'w') as f:
            f.write(json.dumps(vars(config), ensure_ascii=False, indent=4, separators=(',', ':')))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(config.seed)
    np.random.seed(config.seed)

    import torch
    from torch.optim.lr_scheduler import ExponentialLR
    from transformers import get_cosine_schedule_with_warmup
    from data_loader import (
        dataset_split_ensemble,
        nested_pad_cross_validation_ensemble,
        EnsembleDatasetV2,
        process_json,
        dataset_sampling_weights,
    )
    from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
    from ensemble_funcs import TrainingModel
    from losses import FocalLoss
    from torch.nn import CrossEntropyLoss
    from models.linear_ensemble import LinearEnsembleNet
    from models.linear_ensemble_txt import FusionMLP
    from models.simple_self_attention_fusion import SimpleSelfAttentionFusion
    from models.LLM_ensemble import FeatureFusionLLM, setup_lora

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if mode == PRETRAINING:
    # train_list, val_list, test_list, train_label, val_label, test_label = dataset_split_pretraining(config.data_dir, config.csv_path, config.train_ratio, config.val_ratio)

    full_list, full_label, full_var, full_sen_attr, indices, full_group = dataset_split_ensemble( # type: ignore
                            config.train_csv_path, config.num_var, config.var_idx, config.sen_attr_idx, config.group_idx, config.site, False, config.avail_moda)
    
    for fold_id in range(config.folds):
        print("\n-----------Fold: {}------------".format(fold_id))
        (
            train_list,
            val_list,
            test_list,
            train_label,
            val_label,
            test_label,
            train_sen_attr,
            val_sen_attr,
            test_sen_attr,
            train_var,
            val_var,
            test_var,
        ) = nested_pad_cross_validation_ensemble(
            full_list=full_list,
            full_label=full_label,
            full_sen_attr=full_sen_attr,
            full_var=full_var,
            full_group=full_group,
            fold_id=fold_id,
            folds=config.folds,
            inner_fraction=config.inner_fraction,
            seed=config.split_seed,
        )
        
        class_ids = None
        if config.ense_model == "Linear":
            net = LinearEnsembleNet(num_moda=len(config.moda_list) if config.moda_list is not None else 1, d_ensemble=config.d_ensemble)
        elif config.ense_model == "LinearText":
            net = FusionMLP(num_moda=len(config.moda_list) if config.moda_list is not None else 1, mode=config.text_mode,
                    text_max_segs=config.text_max_segs, txt_dim=config.d_ensemble_text, img_dim=config.d_ensemble, num_classes=config.num_classes,
                    hidden1=config.mlp_hidden1, hidden2=config.mlp_hidden2, dropout_rate=config.mlp_dropout)
        elif config.ense_model == "SimpleSelfAttentionFusion":
            net = SimpleSelfAttentionFusion(num_classes=config.num_classes,
                                            num_layers=config.num_layers_btransformer,
                                            hidden_size=config.d_ensemble_text,
                                            img_hidden=config.d_ensemble,
                                            token_mode=config.simple_fusion_token_mode,
                                            dropout_rate=config.transformer_dropout,
                                            attention_dropout_rate=config.transformer_attention_dropout)
        elif config.ense_model == "LLM":
            net = FeatureFusionLLM(base_model_path=config.llm_base_model_path, class_names=config.class_names, img_feat_dim=config.d_ensemble, 
            text_feat_dim=config.d_ensemble_text, feat_dropout=config.feat_dropout, noise_scale=config.noise_scale,
            downsample_rate=config.downsample_rate, padding=config.padding, downsample_rate_text=config.downsample_rate_text,
            use_patch_tokens=config.use_patch_tokens, separate_text_cls=config.separate_text_cls)
            net = setup_lora(net, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules, use_lora=config.use_lora)
            class_ids = net.target_token_ids.cpu().numpy().tolist()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            raise ValueError("Unknown model: %s"%config.model)

        img_dir = os.path.join(config.img_feature_dir, 'fold_{}'.format(fold_id))
        text_dir = os.path.join(config.text_feature_dir, 'fold_{}'.format(fold_id)) if os.path.isdir(os.path.join(config.text_feature_dir, 'fold_{}'.format(fold_id))) else config.text_feature_dir
        dataset_train = EnsembleDatasetV2(img_dir, text_dir, train_list, train_label, train_sen_attr, config.moda_list, 
                    config.d_ensemble, config.d_ensemble_text, config.text_mode, config.loss_mode, class_ids, config.text_max_segs, config.img_num_tokens)  # pyright: ignore[reportArgumentType]
        dataset_val = EnsembleDatasetV2(img_dir, text_dir, val_list, val_label, val_sen_attr, config.moda_list, 
                    config.d_ensemble, config.d_ensemble_text, config.text_mode, config.loss_mode, class_ids, config.text_max_segs, config.img_num_tokens)  # pyright: ignore[reportArgumentType]
        dataset_test = EnsembleDatasetV2(img_dir, text_dir, test_list, test_label, test_sen_attr, config.moda_list,
                    config.d_ensemble, config.d_ensemble_text, config.text_mode, config.loss_mode, class_ids, config.text_max_segs, config.img_num_tokens)  # pyright: ignore[reportArgumentType]


        if config.sampling_mode is not None:
            train_weights = dataset_sampling_weights(config.sampling_mode, train_sen_attr, train_label, weight_power=config.weight_power)
            print(f"Using {config.sampling_mode} sampling strategy for training data.")
            print("Sample weights range: [{:.4f}, {:.4f}]".format(min(train_weights), max(train_weights)))
            sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

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
        
            
        if config.ensemble_loss == "CE":
            ense_loss = CrossEntropyLoss(label_smoothing=config.label_smooth)
        elif config.ensemble_loss == "Focal":
            ense_loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)
        else:
            raise ValueError("Unknown ensemble loss: %s"%config.ensemble_loss)

        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.scheduler == "cosine":
            num_warmup_steps = int(config.warm_epochs * len(loader_train))
            num_training_steps = int(config.nb_epochs * len(loader_train))
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        else:
            scheduler = None
        model = TrainingModel(fold_id, net, ense_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler)

        if config.mode == ENSEMBLE_TRAIN:
            model.train()
            model.test()
        else:
            model.test()

