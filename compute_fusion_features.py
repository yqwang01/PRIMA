from config import Config, FUSION_TESTING
import os, random, argparse
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, help="Override GPU index (e.g. '0')")
    parser.add_argument("--moda", type=str, default=None, choices=["B", "ScS", "W"],
                        help="Modality — appended to output_dir and feature_dir, sets config.moda")
    parser.add_argument("--snapshot_dir", type=str, default=None,
                        help="Full path to the finetuned snapshot dir (overrides config.output_dir + moda suffix)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Full path to save extracted features (overrides config.feature_dir + moda suffix)")
    args = parser.parse_args()

    config = Config(FUSION_TESTING)

    if args.gpu is not None:
        config.gpu = args.gpu
    if args.moda is not None:
        config.moda = args.moda
        if args.snapshot_dir is None:
            config.output_dir = config.output_dir + "_" + args.moda
    if args.snapshot_dir is not None:
        config.output_dir = args.snapshot_dir
    if args.feature_dir is not None:
        config.feature_dir = args.feature_dir

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    random.seed(config.seed)
    np.random.seed(config.seed)

    import torch
    from data_loader import dataset_split_feature, DictDataset, process_json
    from torch.utils.data import DataLoader
    from models.img_txt_fusion import ImgTxtFusion
    from tqdm import tqdm
    from utils import load_trainable_checkpoint

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if config.cuda else "cpu")

    # full_list, full_moda_indices = dataset_split( # type: ignore
    #     config.train_csv_path, 7, config.moda, config.site, for_json=True) # type: ignore

    pat_list, test_list, test_label, test_sen_attr, test_var = dataset_split_feature(config.data_dir, config.train_csv_path, config.num_var, config.moda, # type: ignore
            var_idx=config.var_idx, sen_attr_idx=config.sen_attr_idx, for_json=False, avail_moda=config.avail_moda)
    metadata_dict = process_json(pat_list, config.metadata_path, config.knowledge_path, config.attr_list, config.knowledge_usage, config.label_dict, config.class_penalty, config.merge_texts)

    dataset_test = DictDataset(config.data_dir, test_list, test_label, metadata_dict, test_sen_attr, config.moda, config.moda_list,
                                config.multi_moda, config.input_size, False, False, text_mode="text",
                                image_mean=config.image_mean, image_std=config.image_std)
    loader_test = DataLoader(dataset_test, 
                             batch_size=1,
                             shuffle=False,
                             pin_memory=config.pin_mem,
                             num_workers=config.num_cpu_workers
                             )
    print("Test dataset size:", len(dataset_test))


    for fold_id in range(config.folds):
        print("\n-----------Fold: {}------------".format(fold_id))
        # _, val_list, test_list = k_fold_cross_validation_json(
        #     full_list, config.data_dir, config.moda, full_moda_indices, config.multi_moda, fold_id, config.folds)
        # test_metadata_dict = process_json(val_list, config.metadata_path, config.knowledge_path, config.attr_list, config.knowledge_usage)

        # dataset_test = DictDataset(config.data_dir, test_list, test_metadata_dict, config.moda, config.moda_list,
        #                             config.multi_moda, config.input_size, False, False)
        # loader_test = DataLoader(
        #     dataset_test, batch_size=1, shuffle=False, num_workers=config.num_cpu_workers,
        #     collate_fn=raw_collate_fn, pin_memory=config.pin_mem)
        if config.visual_pretrained is not None:
            visual_pretrained = config.visual_pretrained + '_fold_{}.pth'.format(fold_id)
        else:
            visual_pretrained = None
        net = ImgTxtFusion(config, config.vision_encoder, config.text_encoder, visual_pretrained)
        net = net.to(device)
        load_trainable_checkpoint(net, os.path.join(config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=fold_id)))
        nb_batch = len(loader_test)
        pbar = tqdm(total=nb_batch, desc="Test", mininterval=5)
        # predict_dict, truth_dict, count_dict, ensemble_dict = {}, {}, {}, {}
        # case_correct, img_correct = 0, 0
        # name_list, predict_list, label_list = [], [], []

        with torch.no_grad():
            net.eval() # type: ignore
            for batch in loader_test:
                pbar.update()
                inputs = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                local_text = batch['local_text']
                attr_label = batch['attr_label'].to(device, non_blocking=True)
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                name = batch['name'][0]
                patient = name.split('/')[0]
                if len(name.split('/')) > 2:
                    modality = name.split('/')[1]
                else:
                    modality = ""
                filename = name.split('/')[-1]
                os.makedirs(os.path.join(config.feature_dir, 'fold_{}'.format(fold_id), patient, modality), exist_ok=True)

                return_dict = net(inputs, text=local_text)
                feat = return_dict['img_1_features'].cpu().numpy()
                if config.vision_encoder == "DINOv3":
                    assert feat.shape == (1, 197, 768)
                elif config.vision_encoder == "EVA_vit_g":
                    assert feat.shape == (1, 257, 768)
            
                npz_file = os.path.join(config.feature_dir, 'fold_{}'.format(fold_id), patient, modality, filename)
                np.savez_compressed(npz_file, img=feat)

