import os


def _env_path(name, default):
    return os.environ.get(name, default)


ENSEMBLE_TRAIN = 0
ENSEMBLE_TEST = 1
FUSION_PRETRAINING = 2
FUSION_FINETUNING = 3
FUSION_TESTING = 4


KERATITIS_DICT = {
    "data_dir": _env_path("PRIMA_AQUA_DATA_DIR", "data/AQUA/Day0-2Npz"),
    "train_csv_path": _env_path("PRIMA_AQUA_TRAIN_CSV", "data/AQUA/data_list.csv"),
    "num_classes": 2,
    "sen_attr_idx": 0,
    "group_idx": -1,
    "metadata_path": _env_path("PRIMA_AQUA_METADATA", "data/AQUA/full_data.json"),
    "attr_list": ["Corneal Exam Note",
                  "Stromal Infiltrate - Granular", "Stromal Infiltrate - Suppurative", 
                  "Stromal Infiltrate - Feathery", "Stromal Infiltrate - Purulent", 
                  "Stromal Infiltrate - Smooth Ring-shaped", "Hypopyon", "Satellite Lesions"],
    "class_penalty": 3,
    "text_max_segs": 6,
    "label_dict": {0: "Fungal Keratitis", 1: "Bacterial Keratitis"},
    "class_names": ["Fungal Keratitis", "Bacterial Keratitis"],
    "moda": "W",
    "moda_list": ["B", "ScS", "W"],
    "avail_moda": 5,
    # "multi_moda": True,
    "sampling_mode": "balanced",
    "peft_model_path": _env_path("PRIMA_AQUA_TEXT_ADAPTER", "outputs/AQUA/bert_pretrain"),
    
}

PAD_UFES_20_DICT = {
   "data_dir": _env_path("PRIMA_PAD_DATA_DIR", "data/PAD-UFES-20/organized_images_npz"),
   "train_csv_path": _env_path("PRIMA_PAD_TRAIN_CSV", "data/PAD-UFES-20/datalist_sorted.csv"),
   "num_classes": 6,
   "sen_attr_idx": -1,
   "group_idx": 2,
   "metadata_path": _env_path("PRIMA_PAD_METADATA", "data/PAD-UFES-20/metadata_full.json"),
   "attr_list": [# "Age", "Gender", "Background Father", "Background Mother",
                 # "Smoke", "Drink", "Pesticide", "Has Piped Water", "Has Sewage System", 
                 # "Skin Cancer History", "Cancer History", "Fitzpatrick",
                 # "Region", "Horizontal Diameter", "Vertical Diameter", "Itch", "Grew", "Hurt", "Changed", "Bleed", "Elevation"
                 "Region", "Horizontal Diameter", "Vertical Diameter", "Itch", "Grew", "Hurt", "Changed", "Bleed", "Elevation",
                 ],
   "class_penalty": 3,
   "label_dict": {"BCC": "Basal Cell Carcinoma", "SCC": "Squamous Cell Carcinoma", "MEL": "Melanoma", "ACK": "Actinic Keratosis", "SEK": "Seborrheic Keratosis", "NEV": "Nevus"},
   "class_names": ["Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Melanoma", "Actinic Keratosis", "Seborrheic Keratosis", "Nevus"],
   "text_max_segs": 4,
   "moda": None,
   "moda_list": None,
   "avail_moda": 0,
   # "multi_moda": False,
   "sampling_mode": "class",
   "peft_model_path": _env_path("PRIMA_PAD_TEXT_ADAPTER", "outputs/PAD-UFES-20/bert_pretrain"),
}


class Config:

    @staticmethod
    def _infer_img_num_tokens(img_feature_dir, default_num_tokens):
        feature_dir = str(img_feature_dir).lower()
        if "eva" in feature_dir:
            return 257
        if "vit_base" in feature_dir or "dino" in feature_dir:
            return 197
        return default_num_tokens

    def __init__(self, mode):
        assert mode in {ENSEMBLE_TRAIN, ENSEMBLE_TEST, FUSION_PRETRAINING, 
                        FUSION_FINETUNING, FUSION_TESTING}, "Unknown mode: %i"%mode

        self.dataset_options = "PAD-UFES-20" # "KERATITIS", "PAD-UFES-20"
        if self.dataset_options == "KERATITIS":
            self.dataset_dict = KERATITIS_DICT
        elif self.dataset_options == "PAD-UFES-20":
            self.dataset_dict = PAD_UFES_20_DICT
        else:
            raise ValueError("Unknown dataset options: {}".format(self.dataset_options))

        # System settings
        self.gpu = '0'
        self.cuda = True
        self.pin_mem = True
        self.num_cpu_workers = 20
        self.seed = 0
        self.split_seed = 260904
        # Dataset settings
        self.mode = mode
        self.data_dir = self.dataset_dict["data_dir"]
        self.train_csv_path = self.dataset_dict["train_csv_path"]
        self.folds = 5
        self.inner_fraction = 0.10
        self.input_size = (224, 224) # (224, 224), (448, 448)
        self.in_channel = 3
        self.num_classes = self.dataset_dict["num_classes"] # 2, 6
        self.label_dict = self.dataset_dict["label_dict"] if mode == FUSION_PRETRAINING else None
        # None, {0: "Normal", 1: "Fungal Keratitis", 2: "Bacterial Keratitis"}, 
        # {0: "Basal Cell Carcinoma", 1: "Squamous Cell Carcinoma", 2: "Melanoma", 3: "Actinic Keratosis", 4: "Seborrheic Keratosis", 5: "Nevus"}
        self.num_var = 0 # 7, 13, 24
        self.var_idx = -1
        self.use_var = False
        self.sen_attr_idx = self.dataset_dict["sen_attr_idx"] # 0, -1
        self.group_split = True # True, False
        self.group_idx = self.dataset_dict["group_idx"] # 0, -1
        self.metadata_path = self.dataset_dict["metadata_path"] 
        self.knowledge_path = None 
        self.attr_list = self.dataset_dict["attr_list"]
        self.class_penalty = self.dataset_dict["class_penalty"]
        self.merge_texts = False # True, False
        self.knowledge_usage = False # True, False
        self.imagenet_pretrained = True
        self.moda = self.dataset_dict["moda"] # 'B', 'ScS', 'W', 'O', 'S', None
        self.moda_list = self.dataset_dict["moda_list"] # ['B', 'ScS', 'W'], None
        self.avail_moda = self.dataset_dict["avail_moda"] # 5, 0
        self.multi_moda = True if self.dataset_options == "KERATITIS" and self.mode == FUSION_PRETRAINING else False
        self.site = None # 'India', 'US', None
        self.sampling_mode = self.dataset_dict["sampling_mode"] # "balanced", "group", "label", None
        self.weight_power = 1.0 # 1.0, 0.5
        self.equity_alpha = 1.0 # 1.0, 2.0, 3.0
        # Main - Fusion Backbone settings
        self.vision_encoder = "DINOv3" # "DINOv3", "EVA_vit_g", "ViT_base"
        self.dino_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m" # "facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vits16plus-pretrain-lvd1689m", "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.freeze_dino_vision = True # True, False
        self.text_encoder = "Simonlee711/Clinical_ModernBERT" # "emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased", "Simonlee711/Clinical_ModernBERT"
        self.peft_model_path = self.dataset_dict["peft_model_path"]
        self.visual_pretrained = None 
        self.trained_text_layers = None # []
        self.freeze_text_layers = True # True, False
        # For DINOv3/DINOv2 with built-in LoRA support
        self.dino_use_lora = True # True, False
        self.dino_lora_r = 32 # 16, 32
        self.dino_lora_alpha = 32 # 16, 32
        self.dino_lora_dropout = 0.2 # 0.1, 0.2
        self.dino_lora_target_modules = ["query", "value", "q_proj", "v_proj", "key", "k_proj"]
        # For EVA_vit_g
        self.eva_img_size = 224
        self.eva_patch_size = 14
        self.eva_drop_path_rate = 0.0
        self.eva_use_checkpoint = True
        self.eva_precision = "bf16"
        self.eva_raw_dim = 1408
        self.eva_project_dim = 768
        self.eva_train_mode = "lora" # "freeze_all", "last_n", "lora"
        self.eva_unfreeze_last_n_blocks = 4
        self.eva_lora_r = 32
        self.eva_lora_alpha = 32
        self.eva_lora_dropout = 0.1
        self.eva_lora_target_modules = ["qkv", "proj"]
        # For ViT_base
        self.vit_base_img_size = 224
        self.vit_base_pretrained_source = "local_ckpt" # "deit_url", "local_ckpt", "none"
        self.vit_base_pretrained_url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
        self.vit_base_pretrained_path = _env_path("PRIMA_VIT_BASE_CKPT", "checkpoints/vit_base.ckpt")
        self.vit_base_use_checkpoint = False
        self.vit_base_ckpt_layer = 0
        self.vit_base_drop_path_rate = 0.0
        self.vit_base_train_mode = "lora" # "freeze_all", "lora"
        self.vit_base_lora_r = 32
        self.vit_base_lora_alpha = 32
        self.vit_base_lora_dropout = 0.2
        self.vit_base_lora_target_modules = ["qkv"]
        self.d_model = 768 # 768, 512
        self.d_visual = 768 # 512, 768, 384
        self.image_mean = [0.5, 0.5, 0.5] if self.vision_encoder == "EVA_vit_g" else [0.485, 0.456, 0.406]
        self.image_std = [0.5, 0.5, 0.5] if self.vision_encoder == "EVA_vit_g" else [0.229, 0.224, 0.225]
        self.num_fusion_heads = 8 # 8, 4
        self.forward_dim_scale = 4 # 4, 8
        self.num_fusion_layers = 1 # 1, 2
        self.fusion_dropout = 0.1 # 0.1, 0.2
        # Loss settings
        self.clas_loss = "CE"
        self.ensemble_loss = "CE"
        self.temperature = 0.1
        self.img_cont_loss = "Unsupervised" # "Supervised", "Unsupervised", "AdaSP"
        self.global_clip_loss = "CLIP" # "CLIP", "Unsupervised"
        self.soft_clip_loss = "SoftCLIP" # "SoftCLIP", "SupervisedCLIP"
        self.local_clip_loss = "LocalCLIP" # "LocalCLIP", "DirectTokenPatchContrastive"
        self.enable_fusion = False # True, False

            
        if self.mode == ENSEMBLE_TRAIN or self.mode == ENSEMBLE_TEST:
            self.ense_model = "LLM" # "Linear", "LinearText", "SimpleSelfAttentionFusion", "LLM"
            self.d_ensemble = 768 # 512, 768
            self.img_num_tokens = 197 # 257, 197
            self.text_mode = "full" # "full", "segments", "none"
            self.text_max_segs = self.dataset_dict["text_max_segs"] # 4, 5
            self.d_ensemble_text = 768 
            self.num_layers_btransformer = 2
            self.simple_fusion_token_mode = "all" # "cls", "all"
            self.transformer_dropout = 0.5 # dropout_rate for SimpleSelfAttentionFusion
            self.transformer_attention_dropout = 0.5 # attention_dropout_rate
            self.mlp_hidden1 = 256 # FusionMLP hidden layer 1
            self.mlp_hidden2 = 16 # FusionMLP hidden layer 2
            self.mlp_dropout = 0.0 # FusionMLP dropout
            # For LLM Ensemble
            self.llm_base_model_path = "Qwen/Qwen3-1.7B"
            # "StanfordAIMI/CheXagent-2-3b", "OpenGVLab/InternVL3_5-1B", "lmms-lab/LLaVA-One-Vision-1.5-4B-Instruct", 
            # "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"
            self.noise_scale = 0 # 0.01, 0.05, 0.1
            self.use_patch_tokens = True # True, False
            self.use_lora = True # True, False
            self.lora_r = 8 # 16, 8
            self.lora_alpha = 8
            self.lora_dropout = 0.2
            self.feat_dropout = 0.2 # 0.1, 0.2
            self.separate_text_cls = True
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"], ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            self.downsample_rate = 2 # 2, 4
            self.padding = 0 # 0, 1
            self.downsample_rate_text = 4 # 2, 4
            self.class_names = self.dataset_dict["class_names"]
            self.loss_mode = "target" # "target", "standard"
            self.label_smooth = 0 # 0, 0.1
            self.scheduler = "cosine" # "cosine", "none"
            self.batch_size = 8
            self.nb_epochs_per_saving = 50
            self.warm_epochs = 10
            self.nb_epochs = 100
            # Optimizer
            self.lr = 3e-05
            self.weight_decay = 0.03
            self.lam_mag = 0
            self.metric = "f1" # "f1", "bacc", "acc"
            self.flood_level = 0
            self.img_feature_dir = _env_path("PRIMA_IMG_FEATURE_DIR", "features/PAD-UFES-20/fusion_finetuning_DINOv3_Bert")
            self.img_num_tokens = self._infer_img_num_tokens(self.img_feature_dir, self.img_num_tokens)
            self.text_feature_dir = _env_path("PRIMA_TEXT_FEATURE_DIR", "features/PAD-UFES-20/bert_pretrain_seq")
            self.output_dir = _env_path("PRIMA_OUTPUT_DIR", "outputs/PAD-UFES-20/LLMEnsemble_DINOv3_Bert_Qwen3-1.7B")
            self.out_file_path = os.path.join(self.output_dir, "output_test_" + self.site if self.site is not None else "output_test")
        
        elif self.mode == FUSION_PRETRAINING:
            self.batch_size = 32
            self.nb_epochs_per_saving = 100
            self.nb_epochs = 150
            # Optimizer
            self.lr = 3e-5
            self.warm_epochs = 15
            self.weight_decay = 1e-3
            self.lam_global = 0.3
            self.lam_local = 0.2
            self.lam_soft = 0.3
            self.scheduler = "cosine" # "cos
            self.output_dir = _env_path("PRIMA_OUTPUT_DIR", "outputs/PAD-UFES-20/fusion_pretraining_DINOv3_Bert")

        elif self.mode == FUSION_FINETUNING or self.mode == FUSION_TESTING:
            self.batch_size = 32
            self.nb_epochs_per_saving = 100
            self.nb_epochs = 200
            # Optimizer
            self.lr = 1e-5
            self.warm_epochs = 20
            self.weight_decay = 1e-3
            self.metric = "f1" # "f1", "bacc", "acc"
            self.pretrained_path = None
            self.lam_cls = 1 if self.pretrained_path is None else 0.9
            self.lam_global = 0
            self.lam_local = 0
            self.lam_soft = 0
            self.scheduler = "cosine" # "cosine", "none"
            self.gamma = 2
            self.alpha = 1
            self.output_dir = _env_path("PRIMA_OUTPUT_DIR", "outputs/PAD-UFES-20/fusion_finetuning_DINOv3_Bert")
            self.feature_dir = _env_path("PRIMA_FEATURE_DIR", "features/PAD-UFES-20/fusion_finetuning_DINOv3_Bert")
            self.out_file_path = os.path.join(self.output_dir, "output_test_" + self.site if self.site is not None else "output_test")

        else:
            raise ValueError("Unknown mode: {}".format(mode))
