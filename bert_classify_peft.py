import json
import os
import random
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from data_loader import (
    ClinicalTextDataset,
    dataset_split_ensemble,
    k_fold_cross_validation_ensemble,
    process_json,
    dataset_sampling_weights,
)
from utils import (
    compute_auc, 
    equity_scaled_accuracy, 
    equity_scaled_AUC, 
    equity_class_attr_scaled_f1,
    multiclass_equalized_odds, 
    multiclass_demographic_parity
)
from safetensors.torch import load_file

# --------------------------- #
#      Config: General       #
# --------------------------- #
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
MODEL_NAME = "Simonlee711/Clinical_ModernBERT" # "emilyalsentzer/Bio_ClinicalBERT", 
# "bert-base-uncased", "Simonlee711/Clinical_ModernBERT"， ? "thomas-sounack/BioClinical-ModernBERT-base"
RESUME_FROM_PRETRAINING = True
RESUME_MODEL_DIR = os.environ.get("PRIMA_TEXT_PRETRAIN_DIR", "outputs/PAD-UFES-20/bert_pretrain")
train_csv_path = os.environ.get("PRIMA_PAD_TRAIN_CSV", "data/PAD-UFES-20/datalist_sorted.csv")
site = None  # 'India', 'US', None
avail_moda = 0 # 5, 0
sen_attr_idx = -1
group_idx = 2
group_split = False
data_dir = os.environ.get("PRIMA_PAD_DATA_DIR", "data/PAD-UFES-20/organized_images_npz")
metadata_path = os.environ.get("PRIMA_PAD_METADATA", "data/PAD-UFES-20/metadata_full.json")
knowledge_path = os.environ.get("PRIMA_KNOWLEDGE_JSON", None)
num_classes = 6 # 2, 6
label_dict = {"BCC": 0, "SCC": 1, "MEL": 2, "ACK": 3, "SEK": 4, "NEV": 5}
metric = "overall_f1"
attr_list = ["Age", "Gender", "Background Father", "Background Mother",
                 "Smoke", "Drink", "Pesticide", "Has Piped Water", "Has Sewage System", 
                 "Skin Cancer History", "Cancer History", "Fitzpatrick",
                 "Region", "Horizontal Diameter", "Vertical Diameter", "Itch", "Grew", "Hurt", "Changed", "Bleed", "Elevation"
            ]
knowledge_usage = False
merge_texts = False
use_llm = False
output_dir = os.environ.get("PRIMA_TEXT_CLASSIFY_DIR", "outputs/PAD-UFES-20/bert_classify_peft")
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 30
WARMUP_RATIO = 0.1
LR = 1e-5
WEIGHT_DECAY = 0.1
LOGGING_STEPS = 100
folds = 5
SEED = 0
USE_PEFT = True
USE_WEIGHTED_SAMPLER = True
SAMPLING_MODE = "class"  # Sampling mode: None, "group", "balanced", "class"
WEIGHT_POWER = 1  # Power for sampling weights
EQUITY_ALPHA = 1.0  # Alpha for equity-scaled metrics
COMPUTE_FAIRNESS_METRICS = True  # Whether to compute fairness metrics based on sens_attr
GROUP_WISE_METRICS = True  # Whether to compute detailed metrics for each group (requires COMPUTE_FAIRNESS_METRICS=True)

# --------------------------- #
#      Config: PEFT (LoRA)    #
# --------------------------- #
PEFT_CONFIG = {
    "task_type": TaskType.SEQ_CLS,       # Task type: sequence classification
    "r": 8,                               # Rank of the LoRA decomposition
    "lora_alpha": 8,                    # Scaling factor
    "lora_dropout": 0.2,                 # Dropout probability
    "target_modules": ["query", "value"] # Which modules to apply LoRA on
}

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'hyperparameter.json'), 'w') as f:
    json.dump({
        "model_name": MODEL_NAME,
        "data_dir": data_dir,
        "metadata_path": metadata_path,
        "train_csv_path": train_csv_path,
        "avail_moda": avail_moda,
        "sen_attr_idx": sen_attr_idx,
        "group_idx": group_idx,
        "group_split": group_split,
        "knowledge_path": knowledge_path,
        "label_dict": label_dict,
        "num_classes": num_classes,
        "metric": metric,
        "attr_list": attr_list,
        "knowledge_usage": knowledge_usage,
        "merge_texts": merge_texts,
        "use_llm": use_llm,
        "output_dir": output_dir,
        "resume_from_pretraining": RESUME_FROM_PRETRAINING,
        "resume_model_dir": RESUME_MODEL_DIR,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": LOGGING_STEPS,
        "folds": folds,
        "seed": SEED,
        "use_peft": USE_PEFT,
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "sampling_mode": SAMPLING_MODE,
        "weight_power": WEIGHT_POWER,
        "equity_alpha": EQUITY_ALPHA,
        "compute_fairness_metrics": COMPUTE_FAIRNESS_METRICS,
        "group_wise_metrics": GROUP_WISE_METRICS,
        "peft_config": PEFT_CONFIG
    }, f, indent=4)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

full_list, full_label, full_var, full_sens_attr, full_moda_indices, full_group = dataset_split_ensemble( # type: ignore
                            train_csv_path, 0, -1, sen_attr_idx, group_idx, site, False, avail_moda)

# --------------------------- #
#      Metrics Function       #
# --------------------------- #
def filter_missing_metadata(patient_list, label_list, sens_attr_list, metadata_dict):
    filtered_patients = []
    filtered_labels = []
    filtered_sens_attrs = [] if sens_attr_list is not None and len(sens_attr_list) > 0 else []
    dropped = []

    has_sens_attr = sens_attr_list is not None and len(sens_attr_list) > 0
    for idx, patient_id in enumerate(patient_list):
        if patient_id not in metadata_dict:
            dropped.append(patient_id)
            continue
        filtered_patients.append(patient_id)
        filtered_labels.append(label_list[idx])
        if has_sens_attr:
            filtered_sens_attrs.append(sens_attr_list[idx]) # type: ignore

    if dropped:
        print(f"Warning: dropped {len(dropped)} samples missing from metadata. Examples: {dropped[:5]}")

    return filtered_patients, filtered_labels, filtered_sens_attrs


def normalize_int_list(values):
    normalized = []
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value.tolist()
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) > 0 else -1
        normalized.append(int(value))
    return normalized


def create_compute_metrics(sens_attr_list=None, group_wise=False, equity_alpha=1.):
    """
    Create a compute_metrics function that supports group-wise evaluation based on sens_attr
    
    Args:
        sens_attr_list: List of sensitive attributes. If None, group-wise metrics are not computed
        group_wise: Whether to compute detailed metrics for each group
    """
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        pred_probs = pred.predictions  # Probability predictions for computing AUC and other metrics

        # Compute macro f1 (overall)
        f1_macro = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        auc = compute_auc(pred_probs, labels, num_classes=len(np.unique(labels)))
        bacc = balanced_accuracy_score(labels, preds)

        # Compute per-class F1
        f1_per_class = f1_score(labels, preds, average=None, labels=np.unique(labels)) # type: ignore

        metrics = {}
        for i in range(len(np.unique(labels))):
            metrics[f"f1_class_{i}"] = f1_per_class[i] # pyright: ignore[reportArgumentType, reportIndexIssue], reportIndexIssue
        metrics["overall_f1"] = f1_macro
        metrics["overall_accuracy"] = acc
        metrics["overall_auc"] = auc
        metrics["overall_bacc"] = bacc
        
        # Compute fairness metrics if sens_attr is provided
        if sens_attr_list is not None and len(sens_attr_list) > 0:
            sens_attr = np.array(sens_attr_list)
            
            # Compute fairness metrics
            metrics["overall_ES_acc"] = equity_scaled_accuracy(pred_probs, labels, sens_attr, alpha=equity_alpha)
            metrics["overall_ES_AUC"] = equity_scaled_AUC(pred_probs, labels, sens_attr, alpha=equity_alpha, num_classes=num_classes)
            metrics["overall_ES_Class_Attr_Scaled_F1"] = equity_class_attr_scaled_f1(pred_probs, labels, sens_attr, alpha=equity_alpha)
            metrics["overall_EO"] = multiclass_equalized_odds(pred_probs, labels, sens_attr)
            metrics["overall_DP"] = multiclass_demographic_parity(pred_probs, labels, sens_attr)
            
            # Compute detailed metrics for each group if group_wise=True
            if group_wise:
                for one_attr in np.unique(sens_attr):
                    idx = sens_attr == one_attr
                    attr_preds = preds[idx]
                    attr_labels = labels[idx]
                    attr_probs = pred_probs[idx]
                    
                    attr_f1_macro = f1_score(attr_labels, attr_preds, average="macro")
                    attr_f1_per_class = f1_score(attr_labels, attr_preds, average=None, labels=list(range(num_classes)), zero_division="warn")  # type: ignore
                    metrics[f"attr_{one_attr}_f1"] = attr_f1_macro
                    for class_idx, class_f1 in enumerate(attr_f1_per_class): # type: ignore
                        metrics[f"attr_{one_attr}_f1_class_{class_idx}"] = float(class_f1) # type: ignore
                    metrics[f"attr_{one_attr}_accuracy"] = accuracy_score(attr_labels, attr_preds)
                    metrics[f"attr_{one_attr}_AUC"] = compute_auc(attr_probs, attr_labels, num_classes=num_classes)
        
        return metrics
    
    return compute_metrics


def create_data_collator(tokenizer):
    default_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def data_collator(features):
        model_features = []
        for feature in features:
            model_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"]
            }
            model_features.append(model_feature)
        return default_collator(model_features)
    
    return data_collator


class TrainerWithSampler(Trainer):
    def __init__(self, *args, sampler=None, sampler_generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_sampler = sampler
        # torch.Generator() to control reproducible sampling
        self._sampler_generator = sampler_generator or torch.Generator()

    def get_train_dataloader(self):
        # if user provided a sampler, build DataLoader using it (keeps default batch_size, collate_fn, etc.)
        train_dataset = self.train_dataset
        if self._external_sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            train_dataset, # type: ignore
            batch_size=self.args.train_batch_size,
            sampler=self._external_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )


# --------------------------- #
#      Cross-validation Loop  #
# --------------------------- #
all_fold_metrics = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for fold_id in range(folds):
    print("\n-----------Fold: {}------------".format(fold_id))
    train_list, val_list, train_label, val_label, train_sen_attr, val_sen_attr, train_var, val_var = k_fold_cross_validation_ensemble(
        full_list,
        full_label,
        full_sens_attr,
        full_var,
        full_group,
        fold_id,
        full_moda_indices,
        folds,
        group_split,
    )
    train_metadata_dict = process_json(train_list, metadata_path, knowledge_path, attr_list, knowledge_usage, None, 3, merge_texts, use_llm)
    val_metadata_dict = process_json(val_list, metadata_path, knowledge_path, attr_list, knowledge_usage, None, 3, merge_texts, use_llm)
    train_list, train_label, train_sen_attr = filter_missing_metadata(train_list, train_label, train_sen_attr, train_metadata_dict)
    val_list, val_label, val_sen_attr = filter_missing_metadata(val_list, val_label, val_sen_attr, val_metadata_dict)

    train_dataset = ClinicalTextDataset(train_list, train_metadata_dict, train_sen_attr, tokenizer, label_dict, False, MAX_LENGTH)
    val_dataset = ClinicalTextDataset(val_list, val_metadata_dict, val_sen_attr, tokenizer, label_dict, False, MAX_LENGTH)

    if RESUME_FROM_PRETRAINING:
        print(f"Resuming LoRA model from {RESUME_MODEL_DIR}")
        peft_config = PeftConfig.from_pretrained(RESUME_MODEL_DIR)
        if peft_config.task_type == TaskType.CAUSAL_LM:
            print("Detected legacy CAUSAL_LM LoRA adapter for BERT; overriding task_type to SEQ_CLS.")
        peft_config.task_type = TaskType.SEQ_CLS
        peft_config.inference_mode = False
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,   # pyright: ignore[reportArgumentType]
            num_labels=num_classes
        )
        model = get_peft_model(base_model, peft_config)
        safetensors_path = os.path.join(RESUME_MODEL_DIR, "adapter_model.safetensors")
        bin_path = os.path.join(RESUME_MODEL_DIR, "adapter_model.bin")
        adapter_state_dict = None

        if os.path.exists(safetensors_path):
            print(f"Loading weights from SAFETENSORS: {safetensors_path}")
            adapter_state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Loading weights from BIN: {bin_path}")
            adapter_state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter model file found in {RESUME_MODEL_DIR}")

        keys = model.load_state_dict(adapter_state_dict, strict=False)
        print(f"Loaded LoRA weights. Missing keys (expected for base model): {len(keys.missing_keys)}")
        
        model.train()
        model.print_trainable_parameters()
    else:
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

        # Apply PEFT LoRA configuration
        if USE_PEFT:
            lora_config = LoraConfig(
                task_type=PEFT_CONFIG["task_type"],
                r=PEFT_CONFIG["r"],
                lora_alpha=PEFT_CONFIG["lora_alpha"],
                lora_dropout=PEFT_CONFIG["lora_dropout"],
                target_modules=PEFT_CONFIG["target_modules"],
                inference_mode=False
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
        else:
            model = base_model

    # Set up output directory
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_id}")
    os.makedirs(fold_output_dir, exist_ok=True)

    # Define Trainer arguments
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="steps",  # pyright: ignore[reportCallIssue]
        logging_strategy="steps",
        save_strategy="steps",
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        save_steps=LOGGING_STEPS,
        eval_steps=LOGGING_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=5,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",      # Enable TensorBoard logging
    )

    # Prepare compute_metrics function
    # Extract sens_attr from validation set if fairness metrics are enabled
    val_sens_attr_list = None
    has_val_sens_attr = val_sen_attr is not None and len(val_sen_attr) == len(val_dataset)
    if COMPUTE_FAIRNESS_METRICS and has_val_sens_attr:
        val_sens_attr_list = normalize_int_list(val_sen_attr)
        if len(val_sens_attr_list) == len(val_dataset):
            print(f"Fold {fold_id} - Computing fairness metrics with {len(set(val_sens_attr_list))} groups")
        else:
            print(f"Fold {fold_id} - Warning: sens_attr list length mismatch, disabling fairness metrics")
            val_sens_attr_list = None
    elif COMPUTE_FAIRNESS_METRICS:
        print(f"Fold {fold_id} - No valid sens_attr configured, disabling fairness metrics")
    
    # Create compute_metrics function
    compute_metrics_fn = create_compute_metrics(
        sens_attr_list=val_sens_attr_list if COMPUTE_FAIRNESS_METRICS and val_sens_attr_list is not None else None,
        group_wise=GROUP_WISE_METRICS if COMPUTE_FAIRNESS_METRICS and val_sens_attr_list is not None else False,
        equity_alpha=EQUITY_ALPHA
    )

    # Prepare sampler (if needed)
    sampler = None
    sampler_generator = None
    if SAMPLING_MODE is not None:
        train_label_list = normalize_int_list(train_label)
        train_sens_attr_list = normalize_int_list(train_sen_attr) if train_sen_attr is not None and len(train_sen_attr) == len(train_label_list) else []
        if SAMPLING_MODE in {"group", "balanced"} and len(train_sens_attr_list) != len(train_label_list):
            print(f"Fold {fold_id} - Warning: {SAMPLING_MODE} sampling requires sens_attr; falling back to class sampling.")
            sampling_mode = "class"
        else:
            sampling_mode = SAMPLING_MODE
        
        # Compute sampling weights
        train_weights = dataset_sampling_weights(
            sampling_mode,
            train_sens_attr_list,
            train_label_list,
            weight_power=WEIGHT_POWER
        )
        print(f"Fold {fold_id} - Using {sampling_mode} sampling strategy for training data.")
        print(f"Sample weights range: [{min(train_weights):.4f}, {max(train_weights):.4f}]")
        
        # Create weighted random sampler
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(SEED + fold_id)  # Use different seed for each fold
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
            generator=sampler_generator
        )
    
    data_collator = create_data_collator(tokenizer)

    # Initialize Trainer (use TrainerWithSampler if sampler is provided)
    if sampler is not None:
        trainer = TrainerWithSampler(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            data_collator=data_collator,
            sampler=sampler,
            sampler_generator=sampler_generator
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            data_collator=data_collator,
        )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Fold {fold_id} Results: {metrics}")
    all_fold_metrics.append({"fold": fold_id, **metrics})

    # Save LoRA adapter and tokenizer
    model.save_pretrained(os.path.join(fold_output_dir, "peft_lora"))
    tokenizer.save_pretrained(os.path.join(fold_output_dir, "peft_lora"))

    # Save validation predictions
    preds = trainer.predict(val_dataset)
    pred_labels = np.array(preds.predictions).argmax(-1)
    true_labels = preds.label_ids
    patient_ids = [val_dataset[i]["patient_id"] for i in range(len(val_dataset))]
    sens_attrs = normalize_int_list(val_sen_attr) if val_sen_attr is not None and len(val_sen_attr) == len(val_dataset) else []

    txt_path = os.path.join(fold_output_dir, "val_predictions.txt")
    with open(txt_path, "w") as f:
        if len(sens_attrs) > 0:
            f.write("patient_id\ttrue_label\tpredicted_label\tsens_attr\n")
            for pid, true, pred, sens_attr in zip(patient_ids, true_labels, pred_labels, sens_attrs): # type: ignore
                f.write(f"{pid}\t{true}\t{pred}\t{sens_attr}\n")
        else:
            f.write("patient_id\ttrue_label\tpredicted_label\n")
            for pid, true, pred in zip(patient_ids, true_labels, pred_labels): # type: ignore
                f.write(f"{pid}\t{true}\t{pred}\n")

# --------------------------- #
#      Save Metrics Summary   #
# --------------------------- #
df_result = pd.DataFrame(all_fold_metrics).T
df_result.to_csv(os.path.join(output_dir, "crossval_metrics.csv"), header=True, index=True)
print("\n Cross-validation finished. Results saved to crossval_metrics.csv")
