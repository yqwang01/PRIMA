import os

# ==== Hyperparameters (Modify here for quick tuning) ====
MODEL_ID = "Simonlee711/Clinical_ModernBERT"      # Model checkpoint from HuggingFace Hub
DATA_PATH = os.environ.get("PRIMA_KNOWLEDGE_JSONL", "data/PAD-UFES-20/literature/skin_cancer_sop_dataset_global.jsonl")
OUTPUT_DIR = os.environ.get("PRIMA_TEXT_PRETRAIN_DIR", "outputs/PAD-UFES-20/bert_pretrain")
MAX_LENGTH = 512                                  # Max input token length
PER_DEVICE_BATCH = 16                             # Batch size per GPU
GRAD_ACCUM_STEPS = 8                              # Gradient accumulation steps
EPOCHS = 2500
EARLY_STOPPING_PATIENCE = 30                     # Early stopping patience
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.001                              # Weight decay for optimizer
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.2
SAVE_STEPS = 100
FP16 = True

# ==== Main Code ====
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import math
import random
import torch
import json


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'hyperparameter.json'), 'w') as f:
    json.dump({
        "model_id": MODEL_ID,
        "data_path": DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "max_length": MAX_LENGTH,
        "per_device_batch": PER_DEVICE_BATCH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "save_steps": SAVE_STEPS,
        "fp16": FP16,
        "seed": SEED,
    }, f, indent=4)


# 1. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
# model.gradient_checkpointing_enable()

# 2. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Load your dataset (no split, all data for training)
dataset = load_dataset("json", data_files=DATA_PATH)["train"] # type: ignore
dataset = Dataset.from_list(dataset) # type: ignore
split = dataset.train_test_split(test_size=0.1, seed=SEED) # 10% for validation
train_dataset = split["train"]
val_dataset = split["test"]

def tokenize_fn(examples):
    # If your data only has sentence_1, comment out sentence_2 as needed
    enc = tokenizer(examples["sentence_1"], 
                    examples.get("sentence_2", ""), 
                    truncation=True, 
                    # padding="max_length", 
                    max_length=MAX_LENGTH)
    # Default 'is_next' label to 1 (positive) if not present
    enc["next_sentence_label"] = examples.get("is_next", 1)
    return enc

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)


# 4. Data collator for MLM with token masking
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=8
)

# 5. TrainingArguments: enable TensorBoard logging
training_args = TrainingArguments(
    seed=SEED,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    prediction_loss_only=False,
    eval_strategy="steps",
    eval_accumulation_steps=4,
    save_strategy="steps",
    eval_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_mlm_accuracy",
    greater_is_better=True,
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    logging_steps=SAVE_STEPS,
    disable_tqdm=True,
    ddp_find_unused_parameters=False,
    report_to="tensorboard",      # Enable TensorBoard logging
)


def compute_metrics_trainer(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    pred_ids = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = (pred_ids[mask] == labels[mask]).sum() / mask.sum() if mask.sum() != 0 else 0.0 # type: ignore
    # Perplexity from eval_loss (HF Trainer will pass eval_loss to log_history, but not here)
    # So we only report accuracy, perplexity should be calculated after evaluation as shown below
    return {"eval_mlm_accuracy": acc}

# 6. Build and run Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics_trainer,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],  # patience=2 or 3 is common
)

trainer.train()

# 7. Save final model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Best checkpoint:", trainer.state.best_model_checkpoint)
# best_ckpt = trainer.state.best_model_checkpoint
# model = PeftModel.from_pretrained(model, best_ckpt) # type: ignore
# trainer.model = model

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"]) if "eval_loss" in eval_results else -1
print(f"Eval loss: {eval_results.get('eval_loss', -1):.4f}, Perplexity: {perplexity:.2f}, MLM accuracy: {eval_results.get('eval_mlm_accuracy', -1):.4f}")
