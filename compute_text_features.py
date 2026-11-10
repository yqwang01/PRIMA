#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract text features from a PEFT/LoRA-finetuned BERT-like classifier:
- For each sample, encode the full concatenated text (full_text) and take a 1D vector.
- For each sample, encode each segment (text_list) separately and take a 1D vector per segment.
- Save results as NPZ files, plus a JSON sidecar for segment metadata (order and raw text).

Assumes the following utilities already exist in your codebase:
- ClinicalTextDataset
- dataset_split, k_fold_cross_validation_json, process_json
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed
)
from peft import PeftConfig, PeftModel, TaskType

# Your existing data utilities (these must be available in PYTHONPATH)
from data_loader import ClinicalTextDataset, dataset_split, process_json


# --------------------------- #
#           CONFIG            #
# --------------------------- #

MODEL_NAME = "Simonlee711/Clinical_ModernBERT"   # Base model name (same as training)
FOLDS = 1
SEED = 0
MAX_LENGTH = 512
BATCH_SIZE = 1             # Keep 1 to write features per sample/patient cleanly
NUM_WORKERS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional PEFT adapter directory.
# - Set to a valid adapter path to load LoRA/PEFT weights.
# - Set to None to use base MODEL_NAME directly.
OUTPUT_DIR = os.environ.get(
    "PRIMA_PAD_TEXT_ADAPTER",
    "outputs/PAD-UFES-20/bert_pretrain",
)

# Data & metadata (should match your training script)
TRAIN_CSV_PATH = os.environ.get(
    "PRIMA_PAD_TRAIN_CSV",
    "data/PAD-UFES-20/datalist_sorted.csv",
)
DATA_DIR = os.environ.get(
    "PRIMA_PAD_DATA_DIR",
    "data/PAD-UFES-20/organized_images_npz",
)
METADATA_PATH = os.environ.get(
    "PRIMA_PAD_METADATA",
    "data/PAD-UFES-20/metadata_full.json",
)
KNOWLEDGE_PATH = os.environ.get("PRIMA_KNOWLEDGE_JSON", None)
NUM_CLASSES = 6
SEN_ATTR_IDX = -1
LABEL_DICT = None
ATTR_LIST = [
    "Age", "Gender", "Background Father", "Background Mother",
    "Smoke", "Drink", "Pesticide", "Has Piped Water",
    "Has Sewage System", "Skin Cancer History", "Cancer History",
    "Fitzpatrick", "Region", "Horizontal Diameter", "Vertical Diameter",
    "Itch", "Grew", "Hurt", "Changed", "Bleed", "Elevation",
]

# Split/filter options (keep consistent with training)
SITE = None     # 'India' / 'US' / None
MODA = None
AVAIL_MODA = 0
USE_KNOWLEDGE = False
MERGE_TEXTS = True
USE_LLM = True

# Where to save extracted features
TEXT_FEATURE_DIR = os.environ.get(
    "PRIMA_TEXT_FEATURE_DIR",
    "features/PAD-UFES-20/bert_pretrain_seq",
)

# Pooling strategy for 1D vector: 'cls' (take [CLS]) or 'mean' (attention-masked mean pooling)
POOLING = "full" # "cls", "mean", "full"


# --------------------------- #
#       POOLING HELPERS       #
# --------------------------- #

def masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Attention-masked mean pooling over token embeddings.
    Args:
        last_hidden_state: [B, S, H]
        attention_mask:    [B, S]
    Returns:
        Pooled embeddings: [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, S, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
    return summed / counts


def get_feature_from_outputs(outputs, attention_mask, pooling: str = "cls") -> torch.Tensor:
    """
    Produce a 1D feature per sequence using either CLS or masked mean pooling.
    Args:
        outputs: HF model outputs (must contain hidden_states or last_hidden_state)
        attention_mask: [B, S]
        pooling: 'cls' or 'mean'
    Returns:
        [B, H]
    """
    # Prefer hidden_states if available
    hidden = None
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden = outputs.hidden_states[-1]  # [B, S, H]
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state  # [B, S, H]
    else:
        raise RuntimeError("Model outputs contain neither hidden_states nor last_hidden_state.")

    if pooling == "cls":
        return hidden[:, 0, :]  # [B, H], take token 0 as CLS
    elif pooling == "mean":
        return masked_mean_pooling(hidden, attention_mask)  # [B, H]
    elif pooling == "full":
        return hidden  # [B, S, H]
    else:
        raise ValueError(f"Invalid pooling method: {pooling}")


# --------------------------- #
#      MODEL LOADING (AUTO)   #
# --------------------------- #

def _resolve_base_loader(task_type: Optional[TaskType]):
    if task_type == TaskType.CAUSAL_LM:
        return AutoModelForCausalLM
    if task_type == TaskType.SEQ_CLS:
        return AutoModelForSequenceClassification
    return AutoModel


def _is_peft_adapter_path(path_or_name: Optional[str]) -> bool:
    if path_or_name is None:
        return False
    if not os.path.isdir(path_or_name):
        return False
    has_cfg = os.path.isfile(os.path.join(path_or_name, "adapter_config.json"))
    has_weights = os.path.isfile(os.path.join(path_or_name, "adapter_model.safetensors")) or \
                  os.path.isfile(os.path.join(path_or_name, "adapter_model.bin"))
    return has_cfg and has_weights


def load_text_model_for_infer(model_or_adapter_path: Optional[str], base_model_name: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
    """
    Auto-load text model for feature extraction.
    - If `model_or_adapter_path` is a PEFT adapter directory, load adapter + base.
    - Otherwise treat it as a plain HF model path/name and load base model directly.
    Returns:
        (model, source_type) where source_type in {"peft_adapter", "base_model"}
    """
    if _is_peft_adapter_path(model_or_adapter_path):
        assert model_or_adapter_path is not None
        peft_cfg = PeftConfig.from_pretrained(model_or_adapter_path)
        legacy_masked_lm_adapter = peft_cfg.task_type == TaskType.CAUSAL_LM
        if legacy_masked_lm_adapter:
            peft_cfg.task_type = TaskType.FEATURE_EXTRACTION
        base_path = peft_cfg.base_model_name_or_path or base_model_name
        if base_path is None:
            raise ValueError("Cannot resolve base model path for PEFT adapter loading.")
        base_loader = (
            AutoModelForMaskedLM
            if legacy_masked_lm_adapter
            else _resolve_base_loader(peft_cfg.task_type)
        )
        base_model = base_loader.from_pretrained(base_path)  # pyright: ignore[reportCallIssue]
        model = PeftModel.from_pretrained(
            base_model,
            model_or_adapter_path,
            config=peft_cfg,
            is_trainable=False,
        )
        source_type = "peft_adapter"
    else:
        model_path = model_or_adapter_path or base_model_name
        if model_path is None:
            raise ValueError("Both OUTPUT_DIR and MODEL_NAME are None; cannot load base model.")
        model = AutoModel.from_pretrained(model_path)
        source_type = "base_model"

    model.eval()
    model.to(DEVICE)
    return model, source_type


@torch.no_grad()
def encode_one(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, pooling: str = "cls") -> np.ndarray:
    """
    Encode a single sequence (batch size 1) and return a (H,) numpy vector.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    feat = get_feature_from_outputs(outputs, attention_mask, pooling=pooling)  
    return feat.squeeze(0).detach().cpu().numpy()  



# --------------------------- #
#    MAIN EXTRACTION ROUTINE  #
# --------------------------- #

def extract_and_save_for_split(
    tokenizer,
    patient_list: List[str],
    metadata_dict: Dict[str, dict],
    sens_attr_list: List[int],
    save_root: str,
    max_length: int = 512,
    pooling: str = "cls"
):
    """
    Extract features for a given fold and split and save them to disk.

    Directory layout:
        save_root/fold_{k}/{patient_id}/sample_{i}/
            - full_text_feat.npz     (feat: (H,))
            - segments_feats.npz     (feats: (N, H))
            - segments_meta.json     ([{"index": i, "text": "..."}])
    """

    # Auto-load PEFT adapter or plain base model
    model, source_type = load_text_model_for_infer(OUTPUT_DIR, base_model_name=MODEL_NAME)
    model_source = OUTPUT_DIR if OUTPUT_DIR is not None else MODEL_NAME
    print(f"Loaded text model source: {source_type} ({model_source})")
    os.makedirs(save_root, exist_ok=True)

    # Dataset configured to return per-segment encodings
    dataset = ClinicalTextDataset(
        patient_list=patient_list,
        metadata_dict=metadata_dict,
        sens_attr_list=sens_attr_list,
        tokenizer=tokenizer,
        label_dict=LABEL_DICT,
        return_text_list=True,
        max_length=max_length
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


    for batch in loader:
        # Because batch_size=1, take the first elements
        patient_id: str = batch["patient_id"][0]
        input_ids = batch["input_ids"][0]          # [S]
        attention_mask = batch["attention_mask"][0]

        save_dir = os.path.join(save_root, patient_id)
        os.makedirs(save_dir, exist_ok=True)

        # --- 1) Full text feature ---
        full_feat = encode_one(model, input_ids, attention_mask, pooling=pooling)   # (H,)
        np.savez_compressed(os.path.join(save_dir, "full_text_feat.npz"), feat=full_feat)

        # --- 2) Per-segment features ---
        seg_enc_list = batch["text_list_encodings"]  # list of encodings (each contains 'input_ids','attention_mask')
        seg_text_list = batch["text_list"]           # list of raw segment strings
        seg_feats: List[np.ndarray] = []
        seg_meta: List[Dict] = []

        for i, enc in enumerate(seg_enc_list):
            seg_ids = enc["input_ids"].squeeze(0)        # [S]
            seg_mask = enc["attention_mask"].squeeze(0)  # [S]
            feat_i = encode_one(model, seg_ids, seg_mask, pooling='cls')  # (H,)
            seg_feats.append(feat_i)
            seg_meta.append({"index": i, "text": seg_text_list[i]})

        if len(seg_feats) > 0:
            feats_arr = np.stack(seg_feats, axis=0)      
        else:
            raise ValueError("No segments found in batch.")

        np.savez_compressed(os.path.join(save_dir, "segments_feats.npz"), feats=feats_arr)
        with open(os.path.join(save_dir, "segments_meta.json"), "w", encoding="utf-8") as f:
            json.dump(seg_meta, f, ensure_ascii=False, indent=2)


def main():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use the same split pipeline as training
    test_list, test_label, test_sen_attr, test_var, test_indices, test_group = dataset_split( # type: ignore
        TRAIN_CSV_PATH, 0, MODA, var_idx=-1, sen_attr_idx=SEN_ATTR_IDX, for_json=True, avail_moda=AVAIL_MODA
    )

    # for fold_id in range(FOLDS):
    target_list = test_list
    sens_attr_list = test_sen_attr

    # Build metadata dict for this split (includes hierarchical text fields)
    metadata_dict = process_json(
            target_list,
            METADATA_PATH,
            KNOWLEDGE_PATH,
            attr_list=ATTR_LIST,                # Set to your desired subset of attributes or keep None
            knowledge_usage=USE_KNOWLEDGE,
            merge_texts=MERGE_TEXTS,
            use_llm=USE_LLM
    )

    extract_and_save_for_split(
            tokenizer=tokenizer,
            patient_list=target_list,
            metadata_dict=metadata_dict,
            sens_attr_list=sens_attr_list,  # pyright: ignore[reportArgumentType]
            save_root=TEXT_FEATURE_DIR,
            max_length=MAX_LENGTH,
            pooling=POOLING
    )

    print("\n>>> Text feature extraction finished. Saved to:", TEXT_FEATURE_DIR)


if __name__ == "__main__":
    main()
