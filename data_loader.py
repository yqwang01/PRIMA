from importlib import metadata
from pydoc import text
from typing import Any
from numpy.random import zipf
from sympy import im
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import json
from numpy.core import defchararray
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms as T
import random
from functools import reduce
import torchvision.transforms.functional as F
from copy import deepcopy
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold



def dataset_split(train_csv_path, num_var, moda, var_idx=-1, sen_attr_idx=-1, group_idx=-1, site=None, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    print("The number of total valid patients in the dataset:", len(train_valid_cases))
    moda_dict = {"B": 0, "ScS": 1, "W": 2, "O": 3, "S": 4}
    if site is not None:
        if site == 'India':
            train_indices = ((train_valid_cases[:, moda_dict[moda]] == 1) & (np.char.find(train_valid_cases[:, avail_moda + 3].astype(str), "Aravind") >= 0)).tolist()  # pyright: ignore[reportOperatorIssue]  # pyright: ignore[reportOperatorIssue]
        elif site == 'US':
            train_indices = ((train_valid_cases[:, moda_dict[moda]] == 1) & (np.char.find(train_valid_cases[:, avail_moda + 3].astype(str), "Aravind") < 0)).tolist()
        else:
            raise ValueError("Invalid site specified. Choose either 'India' or 'US'.")
    else:
        if avail_moda != 0:
            train_indices = (train_valid_cases[:, moda_dict[moda]] == 1).tolist()
        else:
            train_indices = [True] * len(train_valid_cases)
    train_cases = train_valid_cases[:, avail_moda:]

    train_list, train_label, train_sen_attr, train_var, train_group = [], [], [], [], []

    for case in train_cases:
        train_list.append(case[0])
        train_label.append(case[1])
        if group_idx >= 0:
            train_group.append(case[group_idx])
        else:
            train_group.append(case[0])
        if sen_attr_idx >= 0:
            train_sen_attr.append(case[sen_attr_idx + 3])
        if num_var > 0:
            train_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
    return train_list, train_label, train_sen_attr, train_var, train_indices, train_group # type: ignore


def dataset_split_ensemble(train_csv_path, num_var, var_idx=-1, sen_attr_idx=-1, group_idx=-1, site=None, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values  # pyright: ignore[reportAttributeAccessIssue]
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    print("The number of total valid patients in training set:", len(train_valid_cases))
    if site is not None:
        if site == 'India':
            train_indices = (np.char.find(train_valid_cases[:, avail_moda + 3].astype(str), "Aravind") >= 0).tolist()  # pyright: ignore[reportOperatorIssue]
        elif site == 'US':
            train_indices = (np.char.find(train_valid_cases[:, avail_moda + 3].astype(str), "Aravind") < 0).tolist()
        else:
            raise ValueError("Invalid site specified. Choose either 'India' or 'US'.")
    else:
        train_indices = None
    train_cases = train_valid_cases[:, avail_moda:]
    
    train_list, train_label, train_var, train_sen_attr, train_group = [], [], [], [], []
    for case in train_cases:
        train_list.append(case[0])
        train_label.append(case[1])
        if group_idx >= 0:
            train_group.append(case[group_idx])
        else:
            train_group.append(case[0])
        if sen_attr_idx >= 0:
            train_sen_attr.append(case[sen_attr_idx + 3])
        if num_var > 0:
            train_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
    return train_list, train_label, train_var, train_sen_attr, train_indices, train_group # type: ignore


def dataset_split_feature(data_dir, train_csv_path, num_var, moda, var_idx=-1, sen_attr_idx=-1, for_json=False, avail_moda=5):
    
    train_csv_file = pd.read_csv(train_csv_path).values
    train_valid_cases = train_csv_file[train_csv_file[:, 1 + avail_moda] > -1]

    np.random.shuffle(train_valid_cases)
    print("The number of total valid patients in training set:", len(train_valid_cases))
    moda_dict = {"B": 0, "ScS": 1, "W": 2, "O": 3, "S": 4}
    if moda is not None:
        train_moda_indices = (train_valid_cases[:, moda_dict[moda]] == 1)
    else:
        train_moda_indices = [True] * len(train_valid_cases)
    train_cases = train_valid_cases[train_moda_indices, avail_moda:]

    pat_list, test_list, test_label, test_sen_attr, test_var = [], [], [], [], []
    for case in train_cases:
        patient = case[0]
        pat_list.append(patient)
        if moda is not None:
            for file_name in os.listdir(os.path.join(data_dir, patient, moda)):
                test_list.append(os.path.join(patient, moda, file_name))
                test_label.append(case[1])
                if sen_attr_idx >= 0:
                    test_sen_attr.append(case[sen_attr_idx + 3])
                if num_var > 0:
                    test_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))
        else:
            for file_name in os.listdir(os.path.join(data_dir, patient)):
                test_list.append(os.path.join(patient, file_name))
                test_label.append(case[1])
                if sen_attr_idx >= 0:
                    test_sen_attr.append(case[sen_attr_idx + 3])
                if num_var > 0:
                    test_var.append(case[var_idx + 3: var_idx + 3 + num_var].astype(np.float32))

    return pat_list, test_list, test_label, test_sen_attr, test_var


def _optional_take(values, indices):
    if values is None or len(values) == 0:
        return []
    return np.asarray(values)[indices].tolist()


def pad_nested_indices(
    full_label,
    full_group,
    fold_id,
    folds=5,
    inner_fraction=0.10,
    seed=260904,
):
    """Return deterministic PAD train/inner-validation/outer-test indices."""

    labels = np.asarray(full_label, dtype=int)
    groups = np.asarray(full_group).astype(str)
    sample_indices = np.arange(len(labels))

    if not 0 <= fold_id < folds:
        raise ValueError(f"fold_id must be in [0, {folds - 1}]")

    outer_splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    outer_splits = list(
        outer_splitter.split(sample_indices, labels, groups)
    )
    outer_train_indices, outer_test_indices = outer_splits[fold_id]

    inner_splits = round(1.0 / inner_fraction)
    if abs(inner_fraction - 1.0 / inner_splits) > 1e-12:
        raise ValueError(
            "PAD inner_fraction must be the reciprocal of an integer"
        )

    inner_splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed + 1000 + fold_id,
    )
    inner_train_relative, inner_val_relative = next(
        inner_splitter.split(
            outer_train_indices,
            labels[outer_train_indices],
            groups[outer_train_indices],
        )
    )

    train_indices = outer_train_indices[inner_train_relative]
    inner_val_indices = outer_train_indices[inner_val_relative]

    train_groups = set(groups[train_indices].tolist())
    val_groups = set(groups[inner_val_indices].tolist())
    test_groups = set(groups[outer_test_indices].tolist())

    if train_groups & val_groups:
        raise AssertionError(
            "Patient overlap between train and inner validation"
        )
    if train_groups & test_groups:
        raise AssertionError(
            "Patient overlap between train and outer test"
        )
    if val_groups & test_groups:
        raise AssertionError(
            "Patient overlap between inner validation and outer test"
        )

    return train_indices, inner_val_indices, outer_test_indices


def nested_pad_cross_validation(
    full_list,
    full_label,
    full_sen_attr,
    full_var,
    full_group,
    data_dir,
    fold_id,
    folds=5,
    inner_fraction=0.10,
    seed=260904,
):
    """PAD Stage 2 lesion-level nested split with image-level testing."""

    train_idx, val_idx, test_idx = pad_nested_indices(
        full_label=full_label,
        full_group=full_group,
        fold_id=fold_id,
        folds=folds,
        inner_fraction=inner_fraction,
        seed=seed,
    )

    train_list = _optional_take(full_list, train_idx)
    val_list = _optional_take(full_list, val_idx)
    test_case_list = _optional_take(full_list, test_idx)

    train_label = _optional_take(full_label, train_idx)
    val_label = _optional_take(full_label, val_idx)
    test_case_label = _optional_take(full_label, test_idx)

    train_var = _optional_take(full_var, train_idx)
    val_var = _optional_take(full_var, val_idx)
    test_case_var = _optional_take(full_var, test_idx)

    train_sen_attr = _optional_take(full_sen_attr, train_idx)
    val_sen_attr = _optional_take(full_sen_attr, val_idx)
    test_case_sen_attr = _optional_take(full_sen_attr, test_idx)

    test_list = []
    test_label = []
    test_var = []
    test_sen_attr = []

    for case_index, (case_id, label) in enumerate(
        zip(test_case_list, test_case_label)
    ):
        case_dir = os.path.join(data_dir, case_id)
        for file_name in sorted(os.listdir(case_dir)):
            test_list.append(os.path.join(case_id, file_name))
            test_label.append(label)
            if len(test_case_var) > 0:
                test_var.append(test_case_var[case_index])
            if len(test_case_sen_attr) > 0:
                test_sen_attr.append(test_case_sen_attr[case_index])

    print(f"Fold {fold_id}")
    print("Training lesions:", len(train_list))
    print("Inner-validation lesions:", len(val_list))
    print("Outer-test lesions:", len(test_case_list))
    print("Outer-test images:", len(test_list))

    return (
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
    )


def nested_pad_cross_validation_ensemble(
    full_list,
    full_label,
    full_sen_attr,
    full_var,
    full_group,
    fold_id,
    folds=5,
    inner_fraction=0.10,
    seed=260904,
):
    """PAD Stage 3 lesion-level nested split."""

    train_idx, val_idx, test_idx = pad_nested_indices(
        full_label=full_label,
        full_group=full_group,
        fold_id=fold_id,
        folds=folds,
        inner_fraction=inner_fraction,
        seed=seed,
    )

    return (
        _optional_take(full_list, train_idx),
        _optional_take(full_list, val_idx),
        _optional_take(full_list, test_idx),
        _optional_take(full_label, train_idx),
        _optional_take(full_label, val_idx),
        _optional_take(full_label, test_idx),
        _optional_take(full_sen_attr, train_idx),
        _optional_take(full_sen_attr, val_idx),
        _optional_take(full_sen_attr, test_idx),
        _optional_take(full_var, train_idx),
        _optional_take(full_var, val_idx),
        _optional_take(full_var, test_idx),
    )



def k_fold_cross_validation(full_list, full_label, full_sen_attr, full_var, full_group, data_dir, moda, moda_indices, multi_moda, fold_id, folds=5, group_split=True):

    full_list = np.array(full_list)
    full_label = np.array(full_label)
    full_group = np.array(full_group)

    full_sen_attr = (
        np.array(full_sen_attr)
        if full_sen_attr is not None and len(full_sen_attr) > 0
        else None
    )
    full_var = (
        np.array(full_var)
        if full_var is not None and len(full_var) > 0
        else None
    )
    moda_indices = (
        np.array(moda_indices)
        if moda_indices is not None
        else None
    )

    if group_split:
        gkf = GroupKFold(n_splits=folds, shuffle=False) # pyright: ignore
        split_pairs = list(gkf.split(X=full_list, y=full_label, groups=full_group))
        train_idx, val_idx = split_pairs[fold_id]
    else:
        cut_point_1 = int(fold_id / folds * len(full_list))
        cut_point_2 = int((fold_id + 1) / folds * len(full_list))
        train_idx = list(range(0, cut_point_1)) + list(range(cut_point_2, len(full_list)))
        val_idx = list(range(cut_point_1, cut_point_2))

    train_list = full_list[train_idx]
    val_list = full_list[val_idx]
    train_label = full_label[train_idx]
    val_label = full_label[val_idx]
    train_group = full_group[train_idx]
    val_group = full_group[val_idx]
    if moda_indices is not None:
        train_indices = moda_indices[train_idx]
        val_indices = moda_indices[val_idx]
    else:
        train_indices = None
        val_indices = None
    if full_var is not None:
        train_var = full_var[train_idx]
        val_var = full_var[val_idx]
    else:
        train_var, val_var = [], []
    if full_sen_attr is not None:
        train_sen_attr = full_sen_attr[train_idx]
        val_sen_attr = full_sen_attr[val_idx]
    else:
        train_sen_attr, val_sen_attr = [], []
    
    if not multi_moda and train_indices is not None:
        train_list = train_list[train_indices]
        val_list = val_list[val_indices]
        train_label = train_label[train_indices]
        val_label = val_label[val_indices]
        train_group = train_group[train_indices]
        val_group = val_group[val_indices]
        if full_var is not None:
            train_var = train_var[train_indices]
            val_var = val_var[val_indices] # type: ignore
        if full_sen_attr is not None:
            train_sen_attr = train_sen_attr[train_indices]
            val_sen_attr = val_sen_attr[val_indices] # type: ignore 

    train_list = train_list.tolist()
    val_list = val_list.tolist()
    train_label = train_label.tolist()
    val_label = val_label.tolist()
    train_group = train_group.tolist()
    val_group = val_group.tolist()
    if isinstance(train_var, np.ndarray):
        train_var = train_var.tolist()
    if isinstance(val_var, np.ndarray):
        val_var = val_var.tolist()
    if isinstance(train_sen_attr, np.ndarray):
        train_sen_attr = train_sen_attr.tolist()
    if isinstance(val_sen_attr, np.ndarray):
        val_sen_attr = val_sen_attr.tolist()

    print("The number of valid cases in training set:", len(train_list))
    print("The number of valid cases in validation set:", len(val_list))
    print("The number of unique patients in training set:", len(set(train_group)))
    print("The number of unique patients in validation set:", len(set(val_group)))
    print("The status of group split:", "Yes" if group_split else "No")
    print("Patient overlap:", len(set(train_group) & set(val_group)))

    test_list, test_label, test_sen_attr, test_var = [], [], [], []

    if not multi_moda:
        if full_sen_attr is not None and len(full_sen_attr) > 0 and full_var is not None and len(full_var) > 0:
            for case in zip(val_list, val_label, val_sen_attr, val_var):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_sen_attr.append(case[2])
                    test_var.append(case[3])
        elif full_sen_attr is not None and len(full_sen_attr) > 0:
            for case in zip(val_list, val_label, val_sen_attr):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_sen_attr.append(case[2])
        elif full_var is not None and len(full_var) > 0:
            for case in zip(val_list, val_label, val_var):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])
                    test_var.append(case[2])
        else:
            for case in zip(val_list, val_label):
                patient = case[0]
                for file_name in os.listdir(os.path.join(data_dir, patient, moda if moda is not None else "")):
                    test_list.append(os.path.join(patient, moda if moda is not None else "", file_name))
                    test_label.append(case[1])

    return train_list, val_list, test_list, train_label, val_label, test_label, train_var, val_var, test_var, train_sen_attr, val_sen_attr, test_sen_attr


def k_fold_cross_validation_ensemble(full_list, full_label, full_sen_attr, full_var, full_group, fold_id, indices=None, folds=5, group_split=True):

    full_list = np.array(full_list)
    full_label = np.array(full_label)
    full_group = np.array(full_group)

    full_sen_attr = (
        np.array(full_sen_attr)
        if full_sen_attr is not None and len(full_sen_attr) > 0
        else None
    )
    full_var = (
        np.array(full_var)
        if full_var is not None and len(full_var) > 0
        else None
    )
    moda_indices = (
        np.array(indices)
        if indices is not None and len(indices) > 0
        else None
    )

    if group_split:
        gkf = GroupKFold(n_splits=folds, shuffle=False) # pyright: ignore
        split_pairs = list(gkf.split(X=full_list, y=full_label, groups=full_group))
        train_idx, val_idx = split_pairs[fold_id]
    else:
        cut_point_1 = int(fold_id / folds * len(full_list))
        cut_point_2 = int((fold_id + 1) / folds * len(full_list))
        train_idx = list(range(0, cut_point_1)) + list(range(cut_point_2, len(full_list)))
        val_idx = list(range(cut_point_1, cut_point_2))

    train_list = full_list[train_idx]
    val_list = full_list[val_idx]
    train_label = full_label[train_idx]
    val_label = full_label[val_idx]
    train_group = full_group[train_idx]
    val_group = full_group[val_idx]
    if moda_indices is not None:
        train_indices = moda_indices[train_idx]
        val_indices = moda_indices[val_idx]
    else:
        train_indices = None
        val_indices = None
    if full_var is not None:
        train_var = full_var[train_idx]
        val_var = full_var[val_idx]
    else:
        train_var, val_var = [], []
    if full_sen_attr is not None:
        train_sen_attr = full_sen_attr[train_idx]
        val_sen_attr = full_sen_attr[val_idx]
    else:
        train_sen_attr, val_sen_attr = [], []
    
    if train_indices is not None:
        train_list = train_list[train_indices]
        val_list = val_list[val_indices]
        train_label = train_label[train_indices]
        val_label = val_label[val_indices]
        train_group = train_group[train_indices]
        val_group = val_group[val_indices]
        if full_var is not None:
            train_var = train_var[train_indices]
            val_var = val_var[val_indices] # type: ignore
        if full_sen_attr is not None:
            train_sen_attr = train_sen_attr[train_indices]
            val_sen_attr = val_sen_attr[val_indices] # type: ignore 

    train_list = train_list.tolist()
    val_list = val_list.tolist()
    train_label = train_label.tolist()
    val_label = val_label.tolist()
    train_group = train_group.tolist()
    val_group = val_group.tolist()
    if isinstance(train_var, np.ndarray):
        train_var = train_var.tolist()
    if isinstance(val_var, np.ndarray):
        val_var = val_var.tolist()
    if isinstance(train_sen_attr, np.ndarray):
        train_sen_attr = train_sen_attr.tolist()
    if isinstance(val_sen_attr, np.ndarray):
        val_sen_attr = val_sen_attr.tolist()

    
    return train_list, val_list, train_label, val_label, train_sen_attr, val_sen_attr, train_var, val_var


def dataset_sampling_weights(sample_mode, sens_attr_list, label_list, weight_power=1.0):

    num_labels = len(set(label_list))
    num_groups = len(set(sens_attr_list))

    if sample_mode == "group" or sample_mode == "balanced":
        group_array = deepcopy(sens_attr_list)
        if sample_mode == "balanced":
            group_array = (np.asarray(group_array) * num_labels + np.asarray(label_list)).tolist()
        
    elif sample_mode == "class":
        group_array = label_list
    
    else:
        raise ValueError("Invalid sample mode. Choose from 'group', 'balanced', or 'class'.")
    
    group_array = torch.LongTensor(group_array)
    if sample_mode == 'group':
        group_counts = (torch.arange(num_groups).unsqueeze(1) == group_array).sum(1).float()
    elif sample_mode == 'balanced':
        group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1) == group_array).sum(1).float()
    elif sample_mode == 'class':
        group_counts = (torch.arange(num_labels).unsqueeze(1) == group_array).sum(1).float()
    else:
        raise ValueError("Invalid sample mode. Choose from 'group', 'balanced', or 'class'.")

    group_weights = [1.0 / (x.item() ** weight_power) for x in group_counts]
    sample_weights = [group_weights[int(i)] for i in group_array]
    return sample_weights


def to_onehot(y, penalty=3,num_classes=None, dtype=np.float32):
    y = np.asarray(y, dtype=np.int64).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    onehot = np.eye(num_classes, dtype=dtype)[y].flatten()
    onehot = onehot * penalty
    return onehot.tolist()


def process_json(id_list, dataset_path, knowledge_bank_path, attr_list=None, knowledge_usage=False, label_dict=None, penalty=3, merge_texts=True, use_llm=False):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if knowledge_bank_path is not None:
        with open(knowledge_bank_path, 'r', encoding='utf-8') as f:
            knowledge_bank = json.load(f)
    else:
        knowledge_bank = None

    # def is_valid_attribute(original_value):
    #     return original_value not in [None, "", "unknown", -1, "-1"]

    result_dict = {}

    for sid in id_list:
        sample = dataset.get(sid)
        if not sample:
            continue
        item_result = {"label": sample["Label"] if "Label" in sample else sample["label"], "text": {}, "attr_label": []}

        for type_name, attr_dict in sample.items():
            if type_name == "label" or type_name == "Label":
                if label_dict is not None:
                    item_result["text"]["label"] = "This is a " + label_dict[attr_dict] + " case."
                    y = list(label_dict.keys()).index(attr_dict)
                    item_result["attr_label"].extend(to_onehot(y, penalty=penalty, num_classes=len(label_dict.keys())))
                continue

            merged_texts = []
            for attr_name, attr_content in attr_dict.items():
                if attr_list is not None and attr_name not in attr_list:
                    continue
                
                if attr_name == "Corneal Exam Note":
                    original_value = attr_content.get("Original Value")
                    cleaned_text_traditional = attr_content.get("Cleaned Text via Traditional", "")
                    cleaned_text_llm = attr_content.get("Cleaned Text via LLM", "")
                    if use_llm:
                        textual_desc = cleaned_text_llm
                    else:
                        textual_desc = cleaned_text_traditional
                    attr_label = None
                else:
                    try:
                        original_value = attr_content.get("Original Value")
                        textual_desc = attr_content.get("Textual Description", "")
                        attr_label = attr_content.get("Attribute Label", None)
                    except AttributeError:
                        original_value = None
                        attr_name = str(type_name)
                        textual_desc = str(attr_content)
                        attr_label = None

                # if not is_valid_attribute(original_value):
                #     continue

                if knowledge_bank is not None:
                    kb_text = knowledge_bank.get(attr_name, "")
                else:
                    kb_text = ""
                if knowledge_usage:
                    if (original_value is not None) and (isinstance(original_value, (float, int))) and (int(original_value) == -1):
                        full_desc = f"{textual_desc}".strip()
                    else:
                        full_desc = f"{textual_desc} {kb_text}".strip()
                else:
                    full_desc = f"{textual_desc}".strip()

                if merge_texts:
                    merged_texts.append(f"{attr_name}: {full_desc}\n")
                else:
                    item_result["text"][attr_name] = f"{full_desc}\n"
                if attr_label is not None:
                    item_result["attr_label"].append(attr_label)
        
            if merge_texts and len(merged_texts) > 0:
                item_result["text"][type_name] = "".join(merged_texts)
        result_dict[sid] = item_result

    return result_dict


def train_transform(image, img_size, mask=None, second_mask=None):
    color_jitter = T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6)
    image = color_jitter(image)

    if random.random() > 0.5:
        image = F.hflip(image)
        if mask is not None:
            mask = F.hflip(mask)
        if second_mask is not None:
            second_mask = F.hflip(second_mask)
    if random.random() > 0.5:
        image = F.vflip(image)
        if mask is not None:
            mask = F.vflip(mask)
        if second_mask is not None:
            second_mask = F.vflip(second_mask)
    
    angle = random.uniform(-30, 30)
    image = F.rotate(image, angle)
    if mask is not None:
        mask = F.rotate(mask, angle)
    if second_mask is not None:
        second_mask = F.rotate(second_mask, angle)

    width, height = image.size # type: ignore
    scale = (0.8, 1.0)
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    area = width * height
    target_area = random.uniform(scale[0], scale[1]) * area
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    crop_width = int(round((target_area * aspect_ratio) ** 0.5))
    crop_height = int(round((target_area / aspect_ratio) ** 0.5))
    if crop_width > width:
        crop_width = width
    if crop_height > height:
        crop_height = height
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)
    image = F.resized_crop(image, top, left, crop_height, crop_width, img_size)
    if mask is not None:
        mask = F.resized_crop(mask, top, left, crop_height, crop_width, img_size)
    if second_mask is not None:
        second_mask = F.resized_crop(second_mask, top, left, crop_height, crop_width, img_size)
    
    return image, mask, second_mask


def raw_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate

    labels = default_collate([b["label"] for b in batch])
    name = default_collate([b["name"] for b in batch])
    texts = [b["text"] for b in batch]
    text_batch = {k: [d[k] for d in texts] for k in texts[0].keys()}

    if "image" in batch[0]:
        images = default_collate([b["image"] for b in batch])
        return {
            "image": images,
            "text": text_batch,
            "label": labels,
            "name": name,
        }
    else:
        images_1 = default_collate([b["image_1"] for b in batch])
        images_2 = default_collate([b["image_2"] for b in batch])
        return {
            "image_1": images_1,
            "image_2": images_2,
            "text": text_batch,
            "label": labels,
            "name": name,
        }


class ClinicalTextDataset(Dataset):
    def __init__(self, patient_list, metadata_dict, sens_attr_list, tokenizer, label_dict=None, return_text_list=False, max_length=512):
        self.patient_list = patient_list
        self.metadata_dict = metadata_dict
        self.sens_attr_list = sens_attr_list
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.max_length = max_length
        self.return_text_list = return_text_list

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        pat_id = self.patient_list[idx]
        sens_attr = self.sens_attr_list[idx] if len(self.sens_attr_list) > 0 else -1
        metadata = self.metadata_dict[pat_id]
        labels = metadata["label"]
        if self.label_dict is not None:
            labels = self.label_dict[labels]
        name = pat_id
        full_text = ""
        text_list = []
        for key, value in metadata["text"].items():
            if key == "label":
                continue
            else:
                full_text += f"{value}"
                text_list.append(f"{value}")

        full_encoding = self.tokenizer(
                full_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
        input_ids = full_encoding['input_ids'].squeeze(0)
        attention_mask = full_encoding['attention_mask'].squeeze(0)

        text_list_encodings = []
        if self.return_text_list:
            for text in text_list:
                text_encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                text_list_encodings.append(text_encoding)

        if self.return_text_list:
            return_dict = {
                "labels": labels,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "text_list": text_list,
                "text_list_encodings": text_list_encodings,
                "sens_attr": sens_attr,
                "patient_id": name,
            }
        else:   
            return_dict = {
                    "labels": labels,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "sens_attr": sens_attr,
                    "patient_id": name,
            }
        
        return return_dict


class DictDataset(Dataset):
    def __init__(self, data_dir, patient_list, label_list, metadata_dict, sens_attr_list, moda, moda_list, 
                 multi_moda=False, img_size=(256, 256), pair=False, train=False, text_mode="text",
                 image_mean=None, image_std=None):
        self.data_dir = data_dir
        self.patient_list = patient_list
        self.label_list = label_list
        self.metadata_dict = metadata_dict
        self.sens_attr_list = sens_attr_list
        self.moda = moda
        self.moda_list = moda_list
        self.multi_moda = multi_moda
        self.img_size = img_size
        self.pair = pair
        self.train = train
        self.text_mode = text_mode
        image_mean = [0.485, 0.456, 0.406] if image_mean is None else image_mean
        image_std = [0.229, 0.224, 0.225] if image_std is None else image_std
        self.transforms_img = T.Compose([
                                # transforms.ToPILImage(),
                                T.Resize(self.img_size), 
                                T.ToTensor(),      
                                T.Normalize(mean=image_mean, std=image_std)
                            ])

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        pat_id = self.patient_list[idx]
        if not self.pair:
            patient = pat_id.split('/')[0]
            metadata = self.metadata_dict[patient]
        else:
            metadata = self.metadata_dict[pat_id]
        labels = self.label_list[idx]
        sens_attr = self.sens_attr_list[idx] if len(self.sens_attr_list) > 0 else None
        name = pat_id
        text_dict = {}
        attr_label = torch.tensor(metadata["attr_label"], dtype=torch.float32)
        for key, value in metadata["text"].items():
            text_dict[key] = str(value)
        if self.text_mode == "text":
            # return_global_text = text_dict["label"]
            return_text = '\n'.join(f"{v}" for k, v in text_dict.items() if v)
        elif self.text_mode == "dict":
            # return_global_text = text_dict["label"]
            return_text = text_dict
        else:
            raise ValueError(f"Unknown text_mode: {self.text_mode}")

        if not self.pair:
            img_path = os.path.join(self.data_dir, pat_id)
            img_file = np.load(img_path, allow_pickle=True)
            image = Image.fromarray(img_file['img'])
            img_file.close()
            if self.train:
                image, _, _ = train_transform(image, self.img_size)
            image = self.transforms_img(image)
            return_dict = {
                "image": image,
                "local_text": return_text,
                # "global_text": return_global_text,
                "label": labels,
                "attr_label": attr_label,
                "name": name,
                "sens_attr": sens_attr if len(self.sens_attr_list) > 0 else -1,
            }
        
        else:
            img_list = []
            if self.multi_moda:
                for moda in self.moda_list:
                    if os.path.exists(os.path.join(self.data_dir, pat_id, moda)):
                        moda_imgs = os.listdir(os.path.join(self.data_dir, pat_id, moda))
                        for moda_img in moda_imgs:
                            img_list.append(moda + '/' + moda_img)
            else:
                if self.moda is not None:
                    moda_imgs = os.listdir(os.path.join(self.data_dir, pat_id, self.moda))     
                else:
                    moda_imgs = os.listdir(os.path.join(self.data_dir, pat_id))
                for moda_img in moda_imgs:
                    if self.moda is not None:
                        img_list.append(self.moda + '/' + moda_img)
                    else:
                        img_list.append(moda_img)

            random.shuffle(img_list)
            img_path_1 = os.path.join(self.data_dir, pat_id, img_list[0])
            img_file_1 = np.load(img_path_1, allow_pickle=True)
            image_1 = Image.fromarray(img_file_1['img'])
            img_file_1.close()
            if self.train:
                image_1, _, _ = train_transform(image_1, self.img_size)
            image_1 = self.transforms_img(image_1)
            
            img_path_2 = os.path.join(self.data_dir, pat_id, img_list[-1])
            img_file_2 = np.load(img_path_2, allow_pickle=True)
            image_2 = Image.fromarray(img_file_2['img'])
            img_file_2.close()
            if self.train:
                image_2, _, _ = train_transform(image_2, self.img_size)
            image_2 = self.transforms_img(image_2)
        
            return_dict = {
                "image_1": image_1,
                "image_2": image_2,
                "local_text": return_text,
                # "global_text": return_global_text,
                "label": labels,
                "attr_label": attr_label,
                "name": name,
                "sens_attr": sens_attr if len(self.sens_attr_list) > 0 else -1,
            }
        
        return return_dict


class EnsembleDatasetV2(Dataset):
    """
    Minimal dataset, no masks, no collate_fn.
    Image features:
      img_feature_dir/{patient}/{modality}/*.npz   (key='img') -> avg within a modality -> (512,)
    Text features (choose ONE mode):
      text_feature_dir/fold_{fold_id}/{patient}/
        - full_text_feat.npz   (key='feat')        -> (768,)
        - segments_feats.npz   (key='feats')       -> (T,768), padded/truncated to (text_max_segs,768)

    Returns (depending on text_mode):
      text_mode == "full":
        (x_img[M,512], x_txt_full[768], y, patient_id)
      text_mode == "segments":
        (x_img[M,512], x_txt_segs[text_max_segs,768], y, patient_id)
    """
    def __init__(self,
                 img_feature_dir,
                 text_feature_dir,
                 patient_list,
                 label_list,
                 sen_attr_list,
                 moda_list,
                 d_ensemble: int = 512,
                 txt_dim: int = 768,
                 text_mode: str = "full",      # "full", "segments", "none"
                 loss_mode: str = "target",    # "target", "standard"
                 class_ids: list = [0, 1],
                 text_max_segs: int = 16,
                 img_num_tokens: int = 197,
                 ):
        super().__init__()
        self.img_feature_dir = img_feature_dir
        self.text_feature_dir = text_feature_dir
        self.patient_list = patient_list
        self.label_list = label_list
        self.sen_attr_list = sen_attr_list
        self.moda_list = moda_list
        self.d_ensemble = d_ensemble
        self.txt_dim = txt_dim
        self.text_mode = text_mode
        self.loss_mode = loss_mode
        self.class_ids = class_ids
        self.text_max_segs = text_max_segs
        self.img_num_tokens = img_num_tokens

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        raw_label = self.label_list[idx]
        sen_attr = self.sen_attr_list[idx] if len(self.sen_attr_list) > 0 else -1

        if self.loss_mode == "standard":
            label = self.class_ids[raw_label]
        else:
            label = raw_label

        # ---- image features: per-modality mean -> (M,512) ----
        if self.moda_list is not None:
            feat_array = np.zeros((len(self.moda_list), self.img_num_tokens, self.d_ensemble), dtype=np.float32)
        else:
            feat_array = np.zeros((1, self.img_num_tokens, self.d_ensemble), dtype=np.float32)
        pat_dir = os.path.join(self.img_feature_dir, patient_id)
        if self.moda_list is not None:
            for i, moda in enumerate(self.moda_list):
                moda_dir = os.path.join(pat_dir, moda)
                if os.path.isdir(moda_dir):
                    moda_feats = []
                    for fname in os.listdir(moda_dir):
                        if fname.endswith(".npz"):
                            fpath = os.path.join(moda_dir, fname)
                            data = np.load(fpath, allow_pickle=True)
                            v = np.asarray(data["img"], dtype=np.float32).squeeze()
                            data.close()
                            if v.shape[0] == self.img_num_tokens:
                                moda_feats.append(v)
                    if len(moda_feats) > 0:
                        feat_array[i] = np.mean(np.stack(moda_feats, axis=0), axis=0)
                    else:
                        raise ValueError(f"No image features found for modality: {moda} for patient: {patient_id}")
            # feat_array = np.reshape(feat_array, (-1, self.d_ensemble))
        else:
            if os.path.isdir(pat_dir):
                pat_feats = []
                for fname in os.listdir(pat_dir):
                    if fname.endswith(".npz"):
                        fpath = os.path.join(pat_dir, fname)
                        data = np.load(fpath, allow_pickle=True)
                        v = np.asarray(data["img"], dtype=np.float32).squeeze()
                        data.close()
                        if v.shape[0] == self.img_num_tokens:
                            pat_feats.append(v)
                if len(pat_feats) > 0:
                    feat_array = np.mean(np.stack(pat_feats, axis=0), axis=0, keepdims=True)
                else:
                    raise ValueError(f"No image features found for patient: {patient_id}")
            else:
                raise ValueError(f"No image features found for patient: {patient_id}")


        x_img = torch.from_numpy(feat_array).float()  # [M, N_tokens, D_ensemble]
        y = torch.tensor(label).long()

        # ---- text features (one mode only) ----
        base_txt = os.path.join(self.text_feature_dir, patient_id)

        if self.text_mode == "full":
            full_vec = None
            fpath = os.path.join(base_txt, "full_text_feat.npz")
            if os.path.isfile(fpath):
                d = np.load(fpath, allow_pickle=True)
                fv = np.asarray(d["feat"], dtype=np.float32)
                d.close()
                if fv.shape[1] == self.txt_dim:
                    full_vec = fv
                else:
                    raise ValueError(f"Text feature dimension mismatch: {fv.shape[0]} != {self.txt_dim}")
            if full_vec is None:
                raise ValueError(f"No text features found for patient: {patient_id}")
            x_txt_full = torch.from_numpy(full_vec).float()  # [1, 768]
            return (x_img, x_txt_full, y, sen_attr, patient_id)

        elif self.text_mode == "segments":
            segs_fixed = None
            spath = os.path.join(base_txt, "segments_feats.npz")
            if os.path.isfile(spath):
                d = np.load(spath, allow_pickle=True)
                feats = np.asarray(d["feats"], dtype=np.float32)
                d.close()
                if feats.ndim == 2 and feats.shape[1] == self.txt_dim and feats.shape[0] <= self.text_max_segs:
                    segs_fixed = feats
                else:
                    raise ValueError(f"Text feature dimension mismatch: {feats.shape[1]} != {self.txt_dim}")
            if segs_fixed is None:
                raise ValueError(f"No text features found for patient: {patient_id}")
            x_txt_segs = torch.from_numpy(segs_fixed).float() 
            return (x_img, x_txt_segs, y, sen_attr, patient_id)
    
        elif self.text_mode == "none":
            return (x_img, y, sen_attr, patient_id)

        else:
            raise ValueError(f"Unknown text_mode: {self.text_mode}")
        
