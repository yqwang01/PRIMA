import torch
import torch.distributed as dist
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import os
import timm.models.hub as timm_hub
from peft import PeftModel


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a URL through timm's cache without importing LAVIS.

    This mirrors lavis.common.dist_utils.download_cached_file for EVA weights:
    rank 0 downloads the file, distributed workers wait, and the cached path is
    returned to all processes.
    """
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash=check_hash, progress=progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return cached_file


def save_trainable_checkpoint(model, path, epoch):
    """
    Saves only the parameters that participated in training (requires_grad=True), 
    filtering out frozen pre-trained parameters.
    
    This typically includes: LoRA weights, Fusion layers, Projection Heads, and Classifiers.
    """
    print(f"Saving trainable parameters to {path}...")
    
    # 1. Get the names of all parameters with requires_grad=True
    trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
    
    # 2. Filter these parameters from the state_dict
    # Note: If your model uses BatchNorm, its running_mean/var (buffers) usually need to be saved 
    # even if frozen. However, since your architecture mainly uses LayerNorm (which has no buffers), 
    # saving only the parameters is sufficient.
    trainable_state_dict = {
        k: v.cpu() for k, v in model.state_dict().items() 
        if k in trainable_keys
    }
    
    torch.save({
        "epoch": epoch,
        "model": trainable_state_dict
    }, path)
    print(f"Saved {len(trainable_state_dict)} keys. (Skipped frozen base models)")


def load_trainable_checkpoint(model, path):
    """
    Loads the lightweight checkpoint file containing only trainable parameters.
    
    Key Point: strict=False must be used because the checkpoint file does NOT 
    contain the frozen parameters (e.g., BERT/DINO backbones).
    """
    print(f"Loading trainable parameters from {path}...")
    
    # 1. Load weights
    checkpoint = torch.load(path, map_location='cpu')["model"]
    
    # 2. Load into the model (Strict=False is crucial here)
    # 'missing_keys' will contain all frozen BERT/DINO parameters (this is expected behavior).
    # 'unexpected_keys' must be empty (indicating the checkpoint matches the model structure).
    msg = model.load_state_dict(checkpoint, strict=False)
    
    # 3. Verify loading results
    if len(msg.unexpected_keys) > 0:
        print("WARNING: Found unexpected keys in checkpoint:", msg.unexpected_keys)
    
    # Briefly verify if the expected parameters (LoRA, heads, etc.) were loaded
    print(f"Success! Missing keys (expected frozen params): {len(msg.missing_keys)}")
    print("Trainable parameters loaded.")


def load_model(model, path, mask_visualization=False):
    
    print(f"Loading fusion model from {path}...")

    if "LLM" in path or os.path.exists(os.path.join(path, "projectors.pth")):
        projector_path = os.path.join(path, "projectors.pth")
        proj_weights = torch.load(projector_path, map_location="cpu") 
        model.img_projector.load_state_dict(proj_weights["img_projector"])
        model.text_projector.load_state_dict(proj_weights["text_projector"])
        print("Projectors loaded.")

        if isinstance(model.llm, PeftModel):
            print("Detected existing PeftModel, loading adapter weights...")
            model.llm.load_adapter(path, adapter_name="default") 
        else:
            print("Loading LoRA adapter into base model...")
            model.llm = PeftModel.from_pretrained(model.llm, path)

        device = model.llm.device
        dtype = model.llm.dtype
        model.img_projector.to(device=device, dtype=dtype)
        model.text_projector.to(device=device, dtype=dtype)
        return

    checkpoint = None
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except BaseException as e:
        print('Impossible to load the checkpoint: %s' % str(e))
    if checkpoint is not None:
        try:
            if hasattr(checkpoint, "state_dict"):
                unexpected = model.load_state_dict(checkpoint.state_dict(), strict=False)
                print('Model loading info: {}'.format(unexpected))
            elif isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    if "SwinT_V2_tiny/MultiModaCL_Pretrained_Best" in path or "SwinT_V2_small/MultiModaCL_Pretrained_Best" in path:
                        state_dict = checkpoint["model"]
                        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
                        unexpected = model.load_state_dict(filtered_dict, strict=False)
                    elif "MaskImage" in path and not mask_visualization:
                        state_dict = checkpoint["model"]
                        filtered_dict = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
                        unexpected = model.load_state_dict(filtered_dict, strict=False)
                    else:
                        state_dict = checkpoint["model"]
                        unexpected = model.load_state_dict(state_dict, strict=False)
                    print('Model loading info: {}'.format(unexpected))
            else:
                unexpected = model.load_state_dict(checkpoint)
                print('Model loading info: {}'.format(unexpected))
        except BaseException as e:
            raise ValueError('Error while loading the model\'s weights: %s' % str(e))


def metric(output, target, sens_attr=None, group_wise=False, equity_alpha=1.):
    
    return_dict = {}
    target = np.array(target)
    output = np.array(output, dtype=np.float32)
    output_pred = np.argmax(output, axis=1)
    if np.isnan(output_pred).any():
        print("Warning: NaN detected in predictions! Replacing with 0.5...")
        output_pred = np.nan_to_num(output_pred, nan=0.5)

    if np.isnan(output).any():
        print("Warning: NaN detected in predictions! Replacing with 0.5...")
        output = np.nan_to_num(output, nan=0.5)
    
    return_dict['overall_confusion_matrix'] = confusion_matrix(target, output_pred)
    return_dict['overall_classification_report'] = classification_report(target, output_pred, output_dict=True)
    return_dict['overall_bacc'] = balanced_accuracy_score(target, output_pred)
    return_dict['overall_auc'] = compute_auc(output, target, num_classes=len(np.unique(target)))
    
    if sens_attr is not None and len(sens_attr) > 0:
        sens_attr = np.array(sens_attr)
        return_dict['overall_ES_acc'] = equity_scaled_accuracy(output, target, sens_attr, alpha=equity_alpha)
        return_dict['overall_ES_AUC'] = equity_scaled_AUC(output, target, sens_attr, alpha=equity_alpha, num_classes=len(np.unique(target)))
        return_dict['overall_ES_Class_Attr_Scaled_F1'] = equity_class_attr_scaled_f1(output, target, sens_attr, alpha=equity_alpha)
        return_dict['overall_EO'] = multiclass_equalized_odds(output, target, sens_attr)
        return_dict['overall_DP'] = multiclass_demographic_parity(output, target, sens_attr)

        if group_wise:
            for one_attr in np.unique(sens_attr):
                idx = sens_attr == one_attr
                attr_output = output[idx]
                attr_target = target[idx]
                attr_output_pred = np.argmax(attr_output, axis=1)
                return_dict['attr_{}_confusion_matrix'.format(one_attr)] = confusion_matrix(attr_target, attr_output_pred)
                return_dict['attr_{}_classification_report'.format(one_attr)] = classification_report(attr_target, attr_output_pred, output_dict=True)
                return_dict['attr_{}_AUC'.format(one_attr)] = compute_auc(attr_output, attr_target, num_classes=len(np.unique(target)))

    return return_dict


def adjust_learning_rate(optimizer, epoch, base_lr, nb_epochs, warm_epochs, power_lr):
    if epoch < warm_epochs:
        lr = epoch / warm_epochs * base_lr
    else:
        lr = base_lr * ((1 - (epoch - warm_epochs) / nb_epochs) ** (power_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    y_onehot = num_to_onehot(y, num_classes)
    auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val


def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec


def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors


def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array


def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)
        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score


def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)
        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score


def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc


def equity_class_attr_scaled_f1(output, target, attrs, alpha=1.):
    es_f1 = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []

    output_pred = np.argmax(output, axis=1)
    overall_f1 = f1_score(target, output_pred, average='macro')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]
        pred_group_pred = np.argmax(pred_group, axis=1)

        f1_per_class = f1_score(gt_group, pred_group_pred, average=None, labels=[0, 1], zero_division="warn") # type: ignore
        for i in range(len(f1_per_class)):  # pyright: ignore[reportArgumentType]
            identity_wise_perf.append(f1_per_class[i])  # pyright: ignore[reportIndexIssue]
            identity_wise_num.append(gt_group.shape[0])


    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_f1)
    es_f1 = (overall_f1 / (alpha*tmp + 1))

    return es_f1


def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    

    y_onehot = num_to_onehot(target, num_classes)
    overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')


    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        y_onehot = num_to_onehot(gt_group, num_classes)
        group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc
