import logging
from typing import Dict, List, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    CATEGORIES,
    ID2CATEGORY,
    DEVICE
)
from src.tasks import get_task_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_recency_bias_metrics(
    model: torch.nn.Module,
    split: str = "test",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Computes dataset-aware dynamic recency metrics, balanced accuracy, and Macro F1.
    Never hardcodes 0.25 or 'Household'; dynamically discovers seen classes from the model
    and derives ground-truth class distributions directly from the evaluation split.
    """
    model.eval()
    all_embs, all_lbls, _ = get_task_data(task_id=3, split=split, cumulative=True)
    total_samples = len(all_lbls)
    if total_samples == 0:
        return {}

    num_seen_classes = model.num_classes
    newest_class_id = num_seen_classes - 1
    old_class_ids = list(range(newest_class_id))

    # Mask samples to currently seen classes
    mask = torch.isin(all_lbls, torch.tensor(list(range(num_seen_classes))))
    embs = all_embs[mask]
    lbls = all_lbls[mask].numpy()
    n_eval = len(lbls)

    with torch.no_grad():
        inputs = embs.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # 1. Overall & Balanced Accuracy
    overall_acc = float(accuracy_score(lbls, preds))
    
    # 2. Macro F1, Precision, Recall
    prec, rec, f1, _ = precision_recall_fscore_support(lbls, preds, average="macro", zero_division=0)
    macro_f1 = float(f1)
    macro_prec = float(prec)
    macro_rec = float(rec)

    # 3. Dynamic Predicted vs True Distributions
    pred_distribution = {}
    true_distribution = {}
    per_class_bias = {}
    per_class_accuracy = {}
    class_acc_list = []

    for c_id in range(num_seen_classes):
        c_name = ID2CATEGORY.get(c_id, f"Class_{c_id}")
        c_pred_count = int(np.sum(preds == c_id))
        c_true_count = int(np.sum(lbls == c_id))

        p_pred = float(c_pred_count / n_eval)
        p_true = float(c_true_count / n_eval)
        pred_distribution[c_name] = p_pred
        true_distribution[c_name] = p_true
        per_class_bias[c_name] = float(p_pred - p_true)

        c_mask = (lbls == c_id)
        if c_mask.sum() > 0:
            c_acc = float(np.mean(preds[c_mask] == c_id))
        else:
            c_acc = 0.0
        per_class_accuracy[c_name] = c_acc
        class_acc_list.append(c_acc)

    balanced_acc = float(np.mean(class_acc_list)) if class_acc_list else 0.0

    # 4. Old vs Newest Class Metrics
    newest_name = ID2CATEGORY.get(newest_class_id, f"Class_{newest_class_id}")
    newest_acc = per_class_accuracy[newest_name]

    old_accs = [per_class_accuracy[ID2CATEGORY.get(c, f"Class_{c}")] for c in old_class_ids]
    old_class_acc = float(np.mean(old_accs)) if old_accs else 0.0
    acc_gap = float(newest_acc - old_class_acc)

    # 5. Dataset-Aware Recency Bias
    p_pred_newest = pred_distribution[newest_name]
    p_true_newest = true_distribution[newest_name]
    recency_bias = float(p_pred_newest - p_true_newest)

    # 6. Normalized Prediction Entropy
    eps = 1e-12
    probs_array = np.array(list(pred_distribution.values()))
    raw_entropy = float(-np.sum(probs_array * np.log2(probs_array + eps)))
    max_entropy = float(np.log2(num_seen_classes)) if num_seen_classes > 1 else 1.0
    normalized_entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "recency_bias": recency_bias,
        "newest_prediction_rate": p_pred_newest,
        "newest_true_rate": p_true_newest,
        "newest_class_accuracy": newest_acc,
        "old_class_accuracy": old_class_acc,
        "accuracy_gap": acc_gap,
        "absolute_imbalance": abs(acc_gap),
        "normalized_entropy": normalized_entropy,
        "raw_entropy": raw_entropy,
        "predicted_distribution": pred_distribution,
        "true_distribution": true_distribution,
        "per_class_bias": per_class_bias,
        "per_class_accuracy": per_class_accuracy,
        "newest_class_name": newest_name,
        "num_seen_classes": num_seen_classes,
        "total_samples": n_eval
    }


def compute_dataset_aware_recency_bias(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_seen_classes: int = None
) -> Dict[str, Any]:
    """
    Computes dataset-aware dynamic recency metrics directly from ground-truth and prediction arrays.
    RecencyBias(c) = P(predicted = c) - P(true = c)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_eval = len(y_true)
    if n_eval == 0:
        return {}

    if num_seen_classes is None:
        num_seen_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1

    newest_class_id = num_seen_classes - 1
    old_class_ids = list(range(newest_class_id))

    overall_acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    pred_distribution = {}
    true_distribution = {}
    per_class_bias = {}
    per_class_accuracy = {}
    class_acc_list = []

    for c_id in range(num_seen_classes):
        c_name = ID2CATEGORY.get(c_id, f"Class_{c_id}")
        c_pred_count = int(np.sum(y_pred == c_id))
        c_true_count = int(np.sum(y_true == c_id))

        p_pred = float(c_pred_count / n_eval)
        p_true = float(c_true_count / n_eval)
        pred_distribution[c_name] = p_pred
        true_distribution[c_name] = p_true
        per_class_bias[c_name] = float(p_pred - p_true)

        c_mask = (y_true == c_id)
        c_acc = float(np.mean(y_pred[c_mask] == c_id)) if c_mask.sum() > 0 else 0.0
        per_class_accuracy[c_name] = c_acc
        class_acc_list.append(c_acc)

    balanced_acc = float(np.mean(class_acc_list)) if class_acc_list else 0.0

    newest_name = ID2CATEGORY.get(newest_class_id, f"Class_{newest_class_id}")
    newest_acc = per_class_accuracy[newest_name]
    old_accs = [per_class_accuracy[ID2CATEGORY.get(c, f"Class_{c}")] for c in old_class_ids]
    old_class_acc = float(np.mean(old_accs)) if old_accs else 0.0
    acc_gap = float(newest_acc - old_class_acc)

    p_pred_newest = pred_distribution[newest_name]
    p_true_newest = true_distribution[newest_name]
    recency_bias = float(p_pred_newest - p_true_newest)

    eps = 1e-12
    probs_array = np.array(list(pred_distribution.values()))
    raw_entropy = float(-np.sum(probs_array * np.log2(probs_array + eps)))
    max_entropy = float(np.log2(num_seen_classes)) if num_seen_classes > 1 else 1.0
    normalized_entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "recency_bias": recency_bias,
        "newest_prediction_rate": p_pred_newest,
        "newest_true_rate": p_true_newest,
        "newest_class_accuracy": newest_acc,
        "old_class_accuracy": old_class_acc,
        "accuracy_gap": acc_gap,
        "absolute_imbalance": abs(acc_gap),
        "normalized_entropy": normalized_entropy,
        "raw_entropy": raw_entropy,
        "predicted_distribution": pred_distribution,
        "true_distribution": true_distribution,
        "per_class_bias": per_class_bias,
        "per_class_accuracy": per_class_accuracy,
        "newest_class_name": newest_name,
        "num_seen_classes": num_seen_classes,
        "total_samples": n_eval
    }
