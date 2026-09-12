import logging
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

from src.config import (
    DEVICE,
    CATEGORIES,
    CATEGORY2ID,
    ID2CATEGORY,
    TASK_CUMULATIVE_CLASSES,
    TASK_CLASSES
)
from src.tasks import get_task_data
from src.recency_bias import compute_dataset_aware_recency_bias

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_on_classes(
    model: nn.Module,
    split: str = "test",
    allowed_classes: List[int] = None,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluates model on specified classes using embeddings from the given split ('test' or 'val').
    Computes overall accuracy, balanced accuracy, macro F1, and per-class accuracies.
    """
    model.eval()
    all_embs, all_lbls, _ = get_task_data(task_id=3, split=split, cumulative=True)

    if allowed_classes is not None:
        mask = torch.isin(all_lbls, torch.tensor(allowed_classes))
        embs = all_embs[mask]
        lbls = all_lbls[mask]
    else:
        embs = all_embs
        lbls = all_lbls

    if len(lbls) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_accuracy": {},
            "total_samples": 0
        }

    with torch.no_grad():
        inputs = embs.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu()

    y_true = lbls.numpy()
    y_pred = preds.numpy()

    overall_acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    m_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_acc = {}
    target_cats = allowed_classes if allowed_classes is not None else sorted(list(torch.unique(lbls).numpy()))
    for c in target_cats:
        c_mask = (y_true == c)
        if c_mask.sum() > 0:
            c_acc = float(accuracy_score(y_true[c_mask], y_pred[c_mask]))
            per_class_acc[ID2CATEGORY.get(c, f"Class_{c}")] = c_acc

    return {
        "accuracy": overall_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": m_f1,
        "per_class_accuracy": per_class_acc,
        "total_samples": len(lbls)
    }


def compute_forgetting(matrix: np.ndarray) -> Dict[str, float]:
    """
    Computes Catastrophic Forgetting from accuracy matrix R where R[i, j] is accuracy on task j after training task i.
    Forgetting for task j after final task T:
    f_j = max_{l < T} R[l, j] - R[T, j]
    """
    num_tasks = matrix.shape[0]
    forgetting_per_task = {}
    total_forgetting = 0.0

    # Only evaluate tasks before the final task (0 to num_tasks - 2)
    for j in range(num_tasks - 1):
        # Best performance on task j at or before task num_tasks - 2
        best_acc = np.max([matrix[l, j] for l in range(j, num_tasks - 1)])
        final_acc = matrix[num_tasks - 1, j]
        f_j = max(0.0, float(best_acc - final_acc))
        forgetting_per_task[f"Task_{j+1}"] = f_j
        total_forgetting += f_j

    avg_forgetting = total_forgetting / max(1, num_tasks - 1)
    return {
        "per_task_forgetting": forgetting_per_task,
        "average_forgetting": float(avg_forgetting)
    }


def compute_average_forgetting(matrix: np.ndarray) -> float:
    """
    Direct accessor for Average Catastrophic Forgetting across all tasks preceding final task T:
    f = (1 / (T-1)) * sum_{j=1}^{T-1} (max_{l < T} R[l, j] - R[T, j])
    """
    return float(compute_forgetting(matrix)["average_forgetting"])


def compute_final_average_task_accuracy(matrix: np.ndarray) -> float:
    """
    Computes Final Average Task Accuracy:
    Average of the individual task accuracies after the final task T:
    (R(T, 1) + R(T, 2) + ... + R(T, T)) / T
    where R(T, k) is accuracy on Task k after completing final Task T.
    """
    final_row = matrix[-1]
    return float(np.mean(final_row))


def evaluate_task_accuracies(
    model: nn.Module,
    current_task: int,
    split: str = "test",
    device: str = DEVICE
) -> Dict[int, float]:
    """
    Evaluates accuracy on each individual task seen so far (1 .. current_task).
    """
    accuracies = {}
    for task_id in range(1, current_task + 1):
        task_classes = TASK_CLASSES[task_id]
        res = evaluate_model_on_classes(model, split=split, allowed_classes=task_classes, device=device)
        accuracies[task_id] = res["accuracy"]
    return accuracies


def compute_prediction_distribution(
    model: nn.Module,
    split: str = "test",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Computes empirical prediction distribution across all classes on the evaluation split.
    Calculates the Recency Bias metric:
        RecencyBias = P(Newest Class) - P_expected
    where P_expected = 1 / num_classes (0.25 for 4 balanced test classes).
    """
    model.eval()
    all_embs, all_lbls, _ = get_task_data(task_id=3, split=split, cumulative=True)
    total_samples = len(all_lbls)
    if total_samples == 0:
        return {}

    with torch.no_grad():
        inputs = all_embs.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    num_classes = len(CATEGORIES)
    distribution = {}
    counts = {}
    percentages = {}

    for c_id, c_name in enumerate(CATEGORIES):
        c_count = int(np.sum(preds == c_id))
        c_frac = float(c_count / total_samples)
        counts[c_name] = c_count
        distribution[c_name] = c_frac
        percentages[c_name] = f"{c_frac * 100:.2f}%"

    newest_class = CATEGORIES[-1]
    p_newest = distribution[newest_class]
    p_expected = 1.0 / num_classes
    recency_bias = float(p_newest - p_expected)

    return {
        "distribution": distribution,
        "percentages": percentages,
        "counts": counts,
        "total_samples": total_samples,
        "newest_class": newest_class,
        "p_newest": p_newest,
        "p_expected": p_expected,
        "recency_bias": recency_bias
    }


def validate_checkpoint(
    model: nn.Module,
    split: str = "val",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Comprehensive validation diagnostic utility.
    Computes classification performance, dataset-aware recency bias metrics,
    and returns a standardized lexicographic comparison tuple (Macro F1, Balanced Acc, Overall Acc).
    """
    model.eval()
    all_embs, all_lbls, _ = get_task_data(task_id=3, split=split, cumulative=True)
    total_samples = len(all_lbls)

    with torch.no_grad():
        inputs = all_embs.to(device)
        targets = all_lbls.to(device)
        logits = model(inputs)
        loss = float(F.cross_entropy(logits, targets).item())
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    y_true = all_lbls.numpy()
    y_pred = preds.cpu().numpy()

    eval_res = evaluate_model_on_classes(model, split=split, allowed_classes=[0, 1, 2, 3], device=device)
    recency_metrics = compute_dataset_aware_recency_bias(y_true, y_pred)

    macro_f1 = eval_res["macro_f1"]
    balanced_acc = eval_res["balanced_accuracy"]
    overall_acc = eval_res["accuracy"]

    return {
        "split": split,
        "loss": loss,
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "per_class_accuracy": eval_res["per_class_accuracy"],
        "recency_metrics": recency_metrics,
        "lexicographic_tuple": (macro_f1, balanced_acc, overall_acc),
        "total_samples": total_samples
    }

