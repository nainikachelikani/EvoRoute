import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from src.config import (
    DEVICE,
    CATEGORIES,
    CATEGORY2ID,
    ID2CATEGORY,
    TASK_CUMULATIVE_CLASSES,
    TASK_CLASSES
)
from src.tasks import get_task_data

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
    Computes overall accuracy and per-class accuracies.
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
        return {"accuracy": 0.0, "per_class_accuracy": {}}

    with torch.no_grad():
        inputs = embs.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu()

    overall_acc = float(accuracy_score(lbls.numpy(), preds.numpy()))

    per_class_acc = {}
    for c in (allowed_classes if allowed_classes is not None else sorted(list(torch.unique(lbls).numpy()))):
        c_mask = (lbls == c).numpy()
        if c_mask.sum() > 0:
            c_acc = float(accuracy_score(lbls[c_mask].numpy(), preds[c_mask].numpy()))
            per_class_acc[ID2CATEGORY.get(c, f"Class_{c}")] = c_acc

    return {
        "accuracy": overall_acc,
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
