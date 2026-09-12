import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.config import (
    DEVICE,
    CALIBRATION_CONFIG,
    METRIC_TOLERANCE,
    MIN_NEWEST_CLASS_ACCURACY,
    ID2CATEGORY
)
from src.tasks import get_task_data
from src.recency_bias import compute_dataset_aware_recency_bias

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Brier Score.
    """
    n_samples, n_classes = probs.shape
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    # 1. Negative Log-Likelihood
    eps = 1e-12
    nll = -float(np.mean(np.log(probs[np.arange(n_samples), labels] + eps)))

    # 2. Brier Score (Multi-class Mean Squared Error of Probabilities)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # 3. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return {
        "nll": nll,
        "brier_score": brier,
        "ece": float(ece)
    }


def optimize_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_t: float = 1.0
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Optimizes Temperature Scaling parameter T > 0 by minimizing Negative Log-Likelihood strictly on validation logits.
    """
    temperature = nn.Parameter(torch.tensor([init_t], dtype=torch.float32, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    criterion = nn.CrossEntropyLoss()

    # Pre-calibration metrics
    with torch.no_grad():
        pre_probs = F.softmax(logits, dim=1).cpu().numpy()
        pre_metrics = compute_calibration_metrics(pre_probs, labels.cpu().numpy())

    def eval_loss():
        optimizer.zero_grad()
        # Enforce positive temperature
        t = torch.clamp(temperature, min=0.1, max=10.0)
        scaled_logits = logits / t
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    optimal_t = float(torch.clamp(temperature, min=0.1, max=10.0).item())

    with torch.no_grad():
        post_probs = F.softmax(logits / optimal_t, dim=1).cpu().numpy()
        post_metrics = compute_calibration_metrics(post_probs, labels.cpu().numpy())

    logger.info(f"Temperature optimization complete: T = {optimal_t:.4f} (NLL: {pre_metrics['nll']:.4f} -> {post_metrics['nll']:.4f}, ECE: {pre_metrics['ece']:.4f} -> {post_metrics['ece']:.4f})")

    return optimal_t, pre_metrics, post_metrics


def optimize_decision_rebalancing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    newest_class_id: int = 3,
    gamma_range: Tuple[float, float] = (0.0, 5.0),
    num_steps: int = 51,
    reg_weight: float = 0.01,
    baseline_newest_acc: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimizes relative logit shift parameter gamma >= 0 for the newest class on the validation split.
    z'_newest = z_newest - gamma
    
    Objective:
      J(gamma) = MacroF1_val(gamma) - reg_weight * gamma^2

    Safety Rejection Rule:
      Reject calibration (set gamma = 0) if:
      1. Newest-class validation accuracy falls below max(0.60, 0.70 * baseline_newest_acc)
      2. Validation lexicographic score (Macro F1, Balanced Acc, Overall Acc) does not improve over gamma = 0.
    """
    y_true = labels.cpu().numpy()
    raw_logits_np = logits.detach().cpu().numpy()

    # Pre-calibration evaluation (gamma = 0)
    raw_preds = np.argmax(raw_logits_np, axis=1)
    base_macro_f1 = float(f1_score(y_true, raw_preds, average="macro", zero_division=0))
    base_bal_acc = float(balanced_accuracy_score(y_true, raw_preds))
    base_acc = float(accuracy_score(y_true, raw_preds))
    
    newest_mask = (y_true == newest_class_id)
    base_newest_acc = float(np.mean(raw_preds[newest_mask] == newest_class_id)) if newest_mask.sum() > 0 else 0.0
    ref_newest_acc = baseline_newest_acc if baseline_newest_acc is not None else base_newest_acc
    min_allowable_newest_acc = max(MIN_NEWEST_CLASS_ACCURACY, 0.70 * ref_newest_acc)

    gammas = np.linspace(gamma_range[0], gamma_range[1], num_steps)
    best_gamma = 0.0
    best_score = base_macro_f1
    best_tuple = (base_macro_f1, base_bal_acc, base_acc)
    best_eval = {
        "macro_f1": base_macro_f1,
        "balanced_accuracy": base_bal_acc,
        "overall_accuracy": base_acc,
        "newest_accuracy": base_newest_acc
    }
    rejection_reason = None

    for gamma in gammas:
        if gamma == 0.0:
            continue

        shifted_logits = raw_logits_np.copy()
        shifted_logits[:, newest_class_id] -= gamma
        preds = np.argmax(shifted_logits, axis=1)

        newest_acc = float(np.mean(preds[newest_mask] == newest_class_id)) if newest_mask.sum() > 0 else 0.0
        # Invariant 1: Do not collapse newest class accuracy
        if newest_acc < min_allowable_newest_acc:
            continue

        m_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        b_acc = float(balanced_accuracy_score(y_true, preds))
        o_acc = float(accuracy_score(y_true, preds))
        current_tuple = (m_f1, b_acc, o_acc)

        # Objective with L2 regularization
        obj = m_f1 - reg_weight * (gamma ** 2)

        # Lexicographic check with tolerance
        is_better = False
        if m_f1 > best_tuple[0] + METRIC_TOLERANCE:
            is_better = True
        elif abs(m_f1 - best_tuple[0]) <= METRIC_TOLERANCE:
            if b_acc > best_tuple[1] + METRIC_TOLERANCE:
                is_better = True
            elif abs(b_acc - best_tuple[1]) <= METRIC_TOLERANCE and o_acc > best_tuple[2] + METRIC_TOLERANCE:
                is_better = True

        if is_better:
            best_gamma = float(gamma)
            best_tuple = current_tuple
            best_eval = {
                "macro_f1": m_f1,
                "balanced_accuracy": b_acc,
                "overall_accuracy": o_acc,
                "newest_accuracy": newest_acc
            }

    # Verify Safety Rejection Rule
    rejection_triggered = False
    if best_gamma == 0.0:
        rejection_triggered = True
        rejection_reason = "No positive gamma candidate improved validation lexicographic score while preserving newest-class accuracy."
        logger.info(f"Decision Rebalancing Safety Rejection triggered: {rejection_reason} Gamma set to 0.0 (no-op).")
    else:
        logger.info(f"Decision Rebalancing optimal gamma selected: {best_gamma:.2f} (Macro F1: {base_macro_f1:.4f} -> {best_eval['macro_f1']:.4f}, Newest Acc: {base_newest_acc:.4f} -> {best_eval['newest_accuracy']:.4f})")

    return {
        "optimal_gamma": best_gamma,
        "rejection_triggered": rejection_triggered,
        "rejection_reason": rejection_reason,
        "pre_rebalancing_metrics": {
            "macro_f1": base_macro_f1,
            "balanced_accuracy": base_bal_acc,
            "overall_accuracy": base_acc,
            "newest_accuracy": base_newest_acc
        },
        "post_rebalancing_metrics": best_eval,
        "min_allowable_newest_acc": min_allowable_newest_acc
    }


class CalibratedModelWrapper(nn.Module):
    """
    Inference-time model wrapper that encapsulates:
    1. Underlying neural classifier (EvoMLP)
    2. Temperature Scaling (T > 0)
    3. Decision Rebalancing (gamma >= 0 for newest class)
    
    Guarantees:
    - Never modifies the underlying network weights.
    - If rejection was triggered during validation calibration, gamma = 0.0.
    """
    def __init__(
        self,
        base_model: nn.Module,
        temperature: float = 1.0,
        gamma: float = 0.0,
        newest_class_id: int = 3,
        num_classes: int = 4
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.newest_class_id = newest_class_id
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        # 1. Temperature Scaling
        scaled_logits = logits / self.temperature
        # 2. Decision Rebalancing
        if self.gamma != 0.0 and self.newest_class_id < scaled_logits.size(1):
            mask = torch.zeros_like(scaled_logits)
            mask[:, self.newest_class_id] = self.gamma
            scaled_logits = scaled_logits - mask
        return scaled_logits

    def get_calibration_state(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "gamma": self.gamma,
            "newest_class_id": self.newest_class_id,
            "num_classes": self.num_classes
        }


def run_post_hoc_calibration(
    model: nn.Module,
    device: str = DEVICE,
    baseline_newest_acc: Optional[float] = None
) -> Tuple[CalibratedModelWrapper, Dict[str, Any]]:
    """
    Full calibration pipeline executed STRICTLY on the held-out validation split.
    Stage 1: Confidence Calibration via Temperature Scaling.
    Stage 2: Decision Rebalancing via Logit Shift with Safety Rejection Rule.
    """
    model.eval()
    val_embs, val_lbls, _ = get_task_data(task_id=3, split="val", cumulative=True)

    with torch.no_grad():
        logits = model(val_embs.to(device))
        labels = val_lbls.to(device)

    # Stage 1: Temperature Scaling
    optimal_t, pre_cal, post_cal = optimize_temperature(logits, labels)

    # Stage 2: Decision Rebalancing on Temperature-Scaled Logits
    scaled_logits = logits / optimal_t
    rebalance_res = optimize_decision_rebalancing(
        scaled_logits,
        labels,
        newest_class_id=3,
        baseline_newest_acc=baseline_newest_acc
    )

    wrapper = CalibratedModelWrapper(
        base_model=model,
        temperature=optimal_t,
        gamma=rebalance_res["optimal_gamma"],
        newest_class_id=3,
        num_classes=4
    )

    calibration_report = {
        "temperature_scaling": {
            "optimal_temperature": optimal_t,
            "pre_calibration": pre_cal,
            "post_calibration": post_cal
        },
        "decision_rebalancing": rebalance_res,
        "calibration_state": wrapper.get_calibration_state()
    }

    return wrapper, calibration_report
