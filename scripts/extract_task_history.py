import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.config import (
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    FINAL_METRICS_PATH,
    CATEGORIES,
    TASK_CUMULATIVE_CLASSES,
    TASK_CLASSES,
    ID2CATEGORY,
    DEVICE
)
from src.model import EvoMLP
from src.evaluate import evaluate_model_on_classes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_HISTORY_PATH = RESULTS_METRICS_DIR / "task_history.json"


def extract_empirical_task_history() -> Dict[str, Any]:
    """
    Evaluates real saved task checkpoints across tasks 1, 2, 3 on the test set
    to build empirical task-wise continual learning trajectories.
    Zero fabricated numbers: every value is measured directly from the model weights.
    """
    methods = ["naive", "ewc", "replay", "lwf", "replay_ewc", "evoroute_br_candidate"]
    task_history_by_method = {}

    for m in methods:
        history = []
        for t in [1, 2, 3]:
            ckpt_path = MODELS_DIR / f"{m}_task{t}.pt"
            if not ckpt_path.exists():
                # Fallback to final checkpoint if task3 checkpoint path differs
                if t == 3 and (MODELS_DIR / f"{m}_final.pt").exists():
                    ckpt_path = MODELS_DIR / f"{m}_final.pt"
                else:
                    logger.warning(f"Checkpoint {ckpt_path} not found for {m} task {t}")
                    continue

            allowed_classes = TASK_CUMULATIVE_CLASSES[t]
            num_classes = len(allowed_classes)
            model = EvoMLP(num_classes=num_classes).to(DEVICE)
            state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            eval_res = evaluate_model_on_classes(
                model=model,
                split="test",
                allowed_classes=allowed_classes,
                device=DEVICE
            )

            history.append({
                "task_id": t,
                "classes": [ID2CATEGORY[c] for c in allowed_classes],
                "new_classes": [ID2CATEGORY[c] for c in TASK_CLASSES[t]],
                "overall_accuracy": eval_res["accuracy"],
                "balanced_accuracy": eval_res["balanced_accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "per_class_accuracy": eval_res["per_class_accuracy"],
                "total_samples": eval_res["total_samples"]
            })

        task_history_by_method[m] = history
        logger.info(f"Extracted verified task history for {m}: {len(history)} tasks.")

    # Joint Training upper bound task trajectories (for reference)
    joint_ckpt = MODELS_DIR / "joint_final.pt"
    if joint_ckpt.exists():
        joint_model = EvoMLP(num_classes=4).to(DEVICE)
        joint_model.load_state_dict(torch.load(joint_ckpt, map_location=DEVICE, weights_only=True))
        joint_model.eval()
        joint_history = []
        for t in [1, 2, 3]:
            allowed = TASK_CUMULATIVE_CLASSES[t]
            j_eval = evaluate_model_on_classes(joint_model, split="test", allowed_classes=allowed, device=DEVICE)
            joint_history.append({
                "task_id": t,
                "classes": [ID2CATEGORY[c] for c in allowed],
                "new_classes": [ID2CATEGORY[c] for c in TASK_CLASSES[t]],
                "overall_accuracy": j_eval["accuracy"],
                "balanced_accuracy": j_eval["balanced_accuracy"],
                "macro_f1": j_eval["macro_f1"],
                "per_class_accuracy": j_eval["per_class_accuracy"],
                "total_samples": j_eval["total_samples"]
            })
        task_history_by_method["joint"] = joint_history

    # EvoRoute-BR Calibrated task trajectory
    cal_ckpt = MODELS_DIR / "evoroute_br_calibrated_final.pt"
    if cal_ckpt.exists() and "evoroute_br_candidate" in task_history_by_method:
        cand_hist = task_history_by_method["evoroute_br_candidate"]
        cal_history = []
        # Task 1 & Task 2 match candidate (since calibration is post-hoc after Task 3 expansion)
        if len(cand_hist) >= 2:
            cal_history.append(cand_hist[0])
            cal_history.append(cand_hist[1])
        # Task 3 evaluates calibrated model on 4 classes
        from src.calibration import CalibratedModelWrapper
        ckpt_dict = torch.load(cal_ckpt, map_location=DEVICE, weights_only=False)
        base_model = EvoMLP(num_classes=4).to(DEVICE)
        base_model.load_state_dict(ckpt_dict["base_model_state"])
        base_model.eval()
        cal_state = ckpt_dict["calibration_state"]
        wrapper = CalibratedModelWrapper(
            base_model=base_model,
            temperature=cal_state["temperature"],
            gamma=cal_state["gamma"],
            newest_class_id=cal_state["newest_class_id"],
            num_classes=cal_state["num_classes"]
        )
        wrapper.eval()
        cal_eval = evaluate_model_on_classes(wrapper, split="test", allowed_classes=[0, 1, 2, 3], device=DEVICE)
        cal_history.append({
            "task_id": 3,
            "classes": CATEGORIES,
            "new_classes": ["Household"],
            "overall_accuracy": cal_eval["accuracy"],
            "balanced_accuracy": cal_eval["balanced_accuracy"],
            "macro_f1": cal_eval["macro_f1"],
            "per_class_accuracy": cal_eval["per_class_accuracy"],
            "total_samples": cal_eval["total_samples"]
        })
        task_history_by_method["evoroute_br_calibrated"] = cal_history

    with open(TASK_HISTORY_PATH, "w") as f:
        json.dump(task_history_by_method, f, indent=2)
    logger.info(f"Saved verified empirical task history to {TASK_HISTORY_PATH}")

    # Also augment final_results.json so task_history is cleanly embedded
    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, "r") as f:
            final_metrics = json.load(f)
        for m_key, hist in task_history_by_method.items():
            if m_key in final_metrics:
                final_metrics[m_key]["task_history"] = hist
        with open(FINAL_METRICS_PATH, "w") as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Embedded task_history into {FINAL_METRICS_PATH}")

    return task_history_by_method


if __name__ == "__main__":
    extract_empirical_task_history()
