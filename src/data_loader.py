import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import (
    CATEGORIES,
    TASK_DEFINITIONS,
    TASK_CLASSES,
    TASK_CUMULATIVE_CLASSES,
    ID2CATEGORY,
    RESULTS_METRICS_DIR,
    FINAL_METRICS_PATH,
    MEMORY_STUDY_PATH,
    NOVELTY_METRICS_PATH,
    OFFICIAL_BENCHMARK_MANIFEST_PATH,
    PREDICTION_TRANSITION_MATRIX_PATH
)

logger = logging.getLogger(__name__)

TASK_HISTORY_PATH = RESULTS_METRICS_DIR / "task_history.json"
SCIENTIFIC_SUMMARY_PATH = RESULTS_METRICS_DIR / "scientific_summary.json"


def _safe_load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely loads a JSON file with validation."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def calculate_empirical_task_forgetting(task_history: List[Dict[str, Any]]) -> List[Optional[float]]:
    """
    Computes empirical catastrophic forgetting across task history.
    For Task 1: None (no prior tasks).
    For Task t > 1: Average accuracy drop on prior classes from their peak to current task t.
    """
    if not task_history:
        return []

    forgetting_by_task = [None] * len(task_history)

    for t_idx in range(1, len(task_history)):
        current_per_class = task_history[t_idx].get("per_class_accuracy", {})
        prior_classes = task_history[t_idx - 1].get("classes", [])
        drops = []

        for c_name in prior_classes:
            # Find peak historical accuracy for this class prior to task t_idx
            peak_acc = None
            for prev_idx in range(t_idx):
                prev_acc = task_history[prev_idx].get("per_class_accuracy", {}).get(c_name)
                if prev_acc is not None:
                    if peak_acc is None or prev_acc > peak_acc:
                        peak_acc = prev_acc

            curr_acc = current_per_class.get(c_name)
            if peak_acc is not None and curr_acc is not None:
                drop = max(0.0, peak_acc - curr_acc)
                drops.append(drop)

        if drops:
            forgetting_by_task[t_idx] = float(sum(drops) / len(drops))
        else:
            forgetting_by_task[t_idx] = None

    return forgetting_by_task


def normalize_task_history(raw_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes task history into a consistent, robust schema.
    Explicitly tracks introduced vs not-yet-introduced classes for each task.
    """
    all_categories = list(CATEGORIES)
    normalized = []
    forgetting_list = calculate_empirical_task_forgetting(raw_history)

    for idx, entry in enumerate(raw_history):
        task_id = entry.get("task_id", idx + 1)
        
        # Expected cumulative classes for this task ID
        expected_cumulative_ids = TASK_CUMULATIVE_CLASSES.get(task_id, [0, 1, 2, 3][:task_id + 1])
        expected_introduced = [ID2CATEGORY[c] for c in expected_cumulative_ids]
        
        # Actual classes reported in entry or fallback to expected
        entry_classes = entry.get("classes", expected_introduced)
        introduced = [c for c in all_categories if c in entry_classes]
        not_yet_introduced = [c for c in all_categories if c not in introduced]

        # Per-class accuracy
        raw_per_class = entry.get("per_class_accuracy", {})
        per_class_acc = {}
        for c in introduced:
            val = raw_per_class.get(c)
            per_class_acc[c] = float(val) if val is not None and isinstance(val, (int, float)) else None

        # Overall accuracy
        overall = entry.get("overall_accuracy")
        if overall is not None and isinstance(overall, (int, float)):
            overall_val = float(overall)
        else:
            overall_val = None

        f_val = forgetting_list[idx] if idx < len(forgetting_list) else None

        normalized.append({
            "task_id": task_id,
            "classes": introduced,
            "not_yet_introduced": not_yet_introduced,
            "overall_accuracy": overall_val,
            "balanced_accuracy": entry.get("balanced_accuracy"),
            "macro_f1": entry.get("macro_f1"),
            "per_class_accuracy": per_class_acc,
            "forgetting": f_val,
            "total_samples": entry.get("total_samples")
        })

    return normalized


def load_experiment_results() -> Dict[str, Any]:
    """
    Centralized, robust loader for all EvoRoute experimental data.
    
    1. Discovers and validates JSON metric files.
    2. Normalizes inconsistent keys.
    3. Guarantees task-wise metrics for Evolution Simulation and Plots.
    4. Never fabricates numbers: returns None when a metric is genuinely unavailable.
    5. Compiles debug information for non-intrusive inspection.
    """
    loaded_files = []
    
    # 1. Load primary metric files
    final_results = _safe_load_json(FINAL_METRICS_PATH)
    if final_results is not None:
        loaded_files.append({
            "name": FINAL_METRICS_PATH.name,
            "path": str(FINAL_METRICS_PATH),
            "status": "✓ loaded",
            "keys": list(final_results.keys())
        })
    else:
        loaded_files.append({
            "name": FINAL_METRICS_PATH.name,
            "path": str(FINAL_METRICS_PATH),
            "status": "✗ missing",
            "keys": []
        })

    task_history_raw = _safe_load_json(TASK_HISTORY_PATH)
    if task_history_raw is not None:
        loaded_files.append({
            "name": TASK_HISTORY_PATH.name,
            "path": str(TASK_HISTORY_PATH),
            "status": "✓ loaded",
            "keys": list(task_history_raw.keys())
        })
    else:
        loaded_files.append({
            "name": TASK_HISTORY_PATH.name,
            "path": str(TASK_HISTORY_PATH),
            "status": "✗ missing",
            "keys": []
        })

    official_manifest = _safe_load_json(OFFICIAL_BENCHMARK_MANIFEST_PATH)
    if official_manifest is not None:
        loaded_files.append({
            "name": OFFICIAL_BENCHMARK_MANIFEST_PATH.name,
            "path": str(OFFICIAL_BENCHMARK_MANIFEST_PATH),
            "status": "✓ loaded",
            "keys": list(official_manifest.keys())
        })

    memory_study = _safe_load_json(MEMORY_STUDY_PATH)
    if memory_study is not None:
        loaded_files.append({
            "name": MEMORY_STUDY_PATH.name,
            "path": str(MEMORY_STUDY_PATH),
            "status": "✓ loaded",
            "keys": list(memory_study.keys())
        })

    novelty_metrics = _safe_load_json(NOVELTY_METRICS_PATH)
    if novelty_metrics is not None:
        loaded_files.append({
            "name": NOVELTY_METRICS_PATH.name,
            "path": str(NOVELTY_METRICS_PATH),
            "status": "✓ loaded",
            "keys": list(novelty_metrics.keys())
        })

    transition_matrices = _safe_load_json(PREDICTION_TRANSITION_MATRIX_PATH)
    if transition_matrices is not None:
        loaded_files.append({
            "name": PREDICTION_TRANSITION_MATRIX_PATH.name,
            "path": str(PREDICTION_TRANSITION_MATRIX_PATH),
            "status": "✓ loaded",
            "keys": list(transition_matrices.keys())
        })

    # 2. Extract & normalize task history
    # If final_results has embedded task_history, merge with task_history_raw
    methods = [
        "naive", "ewc", "replay", "lwf", "replay_ewc", 
        "evoroute_br_candidate", "evoroute_br_calibrated", "joint"
    ]
    
    normalized_task_history = {}
    for m in methods:
        hist = None
        if task_history_raw and m in task_history_raw:
            hist = task_history_raw[m]
        elif final_results and m in final_results and "task_history" in final_results[m]:
            hist = final_results[m]["task_history"]

        if hist and isinstance(hist, list):
            normalized_task_history[m] = normalize_task_history(hist)

    # 3. Build normalized task view for the primary methods (e.g. naive, replay_ewc)
    tasks_view = []
    for t_id in [1, 2, 3]:
        cumulative_classes = [ID2CATEGORY[c] for c in TASK_CUMULATIVE_CLASSES[t_id]]
        not_yet = [c for c in CATEGORIES if c not in cumulative_classes]
        
        rep_task = None
        if "replay_ewc" in normalized_task_history and len(normalized_task_history["replay_ewc"]) >= t_id:
            rep_task = normalized_task_history["replay_ewc"][t_id - 1]

        naive_task = None
        if "naive" in normalized_task_history and len(normalized_task_history["naive"]) >= t_id:
            naive_task = normalized_task_history["naive"][t_id - 1]

        tasks_view.append({
            "task_id": t_id,
            "classes": cumulative_classes,
            "not_yet_introduced": not_yet,
            "replay_ewc": rep_task,
            "naive": naive_task
        })

    # 4. Prepare debug summaries
    debug_info = {
        "loaded_files": loaded_files,
        "available_keys": list(final_results.keys()) if final_results else [],
        "tasks_loaded": len(tasks_view),
        "task_1_metrics": tasks_view[0] if len(tasks_view) > 0 else None,
        "task_2_metrics": tasks_view[1] if len(tasks_view) > 1 else None,
        "task_3_metrics": tasks_view[2] if len(tasks_view) > 2 else None,
    }

    return {
        "final_results": final_results or {},
        "task_history": normalized_task_history,
        "tasks": tasks_view,
        "memory_study": memory_study or {},
        "novelty_metrics": novelty_metrics or {},
        "transition_matrices": transition_matrices or {},
        "official_manifest": official_manifest or {},
        "debug_info": debug_info
    }


def get_task_state(results: Dict[str, Any], method: str, task_id: int) -> Optional[Dict[str, Any]]:
    """
    Convenience helper to retrieve normalized task metrics for a specific method and task ID (1, 2, or 3).
    Returns None if data genuinely does not exist.
    """
    history = results.get("task_history", {}).get(method, [])
    for task in history:
        if task.get("task_id") == task_id:
            return task
    return None
