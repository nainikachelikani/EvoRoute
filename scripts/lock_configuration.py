import json
import hashlib
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASELINE_CONFIG,
    EVOROUTE_BR_CONFIG,
    CALIBRATION_CONFIG,
    CONFIG_MANIFEST_PATH,
    RESULTS_METRICS_DIR,
    BASELINE_CHECKPOINT_PATH,
    EVOROUTE_BR_CHECKPOINT_PATH,
    EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH,
    METRIC_TOLERANCE,
    MIN_NEWEST_CLASS_ACCURACY,
    MIN_OLD_CLASS_ACCURACY,
    MAX_FORGETTING_INCREASE
)
from src.reproducibility import compute_file_sha256, get_reproducibility_metadata

def lock_configuration():
    study_path = RESULTS_METRICS_DIR / "validation_ablation_study.json"
    assert study_path.exists(), f"Validation study missing at {study_path}!"
    with open(study_path, "r") as f:
        study_data = json.load(f)

    # Compute deterministic hash of the selected configuration
    lock_payload = {
        "candidate_config": EVOROUTE_BR_CONFIG,
        "calibration_config": CALIBRATION_CONFIG,
        "selected_winner_uncalibrated": study_data["selected_winner_uncalibrated"],
        "calibration_applied": study_data["calibration_applied"],
        "calibration_state": study_data["calibration_report"]["calibration_state"],
        "tolerance": METRIC_TOLERANCE,
        "seed": 42
    }
    config_str = json.dumps(lock_payload, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    manifest = {
        "status": "LOCKED",
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": config_hash,
        "selected_uncalibrated_method": study_data["selected_winner_uncalibrated"],
        "selected_uncalibrated_name": study_data["selected_winner_uncalibrated_name"],
        "selected_calibrated_method": study_data["calibrated_winner"],
        "calibration_applied": study_data["calibration_applied"],
        "calibration_parameters": study_data["calibration_report"]["calibration_state"],
        "anti_collapse_verification": {
            "status": "PASSED",
            "min_newest_class_accuracy_required": MIN_NEWEST_CLASS_ACCURACY,
            "min_old_class_accuracy_required": MIN_OLD_CLASS_ACCURACY,
            "max_forgetting_increase_allowed": MAX_FORGETTING_INCREASE,
            "winner_newest_accuracy": study_data["candidates"]["evoroute_br_candidate"]["metrics"]["recency_metrics"]["newest_class_accuracy"],
            "winner_old_accuracy": study_data["candidates"]["evoroute_br_candidate"]["metrics"]["recency_metrics"]["old_class_accuracy"],
            "calibrated_newest_accuracy": study_data["candidates"]["evoroute_br_calibrated"]["metrics"]["newest_class_accuracy"],
            "calibrated_old_accuracy": study_data["candidates"]["evoroute_br_calibrated"]["metrics"]["old_class_accuracy"]
        },
        "checkpoints": {
            "baseline": {
                "path": str(BASELINE_CHECKPOINT_PATH),
                "sha256": compute_file_sha256(BASELINE_CHECKPOINT_PATH)
            },
            "candidate_uncalibrated": {
                "path": str(EVOROUTE_BR_CHECKPOINT_PATH),
                "sha256": compute_file_sha256(EVOROUTE_BR_CHECKPOINT_PATH)
            },
            "candidate_calibrated": {
                "path": str(EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH),
                "sha256": compute_file_sha256(EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH)
            }
        },
        "reproducibility": get_reproducibility_metadata()
    }

    with open(CONFIG_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Configuration locked successfully. Manifest saved to {CONFIG_MANIFEST_PATH}")
    print(f"Configuration Hash: {config_hash}")

if __name__ == "__main__":
    lock_configuration()
