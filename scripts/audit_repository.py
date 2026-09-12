import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    FINAL_METRICS_PATH,
    CATEGORIES,
    TASK_DEFINITIONS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS_PER_TASK,
    REPLAY_MEMORY_BUDGET,
    REPLAY_SAMPLE_RATIO,
    EWC_LAMBDA,
    LWF_TEMPERATURE,
    LWF_LAMBDA
)
from src.reproducibility import (
    compute_file_sha256,
    generate_baseline_manifest,
    get_reproducibility_metadata
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_repository():
    """Audits existing checkpoints, architectures, configurations, and records baseline manifest."""
    logger.info("Starting Repository Audit...")
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = MODELS_DIR / "replay_ewc_final.pt"
    baseline_manifest_path = RESULTS_METRICS_DIR / "baseline_manifest.json"

    # 1. Generate programmatic baseline manifest
    if baseline_ckpt.exists():
        baseline_manifest = generate_baseline_manifest(
            baseline_checkpoint_path=baseline_ckpt,
            final_results_path=FINAL_METRICS_PATH,
            output_manifest_path=baseline_manifest_path
        )
    else:
        logger.warning(f"Baseline checkpoint {baseline_ckpt} not found! Checkpoint audit skipped.")
        baseline_manifest = {}

    # 2. Audit all existing checkpoints in models/
    checkpoint_inventory = {}
    for pt_file in sorted(MODELS_DIR.glob("*.pt")):
        checkpoint_inventory[pt_file.name] = {
            "size_bytes": pt_file.stat().st_size,
            "sha256": compute_file_sha256(pt_file)
        }

    audit_summary = {
        "project_name": "EvoRoute — Adaptive E-Commerce Intelligence",
        "continual_learning_track": "Strict Class-Incremental Learning (Zero task oracle)",
        "categories": CATEGORIES,
        "tasks": TASK_DEFINITIONS,
        "baseline_training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs_per_task": EPOCHS_PER_TASK,
            "replay_memory_budget": REPLAY_MEMORY_BUDGET,
            "replay_sample_ratio": REPLAY_SAMPLE_RATIO,
            "effective_batch_size": BATCH_SIZE + int(BATCH_SIZE * REPLAY_SAMPLE_RATIO),  # 64 + 12 = 76
            "ewc_lambda": EWC_LAMBDA,
            "lwf_temperature": LWF_TEMPERATURE,
            "lwf_lambda": LWF_LAMBDA
        },
        "total_checkpoints_found": len(checkpoint_inventory),
        "checkpoints": checkpoint_inventory,
        "baseline_sha256": baseline_manifest.get("checkpoint_sha256"),
        "environment": get_reproducibility_metadata()
    }

    audit_out_path = RESULTS_METRICS_DIR / "repository_audit.json"
    with open(audit_out_path, "w") as f:
        json.dump(audit_summary, f, indent=4)

    logger.info(f"Repository audit successfully saved to: {audit_out_path}")
    logger.info(f"Audited {len(checkpoint_inventory)} checkpoints.")
    return audit_summary


if __name__ == "__main__":
    audit_repository()
