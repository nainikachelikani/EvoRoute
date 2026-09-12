import os
import sys
import json
import random
import hashlib
import platform
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_SEED = 42
ROBUSTNESS_SEEDS = [42, 123, 456]


def set_global_seed(seed: int = PRIMARY_SEED) -> None:
    """
    Sets global seeds across Python random, NumPy, PyTorch CPU & GPU,
    and sets deterministic flags for PyTorch CuDNN.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed} (CuDNN deterministic: True)")


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a binary or text file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_string_hash(text: str) -> str:
    """Computes SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_reproducibility_metadata() -> Dict[str, Any]:
    """Captures runtime environment metadata for auditability."""
    metadata = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "numpy_version": np.__version__,
        "primary_seed": PRIMARY_SEED,
        "robustness_seeds": ROBUSTNESS_SEEDS
    }
    try:
        import sklearn
        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import sentence_transformers
        metadata["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass
    return metadata


def generate_baseline_manifest(
    baseline_checkpoint_path: Path,
    final_results_path: Path,
    output_manifest_path: Path
) -> Dict[str, Any]:
    """
    Programmatically generates baseline_manifest.json from existing files.
    Extracts SHA-256 hash, file size, timestamps, and official metrics from final_results.json.
    """
    if not baseline_checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {baseline_checkpoint_path}")

    sha256_hash = compute_file_sha256(baseline_checkpoint_path)
    file_stat = baseline_checkpoint_path.stat()
    file_size_bytes = file_stat.st_size
    modified_timestamp = file_stat.st_mtime

    baseline_metrics = {}
    if final_results_path.exists():
        with open(final_results_path, "r") as f:
            all_res = json.load(f)
            baseline_metrics = all_res.get("replay_ewc", {})

    manifest = {
        "baseline_method": "Replay + EWC",
        "checkpoint_filename": baseline_checkpoint_path.name,
        "checkpoint_path": str(baseline_checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_hash,
        "file_size_bytes": file_size_bytes,
        "file_modified_timestamp": modified_timestamp,
        "random_seed": PRIMARY_SEED,
        "memory_budget": 200,
        "effective_batch_size": 76,  # 64 new + 12 replay (sample_ratio=0.2)
        "official_metrics": {
            "overall_accuracy": baseline_metrics.get("overall_accuracy"),
            "final_avg_task_accuracy": baseline_metrics.get("final_avg_task_accuracy"),
            "average_forgetting": baseline_metrics.get("average_forgetting"),
            "recency_bias": baseline_metrics.get("recency_bias"),
            "per_class_accuracy": baseline_metrics.get("final_per_class", {}),
            "prediction_distribution": baseline_metrics.get("prediction_distribution", {})
        },
        "environment_metadata": get_reproducibility_metadata()
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info(f"Generated baseline manifest at: {output_manifest_path}")
    logger.info(f"Baseline Checkpoint SHA-256: {sha256_hash}")
    return manifest
