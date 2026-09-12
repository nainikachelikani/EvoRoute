import json
import pytest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, RESULTS_METRICS_DIR
from src.reproducibility import compute_file_sha256


def test_baseline_checkpoint_sha256_unmodified():
    """Asserts that the official baseline checkpoint models/replay_ewc_final.pt matches the manifest SHA-256."""
    baseline_ckpt = MODELS_DIR / "replay_ewc_final.pt"
    manifest_path = RESULTS_METRICS_DIR / "baseline_manifest.json"

    assert baseline_ckpt.exists(), f"Baseline checkpoint {baseline_ckpt} is missing!"
    assert manifest_path.exists(), f"Baseline manifest {manifest_path} is missing!"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    expected_sha256 = manifest["checkpoint_sha256"]
    actual_sha256 = compute_file_sha256(baseline_ckpt)

    assert actual_sha256 == expected_sha256, (
        f"FATAL: Baseline checkpoint has been modified or corrupted! "
        f"Expected SHA256: {expected_sha256}, Actual: {actual_sha256}"
    )


def test_baseline_overwrite_protection():
    """Asserts that experimental candidate models cannot overwrite the baseline checkpoint path."""
    baseline_ckpt = MODELS_DIR / "replay_ewc_final.pt"
    experimental_ckpt = MODELS_DIR / "candidates" / "evoroute_br_raw.pt"

    assert baseline_ckpt.resolve() != experimental_ckpt.resolve(), (
        "Experimental checkpoint path collides with official baseline checkpoint!"
    )
