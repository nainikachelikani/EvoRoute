import math
from pathlib import Path
import torch
import numpy as np
import pytest

from src.config import CATEGORIES, DEVICE, MODELS_DIR, RESULTS_PLOTS_DIR
from src.model import EvoMLP
from src.evaluate import compute_prediction_distribution
from src.visualize import plot_recency_bias_collapse


def test_prediction_distribution_structure_and_sum():
    """Verify that compute_prediction_distribution returns valid probabilities summing to 1.0."""
    model = EvoMLP(num_classes=4).to(DEVICE)
    model.eval()

    res = compute_prediction_distribution(model, split="test", device=DEVICE)

    assert "distribution" in res
    assert "recency_bias" in res
    assert "newest_class" in res
    assert res["newest_class"] == "Household"
    assert res["total_samples"] > 0

    dist = res["distribution"]
    # Verify all classes are represented
    for cat in CATEGORIES:
        assert cat in dist, f"Missing category {cat} in prediction distribution"
        assert not math.isnan(dist[cat]), f"NaN encountered in distribution for {cat}"
        assert not math.isinf(dist[cat]), f"Inf encountered in distribution for {cat}"
        assert 0.0 <= dist[cat] <= 1.0, f"Probability {dist[cat]} out of bounds [0, 1]"

    # Verify probabilities sum to 1.0 within numerical tolerance
    total_prob = sum(dist.values())
    assert abs(total_prob - 1.0) < 1e-5, f"Probabilities sum to {total_prob}, expected 1.0"


def test_recency_bias_metric_formula():
    """Verify recency bias is correctly computed as P(Newest Class) - 0.25."""
    model = EvoMLP(num_classes=4).to(DEVICE)
    model.eval()

    res = compute_prediction_distribution(model, split="test", device=DEVICE)
    dist = res["distribution"]
    p_newest = dist["Household"]
    expected_rb = p_newest - 0.25

    assert abs(res["recency_bias"] - expected_rb) < 1e-6, (
        f"Recency bias {res['recency_bias']} != expected {expected_rb}"
    )
    assert not math.isnan(res["recency_bias"])
    assert -0.25 <= res["recency_bias"] <= 0.75


def test_recency_bias_plot_generation(tmp_path):
    """Verify plot_recency_bias_collapse generates a valid image without crashing."""
    mock_results = {
        "naive": {
            "prediction_distribution": {"Books": 0.0, "Clothing & Accessories": 0.0, "Electronics": 0.0, "Household": 1.0},
            "recency_bias": 0.75
        },
        "replay_ewc": {
            "prediction_distribution": {"Books": 0.24, "Clothing & Accessories": 0.26, "Electronics": 0.24, "Household": 0.26},
            "recency_bias": 0.01
        }
    }
    temp_plot = tmp_path / "recency_bias_collapse.png"
    plot_recency_bias_collapse(mock_results, save_path=temp_plot)

    assert temp_plot.exists()
    assert temp_plot.stat().st_size > 1000
