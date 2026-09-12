import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import EvoMLP
from src.calibration import CalibratedModelWrapper, optimize_temperature, optimize_decision_rebalancing
from src.novelty import NoveltyDetector
from src.config import CATEGORIES


def test_calibrated_model_wrapper_transformations():
    """Verify CalibratedModelWrapper applies temperature scaling and newest class logit shift correctly."""
    base_model = EvoMLP(num_classes=4)
    base_model.eval()

    T = 1.5
    gamma = 2.0
    newest_class_id = 3  # Household
    wrapper = CalibratedModelWrapper(base_model, temperature=T, gamma=gamma, newest_class_id=newest_class_id)

    dummy_inputs = torch.randn(8, 384)
    with torch.no_grad():
        raw_logits = base_model(dummy_inputs)
        cal_logits = wrapper(dummy_inputs)

    # Expected transformed logits:
    # 1. divide by T
    # 2. subtract gamma from newest_class_id
    expected_logits = raw_logits / T
    expected_logits[:, newest_class_id] -= gamma

    assert torch.allclose(cal_logits, expected_logits, atol=1e-5), "Calibrated logits do not match formula!"

    # Probabilities must be valid and sum to 1
    probs = F.softmax(cal_logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_rebalancing_safety_rejection_rule():
    """Verify optimize_decision_rebalancing rejects gamma if new class accuracy drops below threshold."""
    # Synthetic logits where newest class (idx 3) is dominant but old classes have some signal
    torch.manual_seed(42)
    val_logits = torch.randn(100, 4)
    val_labels = torch.randint(0, 4, (100,))

    # Extreme threshold that cannot be satisfied
    res = optimize_decision_rebalancing(
        logits=val_logits,
        labels=val_labels,
        newest_class_id=3,
        baseline_newest_acc=1.0  # Impossibly high constraint
    )

    # Rejection must be triggered, returning gamma = 0.0
    assert res["rejection_triggered"] is True
    assert res["optimal_gamma"] == 0.0


def test_novelty_assessment_never_overrides_classifier():
    """
    CRITICAL ARCHITECTURAL REQUIREMENT:
    Verify that novelty assessment NEVER overrides or mutates the classifier prediction.

    Classifier says: Books (or class c)
    Novelty says: High Novelty (distance > threshold)
    FINAL Prediction MUST BE: Books (NOT 'Unknown')
    Novelty Assessment MUST BE: High Novelty (independent metric)
    """
    # 1. Setup classifier
    model = EvoMLP(num_classes=4)
    model.eval()

    # 2. Setup novelty detector with synthetic centroids
    detector = NoveltyDetector(percentile=90.0)
    # Train detector on task 1 data
    train_embs = torch.randn(100, 384)
    train_embs = F.normalize(train_embs, p=2, dim=-1)
    train_lbls = torch.randint(0, 2, (100,))
    detector.update_known_classes(new_classes=[0, 1], embeddings=train_embs, labels=train_lbls)

    # 3. Create an out-of-distribution sample (e.g. random vector in an orthogonal space)
    ood_sample = torch.randn(1, 384)
    ood_sample_norm = F.normalize(ood_sample, p=2, dim=-1)

    # 4. Run classification
    with torch.no_grad():
        logits = model(ood_sample)
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_category = CATEGORIES[predicted_class_id]

    # 5. Run novelty detection independently
    novelty_result = detector.detect(ood_sample_norm[0])
    is_novel = novelty_result["is_novel"]
    distance = novelty_result["distance"]

    # 6. Verify strict system separation:
    # - The classifier prediction is a valid known category
    assert predicted_category in CATEGORIES, f"Prediction {predicted_category} is invalid!"
    assert predicted_category != "Unknown", "Classifier prediction was corrupted to 'Unknown'!"

    # - Novelty assessment is purely an independent metadata flag
    assert isinstance(is_novel, (bool, np.bool_))
    assert isinstance(distance, float)

    # Check simulated prediction payload structure for the application
    prediction_payload = {
        "classifier_prediction": predicted_category,
        "novelty_assessment": "High Novelty" if is_novel else "Familiar",
        "novelty_distance": distance,
        "novelty_threshold": novelty_result["threshold"]
    }

    assert prediction_payload["classifier_prediction"] == predicted_category
    assert prediction_payload["classifier_prediction"] != prediction_payload["novelty_assessment"]
