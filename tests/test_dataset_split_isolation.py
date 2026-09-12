import sys
from pathlib import Path
import pytest
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLEANED_CSV_PATH,
    TRAIN_EMB_PATH,
    VAL_EMB_PATH,
    TEST_EMB_PATH
)
from src.data import split_data
from src.embeddings import load_embeddings


def test_split_isolation_on_dataframe():
    """Asserts that dataframe split logic produces strictly disjoint train, val, and test sets."""
    if not CLEANED_CSV_PATH.exists():
        pytest.skip("cleaned_data.csv does not exist yet. Run preprocess stage first.")

    df = pd.read_csv(CLEANED_CSV_PATH)
    train_df, val_df, test_df = split_data(df)

    train_texts = set(train_df["description"].str.strip().tolist())
    val_texts = set(val_df["description"].str.strip().tolist())
    test_texts = set(test_df["description"].str.strip().tolist())

    assert train_texts.isdisjoint(val_texts), "Data leakage detected: Train and Val sets overlap!"
    assert train_texts.isdisjoint(test_texts), "Data leakage detected: Train and Test sets overlap!"
    assert val_texts.isdisjoint(test_texts), "Data leakage detected: Val and Test sets overlap!"


def test_split_isolation_on_saved_embeddings():
    """Asserts that precomputed embeddings files have zero text or index overlap."""
    if not (TRAIN_EMB_PATH.exists() and VAL_EMB_PATH.exists() and TEST_EMB_PATH.exists()):
        pytest.skip("Embedding files not yet generated. Run embeddings stage first.")

    train_data = load_embeddings("train")
    val_data = load_embeddings("val")
    test_data = load_embeddings("test")

    train_texts = set(train_data.get("texts", []))
    val_texts = set(val_data.get("texts", []))
    test_texts = set(test_data.get("texts", []))

    if train_texts and val_texts and test_texts:
        assert train_texts.isdisjoint(val_texts), "Data leakage: Train and Val embedding texts overlap!"
        assert train_texts.isdisjoint(test_texts), "Data leakage: Train and Test embedding texts overlap!"
        assert val_texts.isdisjoint(test_texts), "Data leakage: Val and Test embedding texts overlap!"

    # Verify sample counts (80% / 10% / 10%)
    n_train = len(train_data["labels"])
    n_val = len(val_data["labels"])
    n_test = len(test_data["labels"])
    n_total = n_train + n_val + n_test

    assert n_total > 0, "No samples found in precomputed embeddings!"
    assert n_val == n_test, f"Validation ({n_val}) and Test ({n_test}) sizes should match!"
    assert abs(n_train / n_total - 0.80) < 0.02, "Train split proportion diverges from 80%!"
