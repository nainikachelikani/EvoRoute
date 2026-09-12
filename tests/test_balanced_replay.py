import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.replay import ReplayBuffer, construct_incremental_batch


def test_balanced_replay_dynamic_remainder_and_batch_size():
    """Verify sample_class_balanced allocates remainder dynamically and always returns requested batch size."""
    buf = ReplayBuffer(max_budget=200)

    # Populate buffer with 3 classes (like at start of Task 3: Classes 0, 1, 2)
    embs_c0 = torch.randn(60, 384)
    lbls_c0 = torch.zeros(60, dtype=torch.long)
    embs_c1 = torch.randn(60, 384)
    lbls_c1 = torch.ones(60, dtype=torch.long)
    embs_c2 = torch.randn(60, 384)
    lbls_c2 = torch.full((60,), 2, dtype=torch.long)

    buf.add_examples(embs_c0, lbls_c0)
    buf.add_examples(embs_c1, lbls_c1)
    buf.add_examples(embs_c2, lbls_c2)

    assert buf.total_samples <= 200

    # Sample 64 items across 3 classes: 64 // 3 = 21, remainder 1 -> counts should be [22, 21, 21] (or permuted)
    s_embs, s_lbls = buf.sample_class_balanced(num_samples=64)
    assert s_embs.shape == (64, 384)
    assert s_lbls.shape == (64,)

    unique_counts = torch.bincount(s_lbls, minlength=3).tolist()
    assert sum(unique_counts) == 64
    assert sorted(unique_counts) == [21, 21, 22]


def test_balanced_replay_four_classes_exact_split():
    """Verify 4 classes split 64 samples into exactly 16 per class."""
    buf = ReplayBuffer(max_budget=200)
    for c in range(4):
        buf.add_examples(torch.randn(50, 384), torch.full((50,), c, dtype=torch.long))

    assert buf.total_samples == 200

    s_embs, s_lbls = buf.sample_class_balanced(num_samples=64)
    assert s_embs.shape == (64, 384)
    assert s_lbls.shape == (64,)

    counts = torch.bincount(s_lbls, minlength=4).tolist()
    assert counts == [16, 16, 16, 16]


def test_construct_incremental_batch_strategies():
    """Verify construct_incremental_batch handles original, 50_50, and class_balanced."""
    buf = ReplayBuffer(max_budget=200)
    for c in range(3):
        buf.add_examples(torch.randn(50, 384), torch.full((50,), c, dtype=torch.long))

    curr_embs = torch.randn(64, 384)
    curr_lbls = torch.full((64,), 3, dtype=torch.long)

    # 1. Original (current + sample(64 * 0.2) = 64 + 12 = 76)
    b_embs_orig, b_lbls_orig = construct_incremental_batch(curr_embs, curr_lbls, buf, strategy="original", batch_size=64)
    assert len(b_lbls_orig) == 76

    # 2. 50_50 (32 current + 32 replay = 64)
    b_embs_5050, b_lbls_5050 = construct_incremental_batch(curr_embs, curr_lbls, buf, strategy="50_50", batch_size=64)
    assert len(b_lbls_5050) == 64
    assert (b_lbls_5050 == 3).sum().item() == 32
    assert (b_lbls_5050 != 3).sum().item() == 32

    # 3. class_balanced (seen_classes = [0, 1, 2, 3] -> 16 each)
    b_embs_cb, b_lbls_cb = construct_incremental_batch(curr_embs, curr_lbls, buf, strategy="class_balanced", batch_size=64)
    assert len(b_lbls_cb) == 64
    counts_cb = torch.bincount(b_lbls_cb, minlength=4).tolist()
    assert counts_cb == [16, 16, 16, 16]
