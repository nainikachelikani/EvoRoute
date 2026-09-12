import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.replay import ReplayBuffer

def test_replay_budget_and_rebalancing():
    print("Testing ReplayBuffer capacity constraints and rebalancing...")
    buf = ReplayBuffer(max_budget=100)
    
    # Add 200 samples for Class 0
    embs_c0 = torch.randn(200, 384)
    lbls_c0 = torch.zeros(200, dtype=torch.long)
    buf.add_examples(embs_c0, lbls_c0)
    
    # Budget is 100, 1 class -> should have 100 samples
    assert buf.total_samples == 100
    assert len(buf.memory[0]["labels"]) == 100
    
    # Add 200 samples for Class 1 -> should rebalance to 50 each
    embs_c1 = torch.randn(200, 384)
    lbls_c1 = torch.ones(200, dtype=torch.long)
    buf.add_examples(embs_c1, lbls_c1)
    
    assert buf.total_samples == 100
    assert len(buf.memory[0]["labels"]) == 50
    assert len(buf.memory[1]["labels"]) == 50
    
    # Sample batch
    s_embs, s_lbls = buf.sample(num_samples=32)
    assert s_embs.shape == (32, 384)
    assert s_lbls.shape == (32,)
    assert set(s_lbls.tolist()).issubset({0, 1})
    
    print("[OK] ReplayBuffer strictly enforces budget (<= 100) and balances class counts.")

if __name__ == "__main__":
    test_replay_budget_and_rebalancing()
