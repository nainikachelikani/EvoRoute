import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import EvoMLP

def test_dynamic_expansion_invariance():
    print("Testing EvoMLP dynamic output expansion weight invariance...")
    model = EvoMLP(num_classes=2)
    
    # Snapshot original weights and biases
    orig_w = model.head.weight.data.clone()
    orig_b = model.head.bias.data.clone()
    
    # Expand from 2 to 3 classes
    model.expand_classes(3)
    assert model.num_classes == 3
    assert model.head.weight.shape == (3, 128)
    assert model.head.bias.shape == (3,)
    assert torch.equal(model.head.weight[:2], orig_w)
    assert torch.equal(model.head.bias[:2], orig_b)
    
    # Expand from 3 to 4 classes
    w3 = model.head.weight.data.clone()
    b3 = model.head.bias.data.clone()
    model.expand_classes(4)
    assert model.num_classes == 4
    assert model.head.weight.shape == (4, 128)
    assert model.head.bias.shape == (4,)
    assert torch.equal(model.head.weight[:3], w3)
    assert torch.equal(model.head.bias[:3], b3)
    
    print("[OK] Dynamic output expansion preserves prior parameters with bit-for-bit invariance.")

if __name__ == "__main__":
    test_dynamic_expansion_invariance()
