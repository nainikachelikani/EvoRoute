import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import EvoMLP
from src.ewc import EWC

class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"embedding": self.x[idx], "label": self.y[idx]}

def test_ewc_fisher_and_penalty():
    print("Testing EWC Fisher Information accumulation and loss penalty...")
    model = EvoMLP(num_classes=2)
    ewc = EWC(model, ewc_lambda=100.0)
    
    # Create synthetic dataset for Task 1
    x = torch.randn(100, 384)
    y = torch.randint(0, 2, (100,))
    loader = DataLoader(DictDataset(x, y), batch_size=20)
    
    # Compute Fisher
    ewc.compute_fisher(loader, num_samples=100)
    assert 1 in ewc.fisher_matrix
    assert 1 in ewc.optimal_params
    
    # Check that penalty on identical parameters is 0
    pen_0 = ewc.compute_penalty(model).item()
    assert abs(pen_0) < 1e-6, f"Expected 0 penalty for identical weights, got {pen_0}"
    
    # Perturb weights and check that penalty > 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
            
    pen_perturbed = ewc.compute_penalty(model).item()
    assert pen_perturbed > 0, f"Expected positive penalty after perturbation, got {pen_perturbed}"
    
    print(f"[OK] EWC Fisher computed and penalty responds as expected (perturbation penalty = {pen_perturbed:.4f}).")

if __name__ == "__main__":
    test_ewc_fisher_and_penalty()
