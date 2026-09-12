import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model import EvoMLP
from src.evaluate import evaluate_model_on_classes
from src.config import TASK_CUMULATIVE_CLASSES

methods = ["naive", "ewc", "replay", "lwf", "replay_ewc", "evoroute_br_candidate"]

for m in methods:
    print(f"=== {m} ===")
    for t in [1, 2, 3]:
        p = Path(f"models/{m}_task{t}.pt")
        if p.exists():
            n_cls = len(TASK_CUMULATIVE_CLASSES[t])
            model = EvoMLP(num_classes=n_cls)
            model.load_state_dict(torch.load(p, map_location="cpu"))
            res = evaluate_model_on_classes(model, split="test", allowed_classes=TASK_CUMULATIVE_CLASSES[t])
            print(f"  Task {t}: overall = {res['accuracy']*100:.2f}%, per_class = {res['per_class_accuracy']}")
        else:
            print(f"  Task {t}: file {p} not found")
