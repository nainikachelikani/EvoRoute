import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CATEGORIES,
    ID2CATEGORY,
    CATEGORY2ID,
    MODELS_DIR,
    DEVICE
)
from src.embeddings import encode_texts, load_embeddings
from src.model import EvoMLP


QUALITATIVE_DEMO_QUERIES = [
    {"query": "Fantasy novel about young wizards learning magic at school", "expected": "Books"},
    {"query": "Black leather jacket with zip closure for men", "expected": "Clothing & Accessories"},
    {"query": "Wireless Bluetooth headphones with active noise cancellation", "expected": "Electronics"},
    {"query": "Stainless steel kitchen spoon for cooking and serving", "expected": "Household"}
]


def debug_pipeline():
    """
    Traces the entire prediction pipeline for canonical queries:
    Text -> 384-D Unit Vector -> Cosine Distance to Class Centroids -> MLP Logits -> Softmax Probabilities.
    Diagnoses whether recency bias originates in semantic embeddings or MLP output head scaling.
    """
    print("=" * 95)
    print("      EVOROUTE PREDICTION PIPELINE DEBUG & ROOT CAUSE TRACE")
    print("=" * 95)
    print("Note: These queries are qualitative demonstrations, not benchmark evaluation data.\n")

    # 1. Compute class centroids from validation split embeddings
    val_data = load_embeddings("val")
    val_embs = val_data["embeddings"]
    val_lbls = val_data["labels"]

    centroids = {}
    for c_id, c_name in enumerate(CATEGORIES):
        mask = (val_lbls == c_id)
        if mask.sum() > 0:
            c_embs = F.normalize(val_embs[mask], p=2, dim=1)
            centroids[c_id] = c_embs.mean(dim=0, keepdim=True)
            centroids[c_id] = F.normalize(centroids[c_id], p=2, dim=1)

    # 2. Load frozen Replay + EWC baseline model
    model_path = MODELS_DIR / "replay_ewc_final.pt"
    if not model_path.exists():
        print(f"Error: Baseline checkpoint {model_path} not found.")
        return

    model = EvoMLP(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Trace each query
    print(f"{'Expected':<22} | {'Nearest Centroid (Cosine)':<26} | {'MLP Classifier Pred':<22} | {'Logits [B, C, E, H]':<20}")
    print("-" * 95)

    discrepancy_count = 0
    for item in QUALITATIVE_DEMO_QUERIES:
        q_text = item["query"]
        expected_cat = item["expected"]

        # Encode text
        emb = encode_texts([q_text], show_progress=False)
        emb_norm = F.normalize(emb, p=2, dim=1)

        # Nearest centroid
        sims = {c_id: float(torch.matmul(emb_norm, centroids[c_id].T).item()) for c_id in centroids}
        nearest_cid = max(sims, key=sims.get)
        nearest_cat = ID2CATEGORY[nearest_cid]
        nearest_sim = sims[nearest_cid]

        # MLP prediction
        with torch.no_grad():
            logits = model(emb.to(DEVICE)).cpu().numpy()[0]
            probs = F.softmax(torch.tensor(logits), dim=0).numpy()
            mlp_cid = int(np.argmax(logits))
            mlp_cat = ID2CATEGORY[mlp_cid]
            mlp_prob = probs[mlp_cid]

        logits_str = f"[{logits[0]:.1f}, {logits[1]:.1f}, {logits[2]:.1f}, {logits[3]:.1f}]"
        nearest_str = f"{nearest_cat[:18]} ({nearest_sim:.2f})"
        mlp_str = f"{mlp_cat[:18]} ({mlp_prob*100:.0f}%)"

        print(f"{expected_cat:<22} | {nearest_str:<26} | {mlp_str:<22} | {logits_str:<20}")

        if nearest_cat == expected_cat and mlp_cat != expected_cat:
            discrepancy_count += 1
            print(f"  >>> DIAGNOSTIC SIGNAL: Embeddings correctly identify '{expected_cat}', but MLP logits favor '{mlp_cat}'!")
            print(f"      Logit breakdown: Books={logits[0]:.2f}, Clothing={logits[1]:.2f}, Electronics={logits[2]:.2f}, Household={logits[3]:.2f}")

    print("=" * 95)
    print(f"Summary: {discrepancy_count} / {len(QUALITATIVE_DEMO_QUERIES)} queries show Classifier Bias where Embeddings are correct.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    debug_pipeline()
