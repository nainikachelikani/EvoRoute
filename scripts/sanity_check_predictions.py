import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEVICE,
    ID2CATEGORY,
    CATEGORIES,
    BASELINE_CHECKPOINT_PATH,
    EVOROUTE_BR_CHECKPOINT_PATH,
    EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH,
    SANITY_CHECK_RESULTS_PATH,
    CONFIG_MANIFEST_PATH
)
from src.model import EvoMLP
from src.embeddings import encode_texts
from src.tasks import get_task_data
from src.novelty import NoveltyDetector
from src.calibration import CalibratedModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CURATED_TEST_QUERIES = [
    # Books (Task 1)
    {"id": 1, "text": "The Pragmatic Programmer: Your Journey to Mastery (20th Anniversary Edition) Paperback", "true_category": "Books"},
    {"id": 2, "text": "A Brief History of Time by Stephen Hawking - Illustrated Hardcover Scientific Book", "true_category": "Books"},
    {"id": 3, "text": "Fantasy adventure novel about young wizards and magical creatures in ancient kingdom", "true_category": "Books"},
    # Clothing (Task 1)
    {"id": 4, "text": "Men's Regular Fit Cotton Casual Shirt with Button-Down Collar and Long Sleeves", "true_category": "Clothing & Accessories"},
    {"id": 5, "text": "Women's High-Waisted Stretch Skinny Jeans Denim Pants with 5 Pockets", "true_category": "Clothing & Accessories"},
    {"id": 6, "text": "Breathable running athletic sports shoes sneakers for marathon training", "true_category": "Clothing & Accessories"},
    # Electronics (Task 2)
    {"id": 7, "text": "Sony WH-1000XM5 Wireless Noise Canceling Over-Ear Headphones with Bluetooth", "true_category": "Electronics"},
    {"id": 8, "text": "ASUS ROG Strix GeForce RTX 4080 16GB GDDR6X Gaming Graphics Card", "true_category": "Electronics"},
    {"id": 9, "text": "Apple iPad Pro 12.9-inch Liquid Retina XDR Display M2 Chip 256GB Space Gray", "true_category": "Electronics"},
    # Household (Task 3)
    {"id": 10, "text": "Prestige Electric Kettle 1.5L Stainless Steel Body with Auto Cut-Off", "true_category": "Household"},
    {"id": 11, "text": "Cotton King Size Bedsheet with 2 Pillow Covers Floral Print Elastic Fitted", "true_category": "Household"},
    {"id": 12, "text": "Non-Stick 3-Piece Cookware Set Fry Pan Kadhai Dosa Tawa Induction Bottom", "true_category": "Household"},
]

def run_sanity_checks():
    logger.info("Initializing models for sanity check predictions...")
    
    # 1. Baseline Replay + EWC
    baseline = EvoMLP(num_classes=4).to(DEVICE)
    baseline.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    baseline.eval()

    # 2. EvoRoute-BR Candidate
    candidate = EvoMLP(num_classes=4).to(DEVICE)
    candidate.load_state_dict(torch.load(EVOROUTE_BR_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    candidate.eval()

    # 3. EvoRoute-BR Calibrated
    cal_ckpt = torch.load(EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cal_base = EvoMLP(num_classes=4).to(DEVICE)
    cal_base.load_state_dict(cal_ckpt["base_model_state"])
    cal_base.eval()
    cal_state = cal_ckpt["calibration_state"]
    calibrated = CalibratedModelWrapper(
        base_model=cal_base,
        temperature=cal_state["temperature"],
        gamma=cal_state["gamma"],
        newest_class_id=cal_state["newest_class_id"],
        num_classes=cal_state["num_classes"]
    )
    calibrated.eval()

    # 4. Novelty Detector (Trained on training embeddings, calibrated on validation)
    detector = NoveltyDetector()
    train_embs, train_lbls, _ = get_task_data(task_id=3, split="train", cumulative=True)
    val_embs, val_lbls, _ = get_task_data(task_id=3, split="val", cumulative=True)
    detector.update_known_classes(
        new_classes=[0, 1, 2, 3],
        embeddings=train_embs,
        labels=train_lbls,
        val_embeddings=val_embs,
        val_labels=val_lbls
    )

    query_results = []
    print("\n" + "="*85)
    print("EVOROUTE QUALITATIVE SANITY CHECK: 12 CANONICAL QUERIES")
    print("="*85)

    for item in CURATED_TEST_QUERIES:
        q_text = item["text"]
        true_cat = item["true_category"]

        # Encode query
        emb = encode_texts([q_text], show_progress=False).to(DEVICE)

        # Baseline inference
        with torch.no_grad():
            b_logits = baseline(emb)
            b_probs = F.softmax(b_logits, dim=1).cpu().numpy()[0]
            b_pred_id = int(np.argmax(b_probs))
            b_pred_cat = ID2CATEGORY[b_pred_id]
            b_conf = float(b_probs[b_pred_id])

        # Candidate inference
        with torch.no_grad():
            c_logits = candidate(emb)
            c_probs = F.softmax(c_logits, dim=1).cpu().numpy()[0]
            c_pred_id = int(np.argmax(c_probs))
            c_pred_cat = ID2CATEGORY[c_pred_id]
            c_conf = float(c_probs[c_pred_id])

        # Calibrated inference
        with torch.no_grad():
            cal_logits = calibrated(emb)
            cal_probs = F.softmax(cal_logits, dim=1).cpu().numpy()[0]
            cal_pred_id = int(np.argmax(cal_probs))
            cal_pred_cat = ID2CATEGORY[cal_pred_id]
            cal_conf = float(cal_probs[cal_pred_id])

        # Independent Novelty Detection (Never overrides classifier)
        nov_res = detector.detect(emb.cpu())
        nov_assessment = {
            "is_novel": nov_res["is_novel"],
            "distance_to_centroid": nov_res["distance"],
            "calibrated_threshold": nov_res["threshold"],
            "nearest_semantic_category": nov_res["nearest_category_name"],
            "novelty_verdict": "HIGH NOVELTY / UNKNOWN" if nov_res["is_novel"] else "FAMILIAR / KNOWN",
            "separation_guarantee": "Novelty assessment does NOT mutate or override classifier prediction."
        }

        # Print cleanly formatted report
        print(f"\nQuery #{item['id']}: \"{q_text[:70]}...\"")
        print(f"  Ground Truth:           {true_cat}")
        print(f"  Replay + EWC Baseline:  {b_pred_cat} (Conf: {b_conf*100:.1f}%) {'[WRONG]' if b_pred_cat != true_cat else '[OK]'}")
        print(f"  EvoRoute-BR Candidate:  {c_pred_cat} (Conf: {c_conf*100:.1f}%) {'[WRONG]' if c_pred_cat != true_cat else '[OK]'}")
        print(f"  EvoRoute-BR Calibrated: {cal_pred_cat} (Conf: {cal_conf*100:.1f}%) {'[WRONG]' if cal_pred_cat != true_cat else '[OK]'}")
        print(f"  Novelty Assessment:     {nov_assessment['novelty_verdict']} (dist: {nov_res['distance']:.3f} vs tau: {nov_res['threshold']:.3f})")

        query_results.append({
            "id": item["id"],
            "query_text": q_text,
            "ground_truth_category": true_cat,
            "classifier_predictions": {
                "baseline_replay_ewc": {
                    "predicted_category": b_pred_cat,
                    "confidence": b_conf,
                    "is_correct": b_pred_cat == true_cat,
                    "probabilities": {ID2CATEGORY[i]: float(b_probs[i]) for i in range(4)}
                },
                "evoroute_br_candidate": {
                    "predicted_category": c_pred_cat,
                    "confidence": c_conf,
                    "is_correct": c_pred_cat == true_cat,
                    "probabilities": {ID2CATEGORY[i]: float(c_probs[i]) for i in range(4)}
                },
                "evoroute_br_calibrated": {
                    "predicted_category": cal_pred_cat,
                    "confidence": cal_conf,
                    "is_correct": cal_pred_cat == true_cat,
                    "probabilities": {ID2CATEGORY[i]: float(cal_probs[i]) for i in range(4)}
                }
            },
            "novelty_assessment_independent": nov_assessment
        })

    # Save to json artifact
    with open(SANITY_CHECK_RESULTS_PATH, "w") as f:
        json.dump(query_results, f, indent=2)
    logger.info(f"\nSaved sanity check results to {SANITY_CHECK_RESULTS_PATH}")

if __name__ == "__main__":
    run_sanity_checks()
