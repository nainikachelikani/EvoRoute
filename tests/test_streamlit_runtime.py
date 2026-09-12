import os
import sys
import json
from pathlib import Path
import torch
import pandas as pd

# Add workspace root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
    FINAL_METRICS_PATH,
    MEMORY_STUDY_PATH,
    NOVELTY_METRICS_PATH,
    CATEGORIES,
    DEVICE
)
from src.model import EvoMLP
from src.novelty import NoveltyDetector
from src.embeddings import encode_texts, load_embeddings
from src.tasks import get_task_data

def test_streamlit_runtime_verification():
    print("=================================================================")
    print("       STARTING EXTENSIVE STREAMLIT RUNTIME VERIFICATION         ")
    print("=================================================================")
    
    # Check 1: Required model files exist
    required_models = ["naive_final.pt", "replay_final.pt", "replay_ewc_final.pt", "lwf_final.pt", "ewc_final.pt", "joint_final.pt"]
    print("\n[CHECK 1] Verifying all required model files exist...")
    for m in required_models:
        model_path = MODELS_DIR / m
        assert model_path.exists(), f"Missing required model checkpoint: {model_path}"
        # Validate that model loads properly into EvoMLP
        model = EvoMLP(num_classes=4).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        # Test forward pass with dummy vector
        dummy = torch.randn(1, 384, device=DEVICE)
        out = model(dummy)
        assert out.shape == (1, 4), f"Unexpected model output shape for {m}: {out.shape}"
        print(f"  [OK] {m} verified: exists, loads state_dict, forward pass shape (1, 4) OK.")

    # Check 2: Required metrics JSON files exist and have valid structure
    print("\n[CHECK 2] Verifying all metrics JSON files exist...")
    assert FINAL_METRICS_PATH.exists(), f"Missing {FINAL_METRICS_PATH}"
    with open(FINAL_METRICS_PATH, "r") as f:
        final_metrics = json.load(f)
    for method_key in ["naive", "replay", "replay_ewc", "lwf", "ewc", "joint"]:
        assert method_key in final_metrics, f"Missing method {method_key} in final_metrics.json"
        data = final_metrics[method_key]
        assert "overall_accuracy" in data, f"Missing overall_accuracy for {method_key}"
        assert "final_avg_task_accuracy" in data, f"Missing final_avg_task_accuracy for {method_key}"
        assert "average_forgetting" in data, f"Missing average_forgetting for {method_key}"
        assert "final_per_class" in data, f"Missing final_per_class for {method_key}"
        print(f"  [OK] final_results.json -> {method_key} metrics loaded successfully.")

    assert MEMORY_STUDY_PATH.exists(), f"Missing {MEMORY_STUDY_PATH}"
    with open(MEMORY_STUDY_PATH, "r") as f:
        memory_study = json.load(f)
    for b in ["0", "50", "100", "200"]:
        assert b in memory_study, f"Missing budget {b} in memory_sensitivity.json"
        print(f"  [OK] memory_sensitivity.json -> Budget {b}: {memory_study[b]}")

    assert NOVELTY_METRICS_PATH.exists(), f"Missing {NOVELTY_METRICS_PATH}"
    with open(NOVELTY_METRICS_PATH, "r") as f:
        novelty_metrics = json.load(f)
    assert "milestone_1_electronics" in novelty_metrics, "Missing milestone_1_electronics"
    assert "milestone_2_household" in novelty_metrics, "Missing milestone_2_household"
    print(f"  [OK] novelty_metrics.json verified.")

    # Check 3: Dashboard initialization and page logic
    print("\n[CHECK 3] Verifying dashboard component initialization...")
    detector = NoveltyDetector()
    t1_embs, t1_lbls, _ = get_task_data(task_id=1, split="train", cumulative=True)
    val_data = load_embeddings("val")
    detector.update_known_classes(
        new_classes=[0, 1],
        embeddings=t1_embs,
        labels=t1_lbls,
        val_embeddings=val_data["embeddings"],
        val_labels=val_data["labels"]
    )
    assert detector.centroids is not None
    assert detector.threshold is not None
    print(f"  [OK] NoveltyDetector initialized on Task 1 with threshold tau = {detector.threshold:.4f}")

    # Check 4: Simulate Page 1 (Overview & Live Routing)
    print("\n[CHECK 4] Testing Page 1 execution (Live Routing)...")
    sample_text = "Men's Regular Fit Cotton Casual Shirt with Button-Down Collar"
    emb = encode_texts([sample_text], show_progress=False).to(DEVICE)
    nov_res = detector.detect(emb.cpu())
    active_model = EvoMLP(num_classes=4).to(DEVICE)
    active_model.load_state_dict(torch.load(MODELS_DIR / "replay_ewc_final.pt", map_location=DEVICE))
    active_model.eval()
    with torch.no_grad():
        logits = active_model(emb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_cat = CATEGORIES[int(probs.argmax())]
    print(f"  [OK] Page 1 Live Routing executed -> Pred: {pred_cat}, Confidence: {probs.max()*100:.1f}%, Novelty: {nov_res['decision']}")

    # Check 5: Simulate Page 2 (Evolution Simulation)
    print("\n[CHECK 5] Testing Page 2 execution (Evolution Simulation)...")
    tech_text = "Sony WH-1000XM5 Wireless Headphones"
    emb_tech = encode_texts([tech_text], show_progress=False)
    det_tech = detector.detect(emb_tech)
    print(f"  [OK] Page 2 Phase 2 check -> Distance: {det_tech['distance']:.4f}, Novel: {det_tech['is_novel']}")

    home_text = "Prestige Deluxe Stainless Steel Pressure Cooker"
    emb_home = encode_texts([home_text], show_progress=False)
    det_home = detector.detect(emb_home)
    print(f"  [OK] Page 2 Phase 3 check -> Distance: {det_home['distance']:.4f}, Novel: {det_home['is_novel']}")

    # Check 6: Simulate Page 3 (Benchmark Results)
    print("\n[CHECK 6] Testing Page 3 execution (Benchmark Results)...")
    method_order = [
        ("naive", "Naive Sequential", "0"),
        ("ewc", "EWC", "0"),
        ("replay", "Experience Replay", "200"),
        ("lwf", "LwF", "0"),
        ("replay_ewc", "Replay + EWC (Proposed)", "200"),
        ("joint", "Joint Upper Bound", "Full Dataset"),
    ]
    rows = []
    for key, name, mem in method_order:
        m_data = final_metrics[key]
        rows.append({
            "Method": name,
            "Overall Class Accuracy": f"{m_data['overall_accuracy'] * 100:.2f}%",
            "Final Avg Task Accuracy": f"{m_data['final_avg_task_accuracy'] * 100:.2f}%",
            "Avg Forgetting": f"{m_data['average_forgetting'] * 100:.2f}%",
            "Memory": mem
        })
    df_table = pd.DataFrame(rows)
    print(df_table.to_string(index=False))
    print("  [OK] Page 3 Benchmark Table rendered without errors.")

    # Check 7: Simulate Page 4 (Experimental Analysis Plots & Data)
    print("\n[CHECK 7] Testing Page 4 execution (Experimental Analysis)...")
    required_plots = [
        "accuracy_across_tasks.png",
        "catastrophic_forgetting.png",
        "per_class_accuracy.png",
        "memory_vs_accuracy.png",
        "memory_vs_forgetting.png",
        "knowledge_retention_heatmap.png",
        "novelty_detection_distribution.png",
        "lwf_retention_analysis.png"
    ]
    for p in required_plots:
        plot_file = RESULTS_PLOTS_DIR / p
        assert plot_file.exists(), f"Missing required plot: {plot_file}"
        assert plot_file.stat().st_size > 1000, f"Plot {plot_file} is suspiciously small ({plot_file.stat().st_size} bytes)"
        print(f"  [OK] Plot verified: {p} ({plot_file.stat().st_size} bytes)")

    print("\n=================================================================")
    print("   ALL STREAMLIT RUNTIME VERIFICATION CHECKS PASSED PERFECTLY!   ")
    print("=================================================================")

if __name__ == "__main__":
    test_streamlit_runtime_verification()
