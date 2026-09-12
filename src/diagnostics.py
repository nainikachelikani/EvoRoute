import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CATEGORIES,
    ID2CATEGORY,
    MODELS_DIR,
    RESULTS_DIR,
    RESULTS_PLOTS_DIR,
    DEVICE
)
from src.embeddings import load_embeddings
from src.model import EvoMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAGNOSTICS_VAL_DIR = RESULTS_DIR / "diagnostics" / "validation"


def compute_embedding_diagnostics(val_embs: torch.Tensor, val_lbls: torch.Tensor) -> Dict[str, Any]:
    """
    Evaluates embedding space separability strictly on the validation split:
    - Centroid cosine similarity matrix
    - Intra-class vs inter-class similarity
    - Nearest centroid classification accuracy
    """
    val_embs_norm = F.normalize(val_embs, p=2, dim=1)
    num_classes = len(CATEGORIES)

    centroids = []
    for c in range(num_classes):
        mask = (val_lbls == c)
        if mask.sum() > 0:
            c_centroid = val_embs_norm[mask].mean(dim=0, keepdim=True)
            centroids.append(F.normalize(c_centroid, p=2, dim=1))
        else:
            centroids.append(torch.zeros(1, val_embs.size(1)))

    centroids_tensor = torch.cat(centroids, dim=0)  # (C, 384)

    # 1. Centroid-to-Centroid Cosine Similarity Matrix
    centroid_sim_matrix = torch.matmul(centroids_tensor, centroids_tensor.T).numpy().tolist()

    # 2. Nearest Centroid Classification on Validation Data
    # Cosine sim of all validation samples to all centroids
    sim_to_centroids = torch.matmul(val_embs_norm, centroids_tensor.T)  # (N, C)
    nearest_centroid_preds = torch.argmax(sim_to_centroids, dim=1)
    correct_count = int((nearest_centroid_preds == val_lbls).sum().item())
    nearest_centroid_acc = float(correct_count / len(val_lbls))

    # 3. Intra vs Inter class distances
    per_class_centroid_acc = {}
    for c in range(num_classes):
        mask = (val_lbls == c)
        if mask.sum() > 0:
            c_correct = int((nearest_centroid_preds[mask] == c).sum().item())
            per_class_centroid_acc[ID2CATEGORY[c]] = float(c_correct / mask.sum().item())

    # Intra-class mean cosine similarity to class centroid
    intra_class_sim = {}
    for c in range(num_classes):
        mask = (val_lbls == c)
        if mask.sum() > 0:
            c_sim = torch.matmul(val_embs_norm[mask], centroids_tensor[c:c+1].T).squeeze(1)
            intra_class_sim[ID2CATEGORY[c]] = float(c_sim.mean().item())

    embedding_results = {
        "nearest_centroid_accuracy": nearest_centroid_acc,
        "per_class_centroid_accuracy": per_class_centroid_acc,
        "intra_class_mean_similarity": intra_class_sim,
        "centroid_similarity_matrix": centroid_sim_matrix,
        "categories": CATEGORIES
    }

    DIAGNOSTICS_VAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGNOSTICS_VAL_DIR / "embedding_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(embedding_results, f, indent=4)

    logger.info(f"Saved embedding diagnostics to: {out_path}")
    logger.info(f"Validation Nearest Centroid Accuracy: {nearest_centroid_acc * 100:.2f}%")
    return embedding_results


def plot_embedding_space(val_embs: torch.Tensor, val_lbls: torch.Tensor, output_path: Path):
    """Generates PCA 2D scatter visualization of semantic embeddings with centroids."""
    val_embs_norm = F.normalize(val_embs, p=2, dim=1).numpy()
    labels = val_lbls.numpy()

    pca = PCA(n_components=2, random_state=42)
    embs_2d = pca.fit_transform(val_embs_norm)

    colors = ["#2563EB", "#16A34A", "#D97706", "#DC2626"]
    markers = ["o", "s", "^", "D"]

    plt.figure(figsize=(9, 7))
    for c_id, c_name in enumerate(CATEGORIES):
        mask = (labels == c_id)
        if mask.sum() > 0:
            plt.scatter(
                embs_2d[mask, 0],
                embs_2d[mask, 1],
                c=colors[c_id % len(colors)],
                marker=markers[c_id % len(markers)],
                label=f"{c_name} (n={mask.sum()})",
                alpha=0.6,
                edgecolors="none",
                s=35
            )

    plt.title("Semantic Embedding Space Diagnostic (Validation Set, PCA 2D)", fontsize=13, fontweight="bold")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved embedding space plot to: {output_path}")


def compute_classifier_head_diagnostics(model: EvoMLP, val_embs: torch.Tensor, val_lbls: torch.Tensor, device: str = DEVICE) -> Dict[str, Any]:
    """
    Inspects output classification layer weights, biases, and empirical validation logit distribution.
    Computes NormalizedLogitGap and per-class head statistics.
    """
    model.eval()
    head = model.head  # nn.Linear(128, num_classes)
    num_classes = model.num_classes

    # 1. Weights and biases analysis
    head_weights = head.weight.detach().cpu()  # (C, 128)
    head_biases = head.bias.detach().cpu()    # (C,)

    # 2. Forward pass on validation data
    with torch.no_grad():
        inputs = val_embs.to(device)
        logits = model(inputs).cpu()  # (N, C)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    all_logits_flat = logits.numpy().flatten()
    std_all_logits = float(np.std(all_logits_flat)) + 1e-6

    # Newest class is index 3 (or num_classes - 1)
    newest_cid = num_classes - 1
    old_cids = list(range(newest_cid))

    newest_logits = logits[:, newest_cid].numpy()
    old_logits = logits[:, old_cids].numpy()
    max_old_logits = np.max(old_logits, axis=1)

    mean_newest_logit = float(np.mean(newest_logits))
    mean_max_old_logit = float(np.mean(max_old_logits))
    raw_logit_gap = mean_newest_logit - mean_max_old_logit
    normalized_logit_gap = float(raw_logit_gap / std_all_logits)

    per_class_stats = {}
    for c_id, c_name in enumerate(CATEGORIES[:num_classes]):
        c_w = head_weights[c_id].numpy()
        c_logits = logits[:, c_id].numpy()
        c_probs = probs[:, c_id].numpy()
        c_preds_count = int((preds == c_id).sum().item())
        c_true_mask = (val_lbls == c_id).numpy()
        c_acc = float((preds.numpy()[c_true_mask] == c_id).mean()) if c_true_mask.sum() > 0 else 0.0

        per_class_stats[c_name] = {
            "weight_l2_norm": float(np.linalg.norm(c_w)),
            "weight_mean": float(np.mean(c_w)),
            "weight_std": float(np.std(c_w)),
            "bias_value": float(head_biases[c_id].item()),
            "mean_logit": float(np.mean(c_logits)),
            "std_logit": float(np.std(c_logits)),
            "mean_probability": float(np.mean(c_probs)),
            "predicted_frequency": float(c_preds_count / len(val_lbls)),
            "per_class_accuracy": c_acc
        }

    classifier_acc = float((preds == val_lbls).float().mean().item())

    head_results = {
        "classifier_overall_accuracy": classifier_acc,
        "mean_newest_logit": mean_newest_logit,
        "mean_max_old_logit": mean_max_old_logit,
        "raw_logit_gap": raw_logit_gap,
        "normalized_logit_gap": normalized_logit_gap,
        "std_all_logits": std_all_logits,
        "newest_class": CATEGORIES[newest_cid],
        "per_class_head_stats": per_class_stats
    }

    out_path = DIAGNOSTICS_VAL_DIR / "classifier_head_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(head_results, f, indent=4)

    logger.info(f"Saved classifier head diagnostics to: {out_path}")
    logger.info(f"Normalized Logit Gap (Newest vs Max Old): {normalized_logit_gap:+.3f}")
    return head_results


def plot_classifier_head_analysis(head_stats: Dict[str, Any], output_path: Path):
    """Generates bar chart comparing Output Bias values, Weight L2 Norms, and Mean Logits per class."""
    stats = head_stats["per_class_head_stats"]
    cats = list(stats.keys())

    biases = [stats[c]["bias_value"] for c in cats]
    norms = [stats[c]["weight_l2_norm"] for c in cats]
    mean_logits = [stats[c]["mean_logit"] for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width, biases, width, label="Bias Value (b_c)", color="#DC2626", alpha=0.85)
    rects2 = ax1.bar(x, norms, width, label="Weight L2 Norm (||W_c||)", color="#2563EB", alpha=0.85)
    rects3 = ax1.bar(x + width, mean_logits, width, label="Mean Logit on Val Set", color="#16A34A", alpha=0.85)

    ax1.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax1.set_title("Classifier Output Head Diagnostic: Weights, Biases & Logit Disparity", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Annotate values
    for rect in list(rects1) + list(rects2) + list(rects3):
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax1.annotate(f"{h:.2f}",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -10),
                     textcoords="offset points",
                     ha="center", va=va, fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved classifier head analysis plot to: {output_path}")


def run_root_cause_analysis(
    baseline_checkpoint_path: Path = MODELS_DIR / "replay_ewc_final.pt",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Executes full validation diagnostic suite and classifies the root cause into one of 4 scenarios:
    1. 'classifier_head_bias': Centroid acc > Classifier acc + threshold AND NormalizedLogitGap > threshold.
    2. 'embedding_overlap': Centroid acc < embedding_threshold.
    3. 'mixed_failure': Both conditions met.
    4. 'no_clear_bias_detected': Neither condition met.
    """
    logger.info("================ STARTING VALIDATION ROOT CAUSE DIAGNOSTICS ================")
    DIAGNOSTICS_VAL_DIR.mkdir(parents=True, exist_ok=True)

    val_data = load_embeddings("val")
    val_embs = val_data["embeddings"]
    val_lbls = val_data["labels"]

    # 1. Embedding Diagnostics
    emb_stats = compute_embedding_diagnostics(val_embs, val_lbls)
    plot_embedding_space(val_embs, val_lbls, RESULTS_PLOTS_DIR / "embedding_space_visualization.png")

    # 2. Classifier Head Diagnostics
    if not baseline_checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint {baseline_checkpoint_path} not found for diagnostics!")

    model = EvoMLP(num_classes=4).to(device)
    model.load_state_dict(torch.load(baseline_checkpoint_path, map_location=device))
    model.eval()

    head_stats = compute_classifier_head_diagnostics(model, val_embs, val_lbls, device=device)
    plot_classifier_head_analysis(head_stats, RESULTS_PLOTS_DIR / "classifier_head_analysis.png")

    # 3. Classify Root Cause
    centroid_acc = emb_stats["nearest_centroid_accuracy"]
    classifier_acc = head_stats["classifier_overall_accuracy"]
    norm_logit_gap = head_stats["normalized_logit_gap"]
    classifier_centroid_gap = centroid_acc - classifier_acc

    # Configurable Programmatic Thresholds
    thresholds = {
        "classifier_centroid_gap": 0.10,
        "normalized_logit_gap": 0.25,
        "embedding_threshold": 0.60
    }

    is_head_bias = (classifier_centroid_gap >= thresholds["classifier_centroid_gap"]) and (norm_logit_gap >= thresholds["normalized_logit_gap"])
    is_embedding_overlap = (centroid_acc < thresholds["embedding_threshold"])

    if is_head_bias and is_embedding_overlap:
        diagnosis = "mixed_failure"
        recommendation = "Both embedding overlap and classifier bias present. Multi-faceted mitigation warranted."
    elif is_head_bias:
        diagnosis = "classifier_head_bias"
        recommendation = "Embeddings are separable, but MLP logits heavily favor the newest class. Decision-boundary rebalancing (EvoRoute-BR) is strongly justified."
    elif is_embedding_overlap:
        diagnosis = "embedding_overlap"
        recommendation = "Embeddings themselves are weakly separated. Classifier rebalancing alone cannot fix underlying representation weakness."
    else:
        diagnosis = "no_clear_bias_detected"
        recommendation = "No substantial logit disparity or embedding gap observed. Interventions may not be justified."

    root_cause = {
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "metrics": {
            "nearest_centroid_accuracy": centroid_acc,
            "classifier_accuracy": classifier_acc,
            "classifier_centroid_gap": classifier_centroid_gap,
            "raw_logit_gap": head_stats["raw_logit_gap"],
            "normalized_logit_gap": norm_logit_gap,
            "std_all_logits": head_stats["std_all_logits"]
        },
        "thresholds": thresholds,
        "timestamp": None
    }

    out_path = DIAGNOSTICS_VAL_DIR / "root_cause_analysis.json"
    with open(out_path, "w") as f:
        json.dump(root_cause, f, indent=4)

    logger.info(f"Root cause classification: '{diagnosis.upper()}'")
    logger.info(f"Diagnosis details saved to: {out_path}")
    logger.info("============================================================================")
    return root_cause


if __name__ == "__main__":
    run_root_cause_analysis()
