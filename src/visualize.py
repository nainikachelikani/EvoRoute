import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.config import (
    RESULTS_PLOTS_DIR,
    CATEGORIES,
    CATEGORY2ID,
    ID2CATEGORY
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style configuration for clean, professional hackathon figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 7
})

METHOD_COLORS = {
    "naive": "#E63946",       # Red
    "ewc": "#F4A261",         # Orange
    "replay": "#2A9D8F",      # Teal
    "lwf": "#8338EC",         # Violet / Purple (Distillation)
    "replay_ewc": "#1D3557",  # Deep Navy Blue (Proposed)
    "joint": "#457B9D"        # Slate Blue (Upper Bound)
}

METHOD_LABELS = {
    "naive": "Naive Sequential",
    "ewc": "EWC",
    "replay": "Experience Replay (200)",
    "lwf": "LwF (Distillation)",
    "replay_ewc": "Replay + EWC (Proposed)",
    "joint": "Joint Training (Upper Bound)"
}


def plot_accuracy_across_tasks(results_data: Dict[str, Any], save_path: Path = None):
    """Plot 1: Accuracy Across Tasks comparing all methods."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    tasks = [1, 2, 3]

    for method_key, color in METHOD_COLORS.items():
        if method_key not in results_data:
            continue
        history = results_data[method_key].get("task_history", [])
        if not history:
            continue
        accs = [h["overall_accuracy"] * 100 for h in history]
        label = METHOD_LABELS.get(method_key, method_key)
        linestyle = "--" if method_key == "joint" else "-"
        ax.plot(tasks[:len(accs)], accs, marker="o", color=color, label=label, linestyle=linestyle)

    ax.set_title("EvoRoute: Overall Accuracy Across Continual Learning Tasks", fontweight="bold", pad=12)
    ax.set_xlabel("Task Sequence")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_xticks(tasks)
    ax.set_xticklabels(["Task 1\n(Books + Cloth)", "Task 2\n(+Electronics)", "Task 3\n(+Household)"])
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, loc="lower left")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "accuracy_across_tasks.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 1: {save_path}")


def plot_catastrophic_forgetting(results_data: Dict[str, Any], save_path: Path = None):
    """Plot 2: Catastrophic Forgetting Comparison across methods."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    methods = [m for m in ["naive", "ewc", "lwf", "replay", "replay_ewc"] if m in results_data]
    forgetting_vals = []
    labels = []
    colors = []

    for m in methods:
        f_val = results_data[m].get("forgetting", {}).get("average_forgetting", 0.0) * 100
        forgetting_vals.append(f_val)
        labels.append(METHOD_LABELS.get(m, m))
        colors.append(METHOD_COLORS.get(m, "#333333"))

    bars = ax.bar(labels, forgetting_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    ax.set_title("Catastrophic Forgetting Comparison (Lower is Better)", fontweight="bold", pad=12)
    ax.set_ylabel("Average Forgetting on Past Tasks (%)")
    ax.set_ylim(0, max(forgetting_vals + [10]) * 1.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "catastrophic_forgetting.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 2: {save_path}")


def plot_per_class_accuracy(results_data: Dict[str, Any], save_path: Path = None):
    """Plot 3: Per-Class Accuracy after final task."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    methods = [m for m in ["naive", "ewc", "lwf", "replay", "replay_ewc", "joint"] if m in results_data]
    categories = CATEGORIES

    x = np.arange(len(categories))
    width = 0.14

    for idx, m in enumerate(methods):
        per_class = results_data[m].get("final_per_class", {})
        accs = [per_class.get(cat, 0.0) * 100 for cat in categories]
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m, "#666666"))

    ax.set_title("Per-Class Accuracy After Task 3 (Final State)", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "per_class_accuracy.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 3: {save_path}")


def plot_memory_vs_accuracy(memory_study_data: Dict[str, Any], save_path: Path = None):
    """Plot 4A: Memory Sensitivity (0, 50, 100, 200 memory budget) vs Overall Accuracy."""
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    budgets = sorted([int(k) for k in memory_study_data.keys()])
    accuracies = []
    for b in budgets:
        val = memory_study_data[str(b)]
        acc = val["final_accuracy"] if isinstance(val, dict) else val
        accuracies.append(acc * 100)

    ax.plot(budgets, accuracies, marker="s", color="#1D3557", linewidth=2.5, markersize=8, label="Replay Accuracy")
    for b, acc in zip(budgets, accuracies):
        ax.annotate(f"{acc:.1f}%", xy=(b, acc), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontweight="bold")

    ax.set_title("Memory Ablation: Replay Buffer Size vs Final Overall Accuracy", fontweight="bold", pad=12)
    ax.set_xlabel("Replay Memory Budget (Number of Embeddings)")
    ax.set_ylabel("Final Overall Accuracy (%)")
    ax.set_xticks(budgets)
    ax.set_ylim(min(accuracies) - 10, 100)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "memory_vs_accuracy.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 4A: {save_path}")


def plot_memory_vs_forgetting(memory_study_data: Dict[str, Any], save_path: Path = None):
    """Plot 4B: Memory Sensitivity (0, 50, 100, 200 memory budget) vs Catastrophic Forgetting."""
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    budgets = sorted([int(k) for k in memory_study_data.keys()])
    forgettings = []
    for b in budgets:
        val = memory_study_data[str(b)]
        f = val.get("forgetting", 0.0) if isinstance(val, dict) else 0.0
        forgettings.append(f * 100)

    ax.plot(budgets, forgettings, marker="o", color="#E63946", linewidth=2.5, markersize=8, label="Average Forgetting")
    for b, f in zip(budgets, forgettings):
        ax.annotate(f"{f:.1f}%", xy=(b, f), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontweight="bold")

    ax.set_title("Memory Ablation: Replay Buffer Size vs Catastrophic Forgetting", fontweight="bold", pad=12)
    ax.set_xlabel("Replay Memory Budget (Number of Embeddings)")
    ax.set_ylabel("Final Average Forgetting (%) [Lower is Better]")
    ax.set_xticks(budgets)
    ax.set_ylim(0, max(forgettings) * 1.15)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "memory_vs_forgetting.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 4B: {save_path}")


def plot_novelty_distribution(
    known_distances: np.ndarray,
    unknown_distances: np.ndarray,
    threshold: float,
    save_path: Path = None
):
    """Plot 5: Known Samples vs Unknown Samples Cosine Distance Distribution with Threshold."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    sns.kdeplot(known_distances, ax=ax, label="Known Category Samples", color="#2A9D8F", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(unknown_distances, ax=ax, label="Novel / Unfamiliar Samples", color="#E63946", fill=True, alpha=0.35, linewidth=2)

    ax.axvline(threshold, color="#1D3557", linestyle="--", linewidth=2.2, label=f"Calibrated Threshold ($\\tau$ = {threshold:.3f})")

    ax.set_title("Novelty Detection: Semantic Distance Distribution to Known Centroids", fontweight="bold", pad=12)
    ax.set_xlabel("Cosine Distance to Nearest Known Class Centroid")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "novelty_detection_distribution.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 5: {save_path}")


def plot_knowledge_retention_heatmap(results_data: Dict[str, Any], save_path: Path = None):
    """Plot 6: Knowledge Retention Heatmap (Naive vs Replay + EWC) across Tasks 1, 2, and 3."""
    if "naive" not in results_data or "replay_ewc" not in results_data:
        logger.warning("Cannot plot heatmap without naive and replay_ewc results.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    cats = CATEGORIES
    tasks = ["After Task 1", "After Task 2", "After Task 3"]

    def extract_matrix(method_key):
        history = results_data[method_key].get("task_history", [])
        mat = np.zeros((3, 4))
        for t_idx, h in enumerate(history):
            per_class = h.get("per_class_accuracy", {})
            for c_idx, c_name in enumerate(cats):
                if c_name in per_class:
                    mat[t_idx, c_idx] = per_class[c_name] * 100
                else:
                    mat[t_idx, c_idx] = np.nan
        return mat

    naive_mat = extract_matrix("naive")
    prop_mat = extract_matrix("replay_ewc")

    sns.heatmap(naive_mat, annot=True, fmt=".1f", cmap="Reds_r", vmin=0, vmax=100,
                xticklabels=cats, yticklabels=tasks, ax=ax1, cbar=False, linewidths=0.5)
    ax1.set_title("Naive Sequential Learning\n(Catastrophic Forgetting)", fontweight="bold")
    ax1.set_xticklabels(cats, rotation=25, ha="right")

    sns.heatmap(prop_mat, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100,
                xticklabels=cats, yticklabels=tasks, ax=ax2, cbar=True, linewidths=0.5)
    ax2.set_title("Replay + EWC (Proposed)\n(Knowledge Retained)", fontweight="bold")
    ax2.set_xticklabels(cats, rotation=25, ha="right")

    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "knowledge_retention_heatmap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 6: {save_path}")


def plot_lwf_retention_analysis(results_data: Dict[str, Any], save_path: Path = None):
    """
    Plot 7: LwF Retention Analysis comparing Naive vs LwF vs Replay + EWC across Tasks 1, 2, and 3.
    Demonstrates the retention characteristics of knowledge distillation against baseline & replay.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    stages = ["Task 1\n(Books + Cloth)", "Task 2\n(+Electronics)", "Task 3\n(+Household)"]
    x = np.arange(len(stages))

    methods_to_compare = [
        ("naive", "Naive Sequential", "#E63946", "o-"),
        ("lwf", "LwF (Distillation)", "#8338EC", "s-"),
        ("replay_ewc", "Replay + EWC (Proposed)", "#1D3557", "^-")
    ]

    for m_key, m_label, color, fmt in methods_to_compare:
        if m_key in results_data:
            hist = results_data[m_key].get("task_history", [])
            accs = [h["overall_accuracy"] * 100 for h in hist]
            if len(accs) == len(stages):
                ax.plot(x, accs, fmt, color=color, linewidth=2.5, markersize=8, label=m_label)
                for i, txt in enumerate(accs):
                    offset = 8 if m_key != "lwf" else -14
                    ax.annotate(f"{txt:.1f}%", (x[i], txt), textcoords="offset points", xytext=(0, offset),
                                ha='center', fontweight="bold", fontsize=9, color=color)

    ax.set_title("LwF Retention Analysis: Distillation vs Naive vs Replay + EWC", fontweight="bold", pad=12)
    ax.set_xlabel("Continual Learning Stage")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_PLOTS_DIR / "lwf_retention_analysis.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Plot 7: {save_path}")


def generate_all_plots(results_data: Dict[str, Any], memory_data: Dict[str, float] = None, novelty_data: Dict[str, Any] = None):
    """Generates and saves all required plots."""
    plot_accuracy_across_tasks(results_data)
    plot_catastrophic_forgetting(results_data)
    plot_per_class_accuracy(results_data)
    if memory_data is not None:
        plot_memory_vs_accuracy(memory_data)
        plot_memory_vs_forgetting(memory_data)
    plot_knowledge_retention_heatmap(results_data)
    plot_lwf_retention_analysis(results_data)
    if novelty_data is not None and "known_distances" in novelty_data:
        plot_novelty_distribution(
            novelty_data["known_distances"],
            novelty_data["unknown_distances"],
            novelty_data["threshold"]
        )

