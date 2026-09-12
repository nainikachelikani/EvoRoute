import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
import torch
import numpy as np

from src.config import (
    FINAL_METRICS_PATH,
    MEMORY_STUDY_PATH,
    NOVELTY_METRICS_PATH,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
    CLEANED_CSV_PATH,
    TRAIN_EMB_PATH,
    VAL_EMB_PATH,
    TEST_EMB_PATH,
    EWC_LAMBDA,
    LWF_TEMPERATURE,
    LWF_LAMBDA
)
from src.data import clean_dataset, split_data
from src.embeddings import generate_and_save_embeddings, load_embeddings
from src.train import train_continual_method, train_joint_upper_bound
from src.novelty import NoveltyDetector
from src.tasks import get_task_data
from src.visualize import generate_all_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def stage_preprocess():
    """Stage 1: Preprocess raw e-commerce dataset."""
    logger.info(">>> Executing Stage: PREPROCESS <<<")
    df_clean = clean_dataset()
    train_df, val_df, test_df = split_data(df_clean)
    logger.info(f"Preprocessing completed. Cleaned data at: {CLEANED_CSV_PATH}")


def stage_embeddings(force: bool = False):
    """Stage 2: Generate text embeddings using frozen MiniLM-L6-v2."""
    logger.info(">>> Executing Stage: EMBEDDINGS <<<")
    generate_and_save_embeddings(force_recompute=force)
    logger.info("Embeddings generation completed.")


def run_novelty_experiments() -> dict:
    """
    Evaluates Novelty Detection at each continual expansion milestone:
    - Milestone 1: Platform knows Books + Clothing. Incoming stream includes Electronics.
    - Milestone 2: Platform knows Books + Clothing + Electronics. Incoming stream includes Household.
    """
    logger.info("Evaluating Novelty Detection at expansion milestones...")
    val_data = load_embeddings("val")
    test_data = load_embeddings("test")

    detector = NoveltyDetector()
    novelty_results = {}

    # Milestone 1: Known = [0, 1] (Books, Clothing). Novel = [2] (Electronics)
    t1_train_embs, t1_train_lbls, _ = get_task_data(task_id=1, split="train", cumulative=True)
    detector.update_known_classes(
        new_classes=[0, 1],
        embeddings=t1_train_embs,
        labels=t1_train_lbls,
        val_embeddings=val_data["embeddings"],
        val_labels=val_data["labels"]
    )

    # Test stream with Books, Clothing vs Electronics
    m1_mask = torch.isin(test_data["labels"], torch.tensor([0, 1, 2]))
    m1_eval = detector.evaluate_novelty(
        embeddings=test_data["embeddings"][m1_mask],
        labels=test_data["labels"][m1_mask]
    )
    novelty_results["milestone_1_electronics"] = m1_eval
    logger.info(f"Milestone 1 (Electronics Novelty) -> F1: {m1_eval['f1']:.4f}, Precision: {m1_eval['precision']:.4f}, Recall: {m1_eval['recall']:.4f}")

    # Extract distances for plotting
    known_mask = torch.isin(test_data["labels"], torch.tensor([0, 1]))
    novel_mask = (test_data["labels"] == 2)
    known_dists, _ = detector.compute_distances(test_data["embeddings"][known_mask])
    novel_dists, _ = detector.compute_distances(test_data["embeddings"][novel_mask])

    plot_data = {
        "known_distances": known_dists.numpy(),
        "unknown_distances": novel_dists.numpy(),
        "threshold": detector.threshold
    }

    # Milestone 2: After learning Electronics, Household arrives
    t2_train_embs, t2_train_lbls, _ = get_task_data(task_id=2, split="train", cumulative=True)
    detector.update_known_classes(
        new_classes=[2],
        embeddings=t2_train_embs,
        labels=t2_train_lbls,
        val_embeddings=val_data["embeddings"],
        val_labels=val_data["labels"]
    )

    m2_eval = detector.evaluate_novelty(
        embeddings=test_data["embeddings"],
        labels=test_data["labels"]
    )
    novelty_results["milestone_2_household"] = m2_eval
    logger.info(f"Milestone 2 (Household Novelty) -> F1: {m2_eval['f1']:.4f}, Precision: {m2_eval['precision']:.4f}, Recall: {m2_eval['recall']:.4f}")

    with open(NOVELTY_METRICS_PATH, "w") as f:
        json.dump(novelty_results, f, indent=4)

    return plot_data


def print_official_benchmark_table(all_results: dict):
    """Prints the official Class-Incremental Continual Learning benchmark table."""
    logger.info("\n" + "="*90)
    logger.info("                         EVOROUTE OFFICIAL BENCHMARK RESULTS")
    logger.info("="*90)
    header = f"{'Method':<22} | {'Overall Accuracy':<18} | {'Final Avg Task Acc':<20} | {'Avg Forgetting':<16} | {'Memory':<12}"
    logger.info(header)
    logger.info("-" * 90)
    method_order = [
        ("naive", "Naive Sequential", "0"),
        ("ewc", "EWC", "0"),
        ("replay", "Experience Replay", "200"),
        ("lwf", "LwF ⭐", "0"),
        ("replay_ewc", "Replay + EWC ⭐", "200"),
        ("joint", "Joint Upper Bound", "Full Dataset"),
    ]
    for key, name, mem in method_order:
        if key in all_results:
            r = all_results[key]
            overall = f"{r.get('overall_accuracy', 0.0) * 100:.2f}%"
            task_acc = f"{r.get('final_avg_task_accuracy', 0.0) * 100:.2f}%"
            forget = f"{r.get('average_forgetting', 0.0) * 100:.2f}%"
            logger.info(f"{name:<22} | {overall:>18} | {task_acc:>20} | {forget:>16} | {mem:>12}")
    logger.info("="*90 + "\n")


def stage_train_and_evaluate(
    ewc_lambda: float = EWC_LAMBDA,
    lwf_temperature: float = LWF_TEMPERATURE,
    lwf_lambda: float = LWF_LAMBDA
):
    """Stage 3 & 4: Train models across all methods, evaluate, and save metrics."""
    logger.info(">>> Executing Stage: TRAIN & EVALUATE <<<")
    
    methods_to_run = ["naive", "ewc", "replay", "lwf", "replay_ewc"]
    all_results = {}

    for method in methods_to_run:
        logger.info(f"\nRunning continual training experiment: {method}")
        res = train_continual_method(
            method=method,
            ewc_lambda=ewc_lambda,
            lwf_temperature=lwf_temperature,
            lwf_lambda=lwf_lambda
        )
        all_results[method] = res

    # Run Joint Upper Bound
    logger.info("\nRunning Joint Training Upper Bound...")
    joint_res = train_joint_upper_bound()
    all_results["joint"] = joint_res

    # Memory sensitivity experiment for Replay: [0, 50, 100, 200]
    logger.info("\nRunning Memory Sensitivity Study on Replay (0, 50, 100, 200 exemplars)...")
    memory_study = {}
    for budget in [0, 50, 100, 200]:
        if budget == 0:
            memory_study["0"] = {
                "final_accuracy": all_results["naive"]["overall_accuracy"],
                "forgetting": all_results["naive"]["average_forgetting"]
            }
        elif budget == 200:
            memory_study["200"] = {
                "final_accuracy": all_results["replay"]["overall_accuracy"],
                "forgetting": all_results["replay"]["average_forgetting"]
            }
        else:
            budget_res = train_continual_method(method="replay", memory_budget=budget)
            memory_study[str(budget)] = {
                "final_accuracy": budget_res["overall_accuracy"],
                "forgetting": budget_res["average_forgetting"]
            }

    # Save metrics JSON
    with open(FINAL_METRICS_PATH, "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved final results to: {FINAL_METRICS_PATH}")

    with open(MEMORY_STUDY_PATH, "w") as f:
        json.dump(memory_study, f, indent=4)
    logger.info(f"Saved memory study to: {MEMORY_STUDY_PATH}")

    # Print official benchmark table to terminal
    print_official_benchmark_table(all_results)

    # Run novelty experiments
    novelty_plot_data = run_novelty_experiments()

    return all_results, memory_study, novelty_plot_data


def stage_visualize(all_results=None, memory_study=None, novelty_plot_data=None):
    """Stage 5: Generate publication-ready visualizations."""
    logger.info(">>> Executing Stage: VISUALIZE <<<")
    if all_results is None:
        if not FINAL_METRICS_PATH.exists():
            logger.error("Final results not found. Run train stage first.")
            return
        with open(FINAL_METRICS_PATH, "r") as f:
            all_results = json.load(f)

    if memory_study is None and MEMORY_STUDY_PATH.exists():
        with open(MEMORY_STUDY_PATH, "r") as f:
            memory_study = json.load(f)

    if novelty_plot_data is None:
        novelty_plot_data = run_novelty_experiments()

    generate_all_plots(all_results, memory_study, novelty_plot_data)
    logger.info(f"All visualizations saved in: {RESULTS_PLOTS_DIR}")


def stage_demo():
    """Stage 6: Launch Streamlit application."""
    logger.info(">>> Launching Streamlit Demo Application <<<")
    app_path = Path(__file__).resolve().parent / "app" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="EvoRoute: Adaptive E-Commerce Intelligence (Detect -> Learn -> Retain)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "preprocess", "embeddings", "train", "evaluate", "visualize", "demo"],
        help="Pipeline stage to execute."
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=EWC_LAMBDA,
        help="EWC regularization strength lambda (default: 100.0)"
    )
    parser.add_argument(
        "--lwf-temperature",
        type=float,
        default=LWF_TEMPERATURE,
        help="LwF distillation temperature T (default: 2.0)"
    )
    parser.add_argument(
        "--lwf-lambda",
        type=float,
        default=LWF_LAMBDA,
        help="LwF distillation loss weighting coefficient (default: 1.0)"
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force recomputation of text embeddings."
    )

    args = parser.parse_args()

    if args.stage == "preprocess":
        stage_preprocess()
    elif args.stage == "embeddings":
        stage_embeddings(force=args.force_embeddings)
    elif args.stage in ["train", "evaluate"]:
        stage_train_and_evaluate(
            ewc_lambda=args.ewc_lambda,
            lwf_temperature=args.lwf_temperature,
            lwf_lambda=args.lwf_lambda
        )
    elif args.stage == "visualize":
        stage_visualize()
    elif args.stage == "demo":
        stage_demo()
    elif args.stage == "all":
        stage_preprocess()
        stage_embeddings(force=args.force_embeddings)
        results, mem_study, nov_plot_data = stage_train_and_evaluate(
            ewc_lambda=args.ewc_lambda,
            lwf_temperature=args.lwf_temperature,
            lwf_lambda=args.lwf_lambda
        )
        stage_visualize(results, mem_study, nov_plot_data)
        logger.info("\n========================================================")
        logger.info("   EvoRoute Research Pipeline Completed Successfully!   ")
        logger.info("   Run: 'python main.py --stage demo' to launch UI.     ")
        logger.info("========================================================")


if __name__ == "__main__":
    main()
