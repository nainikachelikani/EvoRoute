import os
from pathlib import Path
import torch

# Random Seed for absolute reproducibility
RANDOM_SEED = 42

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"

# File paths
RAW_CSV_PATH = DATA_RAW_DIR / "ecommerceDataset.csv"
CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "cleaned_data.csv"
TRAIN_EMB_PATH = DATA_PROCESSED_DIR / "train_embeddings.pt"
VAL_EMB_PATH = DATA_PROCESSED_DIR / "val_embeddings.pt"
TEST_EMB_PATH = DATA_PROCESSED_DIR / "test_embeddings.pt"

FINAL_METRICS_PATH = RESULTS_METRICS_DIR / "final_results.json"
MEMORY_STUDY_PATH = RESULTS_METRICS_DIR / "memory_sensitivity.json"
NOVELTY_METRICS_PATH = RESULTS_METRICS_DIR / "novelty_metrics.json"

# Manifest and Artifact Paths
BASELINE_MANIFEST_PATH = RESULTS_METRICS_DIR / "baseline_manifest.json"
CONFIG_MANIFEST_PATH = RESULTS_METRICS_DIR / "config_manifest.json"
OFFICIAL_BENCHMARK_MANIFEST_PATH = RESULTS_METRICS_DIR / "official_benchmark_manifest.json"
PREDICTION_TRANSITION_MATRIX_PATH = RESULTS_METRICS_DIR / "prediction_transition_matrix.json"
SANITY_CHECK_RESULTS_PATH = RESULTS_METRICS_DIR / "sanity_check_results.json"
SCIENTIFIC_SUMMARY_PATH = RESULTS_METRICS_DIR / "scientific_summary.json"
PLOT_MANIFEST_PATH = RESULTS_METRICS_DIR / "plot_manifest.json"

# Checkpoint Paths
BASELINE_CHECKPOINT_PATH = MODELS_DIR / "replay_ewc_final.pt"
EVOROUTE_BR_CHECKPOINT_PATH = MODELS_DIR / "evoroute_br_candidate_final.pt"
EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH = MODELS_DIR / "evoroute_br_calibrated_final.pt"

# Ensure output directories exist
for directory in [DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_METRICS_DIR, RESULTS_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Categories & IDs
CATEGORIES = [
    "Books",
    "Clothing & Accessories",
    "Electronics",
    "Household"
]

CATEGORY2ID = {
    "Books": 0,
    "Clothing & Accessories": 1,
    "Electronics": 2,
    "Household": 3
}

ID2CATEGORY = {v: k for k, v in CATEGORY2ID.items()}

# Continual Learning Tasks
TASK_DEFINITIONS = {
    1: ["Books", "Clothing & Accessories"],
    2: ["Electronics"],
    3: ["Household"]
}

TASK_CLASSES = {
    1: [CATEGORY2ID[c] for c in TASK_DEFINITIONS[1]],
    2: [CATEGORY2ID[c] for c in TASK_DEFINITIONS[2]],
    3: [CATEGORY2ID[c] for c in TASK_DEFINITIONS[3]]
}

# Cumulative classes after each task
TASK_CUMULATIVE_CLASSES = {
    1: [0, 1],
    2: [0, 1, 2],
    3: [0, 1, 2, 3]
}

# Data Subsampling for Fast Hackathon Runs
# None to use all rows; 2000 per class gives ~8000 total clean balanced rows
MAX_SAMPLES_PER_CLASS = 2000
MIN_TEXT_LENGTH = 20

# Splits
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64

# Classifier Architecture
HIDDEN_DIMS = [256, 128]
DROPOUT_RATE = 0.2

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_PER_TASK = 10

# Experience Replay
REPLAY_MEMORY_BUDGET = 200  # Max total exemplars across all seen classes
REPLAY_SAMPLE_RATIO = 0.2    # 80% new task data, 20% replay buffer

# Elastic Weight Consolidation (EWC)
EWC_LAMBDA = 100.0           # Configurable EWC quadratic penalty strength

# Learning without Forgetting (LwF)
LWF_TEMPERATURE = 2.0        # Temperature for softening probability distributions
LWF_LAMBDA = 1.0             # Weighting coefficient for knowledge distillation loss

# Novelty Detection Calibration
NOVELTY_CALIBRATION_PERCENTILE = 95.0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scientific Baseline Configuration (Immutable reference)
BASELINE_CONFIG = {
    "method": "replay_ewc",
    "replay_strategy": "original",
    "replay_memory_budget": 200,
    "replay_sample_ratio": 0.2,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs_per_task": 10,
    "ewc_lambda": 100.0,
    "seed": 42
}

# EvoRoute-BR Experimental Candidate Configuration
EVOROUTE_BR_CONFIG = {
    "candidate_name": "evoroute_br_candidate",
    "replay_strategy": "class_balanced",
    "replay_memory_budget": 200,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs_per_task": 10,
    "ewc_lambda": 100.0,
    "seed": 42,
    "finetuning_enabled": True,
    "finetuning_epochs": 10,
    "finetuning_lr": 1e-4,
    "finetuning_patience": 2,
    "metric_tolerance": 1e-4
}

# Post-hoc Calibration Configuration (Validation Split Only)
CALIBRATION_CONFIG = {
    "confidence_temperature_scaling": True,
    "decision_rebalancing": True,
    "gamma_search_range": (0.0, 5.0),
    "gamma_search_steps": 51,
    "regularization_weight": 0.01,
    "max_newest_accuracy_drop": 0.30,
    "min_newest_accuracy_absolute": 0.60
}

# Metric Tolerances & Anti-Collapse Invariants
METRIC_TOLERANCE = 1e-4
MIN_NEWEST_CLASS_ACCURACY = 0.60
MIN_OLD_CLASS_ACCURACY = 0.40
MAX_FORGETTING_INCREASE = 0.05
