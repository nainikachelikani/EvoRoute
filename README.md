# EvoRoute — Adaptive E-Commerce Intelligence
### Detect → Learn → Retain | Class-Incremental Continual Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blue.svg)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Track](https://img.shields.io/badge/Track-Continual%20Learning-purple.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Research Question & Problem Statement

**How can an e-commerce catalog routing model incrementally assimilate newly arriving product categories without suffering catastrophic forgetting on historical categories, in the strict absence of task identity at inference time?**

Modern e-commerce platforms continuously onboard new merchants and product verticals. A production routing system deployed initially on core categories (**Books**, **Clothing & Accessories**) inevitably encounters emergent product streams such as **Electronics** and **Household**.

Standard deep neural networks trained sequentially suffer from **Catastrophic Forgetting**: fine-tuning on a new category overwrites the representations and classification boundaries established for previous categories, driving historical classification accuracy toward zero.

**EvoRoute** addresses this challenge through an end-to-end, modular lifecycle:

$$\textbf{DETECT} \longrightarrow \textbf{LEARN} \longrightarrow \textbf{RETAIN}$$

```
   ┌────────────────────────────────────────────────────────┐
   │                  INCOMING PRODUCT STREAM               │
   │      "Sony WH-1000XM5 Wireless Noise-Cancelling"       │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │            FROZEN MiniLM-L6-v2 ENCODER                 │
   │               (384-D Unit Embeddings)                  │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               NOVELTY DETECTOR (DETECT)                │
   │       Cosine Distance to Known Class Centroids         │
   │       Threshold Test: d_min(e) > tau ?                 │
   └───────────┬────────────────────────────────┬───────────┘
               │ NO                             │ YES
               ▼                                ▼
   ┌───────────────────────┐        ┌───────────────────────┐
   │     KNOWN CATEGORY    │        │ 🚨 NOVEL CATEGORY     │
   │ Route to current head │        │ Dynamic Head Expansion│
   └───────────────────────┘        └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  ADAPTIVE CLASSIFIER  │
                                    │        (LEARN)        │
                                    │ Dynamic Weight Copy   │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  MEMORY & WEIGHT REG  │
                                    │       (RETAIN)        │
                                    │ • Replay (200 budget) │
                                    │ • EWC Fisher Penalty  │
                                    └───────────────────────┘
```

---

## 2. Experimental Setting: Class-Incremental Learning (CIL)

EvoRoute operates under the **Class-Incremental Learning (CIL)** paradigm:
- **No Task Oracle at Inference:** At test time, the model is presented with an item from any seen category and must classify across all known categories without knowing which task it originated from.
- **Dynamic Output Head Expansion:** When a new category arrives, the classification layer dynamically expands from $C \to C+1$ while strictly preserving previous weight values.
- **Strictly Isolated Test Set:** The test set is split at the outset (80% Train, 10% Val, 10% Test, stratified, `random_state=42`) and is never accessed during training, replay sampling, or Fisher Information estimation.

### Continual Learning Tasks
1. **Task 1 (Base Platform):** Categories `Books` (0) and `Clothing & Accessories` (1).
2. **Task 2 (Tech Expansion):** Category `Electronics` (2) arrives.
3. **Task 3 (Home Expansion):** Category `Household` (3) arrives.

---

## 3. Evaluated Methods

1. **Naive Sequential Fine-Tuning:** Standard empirical risk minimization on new tasks with no retention mechanisms.
2. **Elastic Weight Consolidation (EWC):** Prior-task parameter regularization via quadratic penalty on the diagonal of the empirical Fisher Information Matrix.
3. **Experience Replay:** Memory-budgeted exemplar buffer (budget $\le 200$ embeddings) using class-balanced centroid-distance exemplar selection and dynamic rebalancing.
4. **Replay + EWC (Proposed):** Synergistic integration combining balanced exemplar replay to anchor decision boundaries and EWC to regularize parameter drift.
5. **Joint Upper Bound:** Non-continual upper-bound baseline trained jointly on all 4 categories simultaneously.

---

## 4. Primary Evaluation Metrics & Mathematical Definitions

### A. Overall Class Accuracy
Accuracy evaluated across all final test samples across all classes seen up to task $T$:
$$\text{Overall Class Accuracy} = \frac{\sum_{i=1}^{N_{test}} \mathbb{I}(\hat{y}_i = y_i)}{N_{test}}$$

### B. Final Average Task Accuracy
The unweighted arithmetic mean of test accuracies evaluated on each individual task after completing the final task $T$:
$$\text{Final Avg Task Accuracy} = \frac{1}{T} \sum_{k=1}^T R(T, k)$$
where $R(T, k)$ represents test accuracy on Task $k$ after learning Task $T$.

### C. Final Average Forgetting
The average degradation in performance across all previously learned tasks relative to their peak historical performance:
$$F = \frac{1}{T-1} \sum_{k=1}^{T-1} f_k^T \quad \text{where} \quad f_k^T = \max_{l \in \{1, \dots, T-1\}} R(l, k) - R(T, k)$$

---

## 5. Official Benchmark Results

All metrics below are generated directly from actual execution (`results/metrics/final_results.json`):

| Method | Overall Accuracy | Final Avg Task Accuracy | Avg Forgetting | Memory |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Sequential** | 25.00% | 33.33% | 99.50% | 0 |
| **EWC** | 25.00% | 33.33% | 99.50% | 0 |
| **Experience Replay** | 64.62% | 67.17% | 47.75% | 200 |
| **Replay + EWC ⭐** | **65.62%** | **68.00%** | **46.50%** | 200 |
| **Joint Upper Bound** | 93.75% | 92.50% | 0.00% | Full Dataset |

### Key Empirical Findings & EWC Verification:
1. **Severe Catastrophic Forgetting in Naive Baseline:** Naive fine-tuning collapses to predicting only the most recently trained category (Household), suffering 99.50% forgetting.
2. **EWC in Class-Incremental Learning:** 
   - A rigorous code inspection identified and resolved a batch-averaging gradient squaring bug in Fisher computation, replacing it with exact sample-wise gradient accumulation $\frac{1}{N} \sum (\nabla_\theta \log p_i)^2$. Parameter snapshots and loss additions were verified.
   - Despite mathematically correct Fisher penalization, standalone EWC under Class-Incremental Learning experiences logit drift: newly expanded output heads lack prior Fisher penalties, while incoming batches contain only the new class. Without replay exemplars to anchor multi-class boundaries, the unconstrained new output head dominates predictions.
3. **Synergy of Replay + EWC:** 
   - Experience Replay provides the foundational anchor for multi-class decision boundaries, raising accuracy to 64.62%.
   - Adding EWC regularizes parameter trajectories against drifting, yielding the highest overall accuracy (**65.62%**), highest task accuracy (**68.00%**), and lowest catastrophic forgetting (**46.50%**).

---

## 6. Memory Sensitivity Ablation Study

Evaluated on Experience Replay across memory budgets of 0, 50, 100, and 200 exemplars (`results/metrics/memory_sensitivity.json`):

| Replay Budget | Overall Accuracy | Average Forgetting | Status |
| :---: | :---: | :---: | :--- |
| **0 exemplars** | 25.00% | 99.50% | Total catastrophic forgetting (equivalent to Naive) |
| **50 exemplars** | 51.00% | 65.75% | Substantial retention with only ~12 exemplars/class |
| **100 exemplars** | 58.25% | 56.62% | Diminishing marginal returns on memory growth |
| **200 exemplars** | 64.62% | 47.75% | Robust multi-class retention at compact memory footprint |

---

## 7. Supporting Novelty Detection Mechanism

Novelty detection serves as a secondary, lightweight gatekeeper to flag unfamiliar items before triggering output expansion:
- **Milestone 1 (Electronics arrival):** F1 = 0.7966, Precision = 0.9329, Recall = 0.6950 (calibrated $\tau = 0.8111$).
- **Milestone 2 (Household arrival):** F1 = 0.4057, Precision = 0.7037, Recall = 0.2850 (calibrated $\tau = 0.7829$).

*Note on Novelty Detection:* Semantic overlap between categories (e.g. household electronic appliances) presents natural boundary challenges. Novelty detection is treated as a supporting heuristic, while continual retention remains the core contribution.

---

## 8. Directory Structure

```
EvoRoute/
├── data/
│   ├── raw/ecommerceDataset.csv          # Raw e-commerce product catalog
│   └── processed/                         # Cleaned data and cached 384-D MiniLM embeddings
├── src/
│   ├── config.py                         # Hyperparameters, paths, class mappings
│   ├── data.py                           # Auto-detection, stratified splitting, cleaning
│   ├── embeddings.py                     # Frozen all-MiniLM-L6-v2 embedding generation
│   ├── tasks.py                          # Task partition loaders and dataset iterators
│   ├── model.py                          # EvoMLP with dynamic expansion and weight invariance
│   ├── novelty.py                        # Hyperspherical centroid detector & threshold calibration
│   ├── replay.py                         # ReplayBuffer (budget <= 200) with balanced exemplar sampling
│   ├── ewc.py                            # Exact sample-wise Fisher calculation & quadratic penalty
│   ├── train.py                          # Training loops for all 5 methods & upper bound
│   ├── evaluate.py                       # Overall accuracy, task accuracy, and forgetting metrics
│   └── visualize.py                      # 7 publication-ready plots (accuracy, forgetting, heatmaps)
├── models/                               # Checkpoints (.pt) for all trained methods
├── results/
│   ├── metrics/                          # final_results.json, memory_sensitivity.json, novelty_metrics.json
│   └── plots/                            # High-resolution PNG figures
├── app/
│   └── app.py                            # 5-page interactive Streamlit dashboard
├── tests/
│   ├── test_model_expansion.py           # Unit tests for weight invariance during expansion
│   ├── test_replay_buffer.py             # Unit tests for buffer budget enforcement & rebalancing
│   ├── test_ewc.py                       # Unit tests for Fisher accumulation & loss integration
│   └── test_streamlit_runtime.py         # End-to-end dashboard & model runtime validation
├── main.py                               # Unified CLI runner
├── requirements.txt
└── README.md
```

---

## 9. Reproducibility & Quick Start

### 1. Installation
```bash
git clone https://github.com/your-username/EvoRoute.git
cd EvoRoute
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
```bash
# Runs full pipeline: preprocess -> embeddings -> train -> evaluate -> visualize
python main.py --stage all
```

### 3. Modular Stage Execution
```bash
# Run training and evaluation across all 5 methods
python main.py --stage train

# Generate all 7 publication figures
python main.py --stage visualize

# Launch Streamlit interactive dashboard
python main.py --stage demo
```

### 4. Run Complete Verification Test Suite
```bash
python tests/test_model_expansion.py
python tests/test_replay_buffer.py
python tests/test_ewc.py
python tests/test_streamlit_runtime.py
```

---

## 10. License & Citation
Developed for the Deep Learning Continual Learning Hackathon (Track 5: Continual Learning). Released under the MIT License.
