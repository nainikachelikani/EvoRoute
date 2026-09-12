# EvoRoute — Adaptive E-Commerce Intelligence
> **A continual deep learning system that detects, learns, and retains evolving e-commerce categories.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blue.svg)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Track](https://img.shields.io/badge/Track-Continual%20Learning-purple.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title
**EvoRoute: Adaptive E-Commerce Intelligence**

---

## 2. One-Line Tagline
A continual deep learning system that detects, learns, and retains evolving e-commerce categories.

---

## 3. Problem Statement
When standard neural networks are fine-tuned sequentially on newly emerging data distributions:
$$\mathcal{L}_{seq} = \mathbb{E}_{(x, y) \sim \mathcal{D}_{new}} [\ell(f_\theta(x), y)]$$
they suffer from **Catastrophic Forgetting**. The gradient updates overwrite the weight trajectories and representation subspaces critical to earlier categories, causing classification accuracy on historical classes to plummet toward zero.

---

## 4. Real-World E-Commerce Scenario
An enterprise e-commerce marketplace rarely launches with all product verticals simultaneously. Catalog taxonomy evolves dynamically:
1. **Initial Launch:** The platform routes core retail categories: **Books** and **Clothing & Accessories**.
2. **Expansion Wave 1:** High-volume **Electronics** products arrive.
3. **Expansion Wave 2:** **Household** goods are onboarded.

Traditional architectures require costly, full-dataset retraining from scratch whenever a new vertical is added. EvoRoute allows the system to incrementally assimilate new product categories without service downtime and without forgetting historical catalog knowledge.

---

## 5. Key Idea: Detect → Learn → Retain
EvoRoute implements a closed-loop tripartite continual learning lifecycle:

$$\textbf{DETECT} \longrightarrow \textbf{LEARN} \longrightarrow \textbf{RETAIN}$$

1. **DETECT:** Hyperspherical semantic novelty detection identifies unfamiliar product categories before ingestion.
2. **LEARN:** Dynamic output layer expansion ($C \to C+1$) preserves previously learned connection weights.
3. **RETAIN:** Bounded exemplar replay (200 memory budget) anchors decision boundaries, while Elastic Weight Consolidation (EWC) penalizes critical parameter drift.

---

## 6. System Architecture

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

## 7. Class-Incremental Learning (CIL) Setting
EvoRoute operates under strict **Class-Incremental Learning (CIL)**:
- **Zero Task Oracle at Inference:** The model receives no task identifier $t$ at test time. Given an unlabelled product vector, it must choose among *all* classes learned up to the present task.
- **Strict Data Isolation:** Stratified splitting (80% Train, 10% Validation, 10% Test) is performed at the outset with `random_state=42`. Test split samples are strictly held out and are never exposed during training, replay memory buffering, or Fisher Information estimation.

---

## 8. Continual Learning Tasks
The categories arrive sequentially in three distinct tasks:
- **Task 1 (Base Platform):** Categories `Books` (0) and `Clothing & Accessories` (1).
- **Task 2 (Tech Expansion):** Category `Electronics` (2) arrives.
- **Task 3 (Home Expansion):** Category `Household` (3) arrives.

---

## 9. Model Architecture
- **Semantic Text Encoder:** Pretrained `sentence-transformers/all-MiniLM-L6-v2` completely frozen. Generates 384-dimensional $L_2$-normalized unit vectors with zero representation drift.
- **Classifier (`EvoMLP`):**
  $$\text{Input (384)} \longrightarrow \text{Linear(256)} \longrightarrow \text{ReLU} \longrightarrow \text{Dropout(0.2)} \longrightarrow \text{Linear(128)} \longrightarrow \text{ReLU} \longrightarrow \text{Dropout(0.2)} \longrightarrow \text{Head}(C)$$
  where $C \in \{2, 3, 4\}$ dynamically expands across tasks.

---

## 10. Evaluated Methods
1. **Naive Sequential Fine-Tuning:** Standard backpropagation on incoming task data without memory or regularization baselines.
2. **Elastic Weight Consolidation (EWC):** Prior-task parameter protection via Fisher Information quadratic penalty.
3. **Experience Replay:** Bounded memory buffer (max 200 exemplars) with centroid-distance exemplar sampling.
4. **Replay + EWC ⭐ (Proposed):** Dual retention mechanism combining exemplar replay with Fisher regularization.
5. **Joint Training Upper Bound:** Offline upper-bound baseline trained simultaneously across all 4 categories.

---

## 11. Experience Replay
- **Strict Capacity Constraint:** Memory budget $M \le 200$ embeddings total.
- **Class-Balanced Allocation:** Memory partitions dynamically across seen classes:
  $$|\mathcal{M}_c| = \left\lfloor \frac{M}{|\mathcal{C}_{seen}|} \right\rfloor$$
  (Task 1: 100/class $\to$ Task 2: 66/class $\to$ Task 3: 50/class).
- **Centroid-Based Selection:** For each class, exemplars closest to the class mean centroid in embedding space are prioritized to preserve cluster geometry.
- **Training Mixture:** Each training step samples an 80/20 mixture of incoming task data and replay buffer exemplars.

---

## 12. Elastic Weight Consolidation (EWC)
EWC estimates the diagonal of the empirical Fisher Information Matrix $F$ using sample-wise squared gradients of log-likelihood:
$$F_j = \frac{1}{N} \sum_{i=1}^N \left( \frac{\partial \log p(y_i \mid x_i, \theta)}{\partial \theta_j} \right)^2$$
During subsequent tasks, an additional quadratic penalty discourages movement along high-curvature parameter directions:
$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{CE}(\theta) + \sum_{k=1}^{t-1} \frac{\lambda}{2} \sum_j F_{k, j} (\theta_j - \theta_{k, j}^*)^2$$
EvoRoute automatically aligns prior Fisher matrices with expanded output layers by slicing across historical row dimensions.

---

## 13. Dynamic Output Layer Expansion
When a novel category is detected, the classification head dynamically expands from $C \to C+1$:
1. A new `nn.Linear(128, C+1)` layer is instantiated.
2. Existing weight rows and bias terms for previously learned categories ($0 \le c < C$) are directly copied into the new head.
3. New class weights are initialized using Xavier normal initialization; biases are initialized to zero.
4. **Automated Assertion Verification:** Automated unit tests verify bit-for-bit invariance:
   $$\theta_{new}[:C] \equiv \theta_{old}[:C]$$

---

## 14. Novelty Detection
- **Centroid Distance:** For known classes $\mathcal{C}_{seen}$, normalized centroids $\mu_c$ are maintained. An incoming product $e$ is tested via:
  $$d_{min}(e) = \min_{c \in \mathcal{C}_{seen}} (1 - e \cdot \mu_c)$$
- **Threshold Calibration:** $\tau$ is calibrated dynamically at the 95th percentile of distances across known validation samples.
- **Milestone Performance:**
  - Milestone 1 (Electronics): F1 = 0.7966, Precision = 0.9329, Recall = 0.6950 ($\tau = 0.8111$).
  - Milestone 2 (Household): F1 = 0.4057, Precision = 0.7037, Recall = 0.2850 ($\tau = 0.7829$).

---

## 15. Primary Evaluation Metrics
1. **PRIMARY METRIC 1: Overall Final Accuracy:** Accuracy evaluated across all test samples after completing final Task 3.
2. **PRIMARY METRIC 2: Final Average Forgetting:**
   $$F = \frac{1}{T-1} \sum_{k=1}^{T-1} \left( \max_{l < T} R(l, k) - R(T, k) \right)$$
3. **Final Average Task Accuracy:**
   $$\text{Final Avg Task Accuracy} = \frac{1}{T} \sum_{k=1}^T R(T, k)$$

*Distinction:* Overall Accuracy evaluates global sample accuracy; Final Average Task Accuracy weighs each task equally; Average Forgetting isolates historical retention loss.

---

## 16. Official Benchmark Results

All metrics below are generated directly from actual execution (`results/metrics/final_results.json`):

| Method | Overall Accuracy | Final Avg Task Accuracy | Avg Forgetting | Memory |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Sequential** | 25.00% | 33.33% | 99.50% | 0 |
| **EWC** | 25.00% | 33.33% | 99.50% | 0 |
| **Experience Replay** | 64.62% | 67.17% | 47.75% | 200 |
| **Replay + EWC ⭐** | **65.62%** | **68.00%** | **46.50%** | **200** |
| **Joint Upper Bound** | 93.75% | 92.50% | 0.00% | Full Dataset |

*Clarification on Joint Upper Bound:* Joint Training achieves 93.75% because it is trained offline with all data available simultaneously. **Joint Training is NOT a continual learning method** and is included strictly as an empirical upper bound.

---

## 17. Memory Sensitivity Ablation Study

Evaluated on Experience Replay across memory budgets of 0, 50, 100, and 200 exemplars (`results/metrics/memory_sensitivity.json`):

| Replay Budget | Overall Accuracy | Average Forgetting | Observation |
| :---: | :---: | :---: | :--- |
| **0 exemplars** | 25.00% | 99.50% | Complete catastrophic forgetting (identical to Naive) |
| **50 exemplars** | 51.00% | 65.75% | Drastic retention jump with only ~12 exemplars/class |
| **100 exemplars** | 58.25% | 56.62% | Steady gains with compact memory footprint |
| **200 exemplars** | 64.62% | 47.75% | Strongest multi-class retention at 200 embeddings |

---

## 18. Key Scientific Findings
1. **Severe Catastrophic Forgetting in Naive Baseline:** Fine-tuning naively causes the network to predict only the newest class (Household), dropping historical accuracy to 0% (99.50% forgetting).
2. **Why Standalone EWC Fails in Class-Incremental Learning:** 
   Under CIL without task oracles, newly added output heads lack historical Fisher penalties. Incoming batches only contain the new class, causing the unregularized newest output logits to dominate all predictions.
3. **Synergistic Power of Replay + EWC:** 
   Experience Replay anchors the multi-class decision boundaries, allowing EWC to regularize parameter trajectories. Replay + EWC achieves the highest overall accuracy (**65.62%**), highest task accuracy (**68.00%**), and lowest catastrophic forgetting (**46.50%**).

---

## 19. Limitations
1. **Novelty Semantic Overlap:** When an unseen category shares semantic features with known categories (e.g., household electronic appliances vs tech gadgets), centroid distances can dip below the threshold, lowering recall.
2. **Buffer Scaling:** While a 200-sample buffer works well for 4 classes, scaling to hundreds of classes requires sub-linear exemplar selection or generative replay.
3. **Linear Head Expansion:** Expanding the output head linearly scales parameter count slightly per class.

---

## 20. Future Work
1. **Generative Feature Replay:** Generating synthetic embeddings using diffusion or VAEs to eliminate exemplar storage entirely.
2. **Adaptive Thresholds:** Dynamic, class-specific novelty thresholds instead of a global percentile.
3. **Contrastive Representation Learning:** Training an end-to-end projection head with contrastive loss to maximize margin between category centroids.

---

## 21. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EvoRoute.git
cd EvoRoute

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 22. Run Commands

```bash
# Execute entire pipeline end-to-end (preprocess -> embeddings -> train -> evaluate -> visualize)
python main.py --stage all

# Run individual stages modularly:
python main.py --stage preprocess
python main.py --stage embeddings
python main.py --stage train --ewc-lambda 100.0
python main.py --stage evaluate
python main.py --stage visualize

# Run complete automated verification test suite:
python -m pytest
```

---

## 23. Project Directory Structure

```
EvoRoute/
├── app/
│   └── app.py                            # Modern 5-page interactive Streamlit dashboard
├── data/
│   ├── raw/
│   │   └── ecommerceDataset.csv          # Raw product descriptions catalog
│   └── processed/
│       ├── cleaned_data.csv              # Deduplicated, filtered tabular data
│       ├── train_embeddings.pt           # 384-D MiniLM train embeddings
│       ├── val_embeddings.pt             # 384-D MiniLM validation embeddings
│       └── test_embeddings.pt            # Strictly isolated test embeddings
├── models/                               # Checkpoints (.pt) for all trained methods
├── results/
│   ├── metrics/                          # final_results.json, memory_sensitivity.json, novelty_metrics.json
│   └── plots/                            # 7 publication-ready PNG figures
├── src/
│   ├── config.py                         # Hyperparameters, directories, task definitions
│   ├── data.py                           # Dataset loading, cleaning, stratified splitting
│   ├── embeddings.py                     # Frozen sentence-transformers MiniLM encoder
│   ├── tasks.py                          # Incremental task data slicing & DataLoader iterators
│   ├── model.py                          # EvoMLP with dynamic expansion and weight invariance
│   ├── novelty.py                        # Hyperspherical centroid detector & threshold calibration
│   ├── replay.py                         # ReplayBuffer (budget <= 200) with balanced sampling
│   ├── ewc.py                            # Sample-wise Fisher Information & quadratic loss penalty
│   ├── train.py                          # Continual training loops & joint upper bound
│   ├── evaluate.py                       # Accuracy, task accuracy, and forgetting metrics
│   └── visualize.py                      # 7 publication matplotlib visualizations
├── tests/
│   ├── test_model_expansion.py           # Unit tests for weight invariance during expansion
│   ├── test_replay_buffer.py             # Unit tests for buffer capacity bounds & rebalancing
│   ├── test_ewc.py                       # Unit tests for Fisher accumulation & loss integration
│   ├── test_consistency_audit.py         # Cross-artifact metric consistency audit
│   └── test_streamlit_runtime.py         # End-to-end dashboard & model runtime validation
├── main.py                               # CLI entrypoint supporting all stages
├── requirements.txt
└── README.md
```

---

## 24. Reproducibility
- **Deterministic Seeding:** `set_seed(42)` sets seeds across PyTorch, NumPy, Python standard library, and sets `torch.backends.cudnn.deterministic = True`.
- **Precomputed Embeddings:** Feature vectors are deterministically cached to ensure identical initial representations across runs.
- **Zero Hardcoded Numbers:** All plots, tables, and dashboard elements load directly from generated metrics files.

---

## 25. Hackathon Demo Instructions (2–4 Minute Pitch Sequence)

Launch the interactive dashboard:
```bash
python main.py --stage demo
# Or directly:
streamlit run app/app.py
```

### Demonstration Walkthrough:
- **Step 1: Explain the Problem (Page 1):** Point to the problem card: "Traditional models overwrite old categories when learning new ones."
- **Step 2: Show Task 1 (Page 2):** Platform launches with Books & Clothing (99% accuracy).
- **Step 3: Introduce Electronics (Page 2):** Show novelty detection flagging incoming headphones as unfamiliar ($d=0.859 > \tau=0.811$).
- **Step 4: Show Dynamic Expansion (Page 2):** Head expands from 2 to 3 classes without resetting prior weights.
- **Step 5: Demonstrate Forgetting (Page 4):** Switch to Live Forgetting Demonstration. Show Naive fine-tuning dropping Books to 9% and Clothing to 0%.
- **Step 6: Show Replay + EWC Retention (Page 4):** Show Replay + EWC maintaining Books at 67.5% and Clothing at 90.0%.
- **Step 7: Introduce Household (Page 2):** Head expands to 4 classes; Replay rebalances memory to 50/class.
- **Step 8: Show Final Benchmark (Page 3):** Point to Official Benchmark Table: Replay + EWC achieves **65.62% overall accuracy** and **46.50% forgetting** vs Naive (25.00% / 99.50%). Clarify Joint (93.75%) is offline.
- **Step 9: Highlight Memory Efficiency (Page 4):** Show Memory Ablation Curve: a compact 200-sample buffer preserves knowledge across the entire catalog.

---

## License & Citation
Developed for the Deep Learning Continual Learning Hackathon (Track 5). Released under the MIT License.
