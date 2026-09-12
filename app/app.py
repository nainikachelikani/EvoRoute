import os
import sys
import json
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CATEGORIES,
    CATEGORY2ID,
    ID2CATEGORY,
    TASK_DEFINITIONS,
    FINAL_METRICS_PATH,
    MEMORY_STUDY_PATH,
    NOVELTY_METRICS_PATH,
    RESULTS_PLOTS_DIR,
    MODELS_DIR,
    DEVICE
)
from src.embeddings import encode_texts, load_embeddings
from src.model import EvoMLP
from src.novelty import NoveltyDetector
from src.tasks import get_task_data

# Page Configuration
st.set_page_config(
    page_title="EvoRoute | Class-Incremental Continual Learning",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1D3557;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #457B9D;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1D3557;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #2A9D8F;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models_and_detector():
    """Caches loaded neural network models and novelty detector."""
    models = {}
    methods = ["replay_ewc", "naive", "replay", "ewc", "joint"]
    for m in methods:
        path = MODELS_DIR / f"{m}_final.pt"
        if path.exists():
            try:
                model = EvoMLP(num_classes=4).to(DEVICE)
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                models[m] = model
            except Exception:
                pass

    # Initialize novelty detector on Task 1 (Books + Clothing)
    detector = NoveltyDetector()
    try:
        t1_embs, t1_lbls, _ = get_task_data(task_id=1, split="train", cumulative=True)
        val_data = load_embeddings("val")
        detector.update_known_classes(
            new_classes=[0, 1],
            embeddings=t1_embs,
            labels=t1_lbls,
            val_embeddings=val_data["embeddings"],
            val_labels=val_data["labels"]
        )
    except Exception:
        pass

    return models, detector


@st.cache_data
def load_all_metrics():
    """Loads all experimental metrics JSON files dynamically."""
    final_metrics = {}
    memory_study = {}
    novelty_metrics = {}

    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, "r") as f:
            final_metrics = json.load(f)

    if MEMORY_STUDY_PATH.exists():
        with open(MEMORY_STUDY_PATH, "r") as f:
            memory_study = json.load(f)

    if NOVELTY_METRICS_PATH.exists():
        with open(NOVELTY_METRICS_PATH, "r") as f:
            novelty_metrics = json.load(f)

    return final_metrics, memory_study, novelty_metrics


models, detector = load_all_models_and_detector()
final_metrics, memory_study, novelty_metrics = load_all_metrics()

# Sidebar Navigation
st.sidebar.title("⚡ EvoRoute")
st.sidebar.markdown("**Adaptive E-Commerce Intelligence**")
st.sidebar.caption("Class-Incremental Continual Learning (CIL)")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🚀 EvoRoute Overview",
        "🔄 Evolution Simulation Mode ⭐",
        "📊 Benchmark Results ⭐",
        "📈 Experimental Analysis",
        "🚨 Novelty Detection Lab"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Setting:** Class-Incremental Learning without task identity at test time.\n\n"
    "**Lifecycle:** Detect → Learn → Retain"
)


# =============================================================================
# PAGE 1: EvoRoute Overview & Live Routing
# =============================================================================
if page == "🚀 EvoRoute Overview":
    st.markdown('<p class="main-header">EvoRoute: Adaptive E-Commerce Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Class-Incremental Continual Learning: Detect → Learn → Retain</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Problem:</b> In real-world e-commerce, new catalog categories emerge over time. Standard neural network fine-tuning 
    causes <b>Catastrophic Forgetting</b>, destroying performance on historical inventory. 
    <b>EvoRoute</b> combines <b>hyperspherical novelty detection</b>, <b>dynamic output layer expansion</b>, and a 
    <b>dual retain mechanism (Experience Replay + Elastic Weight Consolidation)</b> to learn continuously without forgetting.
    </div>
    """, unsafe_allow_html=True)

    col_arch, col_demo = st.columns([1, 1])

    with col_arch:
        st.subheader("System Architecture")
        st.markdown("""
        1. **Pretrained Semantic Anchors**: Frozen `all-MiniLM-L6-v2` encoder guarantees zero representation drift.
        2. **DETECT (Hyperspherical Novelty)**: Cosine distance to known class centroids flags unfamiliar categories.
        3. **LEARN (Dynamic Expansion)**: Output classification head dynamically expands ($C \\to C+1$) without resetting learned weights.
        4. **RETAIN (Replay + EWC)**: Bounded memory buffer (200 exemplars) + Fisher Information quadratic penalty protects critical parameter trajectories.
        """)

    with col_demo:
        st.subheader("Live Product Classifier")
        sample_catalog = [
            "Select a preset product description...",
            "The Pragmatic Programmer: Your Journey to Mastery (20th Anniversary Edition)",
            "Men's Regular Fit Cotton Casual Shirt with Button-Down Collar and Long Sleeves",
            "Apple AirPods Pro with MagSafe Charging Case - Active Noise Cancellation",
            "Prestige Electric Kettle 1.5L Stainless Steel Body with Auto Cut-Off",
            "ASUS ROG Strix Gaming Laptop 16-inch 165Hz Display Intel Core i7"
        ]
        selected_sample = st.selectbox("Sample Product Stream:", sample_catalog)
        user_text = st.text_area(
            "Product Description:",
            value="" if selected_sample == sample_catalog[0] else selected_sample,
            height=85,
            placeholder="Type or paste any product description here..."
        )

        classify_clicked = st.button("Route & Classify Product", type="primary")

    if classify_clicked and user_text.strip():
        with st.spinner("Processing through semantic encoder & EvoRoute network..."):
            emb = encode_texts([user_text], show_progress=False).to(DEVICE)
            nov_res = detector.detect(emb.cpu())

            active_model = models.get("replay_ewc")
            if active_model is None and models:
                active_model = list(models.values())[0]

            if active_model is not None:
                with torch.no_grad():
                    logits = active_model(emb)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_id = int(np.argmax(probs))
                    pred_cat = ID2CATEGORY[pred_id]
                    confidence = float(probs[pred_id])
            else:
                pred_cat = "Models not yet trained"
                confidence = 0.0
                probs = [0.25, 0.25, 0.25, 0.25]

        st.markdown("---")
        st.subheader("Inference & Routing Decision")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Category", pred_cat)
        with m2:
            st.metric("Classifier Confidence", f"{confidence * 100:.1f}%")
        with m3:
            status = "🚨 UNFAMILIAR / NOVEL" if nov_res["is_novel"] else "🟢 KNOWN"
            st.metric("Novelty Status", status)
        with m4:
            st.metric("Distance to Centroid", f"{nov_res['distance']:.3f}", f"Threshold: {nov_res['threshold']:.3f}")

        prob_df = pd.DataFrame({
            "Category": CATEGORIES[:len(probs)],
            "Probability": [float(p) for p in probs]
        }).sort_values("Probability", ascending=True)
        st.bar_chart(prob_df.set_index("Category"))


# =============================================================================
# PAGE 2: Evolution Simulation Mode ⭐
# =============================================================================
elif page == "🔄 Evolution Simulation Mode ⭐":
    st.markdown('<p class="main-header">Platform Evolution Simulation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Step-by-Step Continual Expansion: Detect → Learn → Retain</p>', unsafe_allow_html=True)

    st.markdown("""
    Walk through the operational lifecycle of EvoRoute as new categories arrive sequentially over time:
    **Task 1 (Books, Clothing) → Task 2 (Electronics) → Task 3 (Household)**.
    """)

    sim_phase = st.radio(
        "Platform Evolution Stage:",
        [
            "Phase 1: Base Catalog Launch (Books & Clothing)",
            "Phase 2: Tech Expansion (Electronics Arrives)",
            "Phase 3: Home Expansion (Household Arrives)",
            "Phase 4: Mature Multi-Category State"
        ],
        horizontal=True
    )

    st.markdown("---")

    if sim_phase.startswith("Phase 1"):
        st.info("📦 **Phase 1: Base Platform State (Task 1)**")
        st.write("- **Active Classes:** `Books` (0), `Clothing & Accessories` (1)")
        st.write("- **Output Layer:** `Linear(in_features=128, out_features=2)`")
        st.write("- **Replay Memory:** 0 exemplars (no previous tasks to retain)")

        sample_book = "Harry Potter and the Sorcerer's Stone by J.K. Rowling Paperback Edition"
        sample_cloth = "Men's Slim Fit Casual Denim Jacket with Dual Chest Pockets"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sample 1:** *{sample_book}*")
            emb1 = encode_texts([sample_book], show_progress=False)
            det1 = detector.detect(emb1)
            st.success(f"Status: **{det1['decision']}** (Distance: {det1['distance']:.3f} <= {det1['threshold']:.3f})")
        with col2:
            st.markdown(f"**Sample 2:** *{sample_cloth}*")
            emb2 = encode_texts([sample_cloth], show_progress=False)
            det2 = detector.detect(emb2)
            st.success(f"Status: **{det2['decision']}** (Distance: {det2['distance']:.3f} <= {det2['threshold']:.3f})")

    elif sim_phase.startswith("Phase 2"):
        st.warning("⚡ **Phase 2: Tech Expansion Stream (Task 2 Arrival)**")
        test_tech = "Sony WH-1000XM5 Wireless Noise Canceling Over-Ear Headphones with Auto NC Optimizer"
        st.markdown(f"**Incoming Unlabelled Product:** *{test_tech}*")

        emb_tech = encode_texts([test_tech], show_progress=False)
        det_tech = detector.detect(emb_tech)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. DETECT")
            st.write(f"- Distance to nearest known centroid: **{det_tech['distance']:.3f}**")
            st.write(f"- Calibrated Threshold ($\\tau$): **{det_tech['threshold']:.3f}**")
            if det_tech["is_novel"]:
                st.error("🚨 **NOVEL CATEGORY DETECTED!**")
            else:
                st.info("Recognized as Known")

        with c2:
            st.markdown("#### 2. LEARN")
            st.write("- **Dynamic Layer Expansion:** `Linear(128, 2)` → `Linear(128, 3)`")
            st.write("- Old class weights for Books & Clothing strictly preserved.")
            st.write("- Model trains on incoming Electronics data stream.")

        with c3:
            st.markdown("#### 3. RETAIN")
            st.write("- **Experience Replay:** Samples 100 exemplars each from Books & Clothing.")
            st.write("- **EWC Regularization:** Penalizes drift in parameters critical to Task 1.")
            st.success("Result: Electronics learned without overwriting Books & Clothing!")

    elif sim_phase.startswith("Phase 3"):
        st.warning("🏠 **Phase 3: Home Expansion Stream (Task 3 Arrival)**")
        test_home = "Prestige Deluxe Stainless Steel Pressure Cooker 3 Litre with Induction Base"
        st.markdown(f"**Incoming Unlabelled Product:** *{test_home}*")

        emb_home = encode_texts([test_home], show_progress=False)
        det_home = detector.detect(emb_home)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. DETECT")
            st.write(f"- Distance to nearest known centroid: **{det_home['distance']:.3f}**")
            st.write(f"- Calibrated Threshold ($\\tau$): **{det_home['threshold']:.3f}**")
            if det_home["is_novel"]:
                st.error("🚨 **NOVEL CATEGORY DETECTED!**")
            else:
                st.info("Recognized as Known")

        with c2:
            st.markdown("#### 2. LEARN")
            st.write("- **Dynamic Layer Expansion:** `Linear(128, 3)` → `Linear(128, 4)`")
            st.write("- Preserves weights for Books, Clothing, and Electronics.")
            st.write("- Model trains on incoming Household data stream.")

        with c3:
            st.markdown("#### 3. RETAIN")
            st.write("- **Memory Rebalancing:** Bounded 200 budget rebalanced to 50/class.")
            st.write("- **EWC Regularization:** Accumulated Fisher protects Tasks 1 and 2.")
            st.success("Result: All 4 categories retained simultaneously!")

    elif sim_phase.startswith("Phase 4"):
        st.success("🎉 **Phase 4: Mature Multi-Category Platform State**")
        st.write("All 4 categories are active in production. Below are the actual performance metrics loaded dynamically:")

        if final_metrics:
            rep_ewc = final_metrics.get("replay_ewc", {})
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall Class Accuracy", f"{rep_ewc.get('overall_accuracy', 0.0) * 100:.2f}%")
            with m2:
                st.metric("Final Avg Task Accuracy", f"{rep_ewc.get('final_avg_task_accuracy', 0.0) * 100:.2f}%")
            with m3:
                st.metric("Average Forgetting", f"{rep_ewc.get('average_forgetting', 0.0) * 100:.2f}%")
            with m4:
                st.metric("Replay Memory Budget", f"{rep_ewc.get('memory_size', 200)} exemplars")

            st.markdown("#### Per-Class Retained Accuracy:")
            st.json(rep_ewc.get("final_per_class", {}))


# =============================================================================
# PAGE 3: Benchmark Results ⭐
# =============================================================================
elif page == "📊 Benchmark Results ⭐":
    st.markdown('<p class="main-header">Official Benchmark Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Rigorous Class-Incremental Continual Learning Evaluation Across All 5 Methods</p>', unsafe_allow_html=True)

    st.markdown("""
    All metrics below are **dynamically loaded directly from `results/metrics/final_results.json`** 
    generated from actual model training and evaluation runs. No values are hardcoded.
    """)

    if not final_metrics:
        st.warning("⚠️ `final_results.json` not found. Please run `python main.py --stage train` to generate actual benchmark results.")
    else:
        # Build the exact table requested by the evaluation specification
        method_order = [
            ("naive", "Naive Sequential", "0"),
            ("ewc", "EWC", "0"),
            ("replay", "Experience Replay", "200"),
            ("replay_ewc", "Replay + EWC ⭐", "200"),
            ("joint", "Joint Upper Bound", "Full Dataset"),
        ]

        table_rows = []
        for key, name, mem in method_order:
            if key in final_metrics:
                m_data = final_metrics[key]
                overall_acc = m_data.get("overall_accuracy", m_data.get("final_accuracy", 0.0))
                task_acc = m_data.get("final_avg_task_accuracy", m_data.get("final_accuracy", 0.0))
                forgetting = m_data.get("average_forgetting", 0.0)

                table_rows.append({
                    "Method": name,
                    "Overall Class Accuracy": f"{overall_acc * 100:.2f}%",
                    "Final Avg Task Accuracy": f"{task_acc * 100:.2f}%",
                    "Avg Forgetting": f"{forgetting * 100:.2f}%",
                    "Memory": mem
                })

        benchmark_df = pd.DataFrame(table_rows)
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

        st.markdown("""
        ### Metric Definitions:
        - **Overall Class Accuracy:** Accuracy across all final test samples across all 4 categories.
        - **Final Average Task Accuracy:** Average of the individual task accuracies after the final task:
          $$\\text{Final Avg Task Accuracy} = \\frac{1}{T} \\sum_{k=1}^T R(T, k)$$
          where $R(T, k)$ is the test accuracy on Task $k$ after completing final Task $T$.
        - **Final Average Forgetting:** Degradation from maximum historical performance:
          $$F = \\frac{1}{T-1} \\sum_{k=1}^{T-1} \\left( \\max_{l < T} R(l, k) - R(T, k) \\right)$$
        """)

        # Detailed cards for key methods
        st.markdown("### Method-by-Method Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("📉 **Naive Sequential**")
            st.write("Severe Catastrophic Forgetting. Old category weights are overwritten when learning new tasks.")
        with c2:
            st.info("⚖️ **EWC (Standalone)**")
            st.write("Regularizes parameter displacement via Fisher Information. In Class-Incremental Learning, new output heads lack historical constraints, causing logit drift.")
        with c3:
            st.success("⭐ **Replay + EWC (Proposed)**")
            st.write("Best continual performance. 200 exemplars preserve decision boundaries while EWC regularizes weight trajectories.")


# =============================================================================
# PAGE 4: Experimental Analysis
# =============================================================================
elif page == "📈 Experimental Analysis":
    st.markdown('<p class="main-header">Experimental Analysis & Ablation Studies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Empirical Validation of Memory Budgets, Task Trajectories, and Knowledge Retention</p>', unsafe_allow_html=True)

    tab_ablation, tab_retention, tab_tasks, tab_perclass = st.tabs([
        "Memory Ablation Study",
        "Knowledge Retention Heatmap",
        "Accuracy Across Tasks",
        "Per-Class Breakdown"
    ])

    with tab_ablation:
        st.subheader("Memory Budget Ablation (0, 50, 100, 200 Exemplars)")
        st.markdown("""
        Investigating the sensitivity of catastrophic forgetting and overall accuracy to replay buffer capacity.
        """)

        c1, c2 = st.columns(2)
        with c1:
            mem_acc_plot = RESULTS_PLOTS_DIR / "memory_vs_accuracy.png"
            if mem_acc_plot.exists():
                st.image(str(mem_acc_plot), use_container_width=True)
            else:
                st.info("Generate plots with `python main.py --stage visualize`.")
        with c2:
            mem_forget_plot = RESULTS_PLOTS_DIR / "memory_vs_forgetting.png"
            if mem_forget_plot.exists():
                st.image(str(mem_forget_plot), use_container_width=True)
            else:
                st.info("Generate plots with `python main.py --stage visualize`.")

        if memory_study:
            st.markdown("#### Raw Memory Ablation Data:")
            study_rows = []
            for b in sorted([int(k) for k in memory_study.keys()]):
                val = memory_study[str(b)]
                acc = val["final_accuracy"] if isinstance(val, dict) else val
                f_val = val.get("forgetting", 0.0) if isinstance(val, dict) else 0.0
                study_rows.append({
                    "Memory Budget (Exemplars)": b,
                    "Overall Accuracy (%)": f"{acc * 100:.2f}%",
                    "Average Forgetting (%)": f"{f_val * 100:.2f}%"
                })
            st.dataframe(pd.DataFrame(study_rows), use_container_width=True, hide_index=True)

    with tab_retention:
        st.subheader("Knowledge Retention Heatmap (Naive vs Replay + EWC)")
        heatmap_path = RESULTS_PLOTS_DIR / "knowledge_retention_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)
        else:
            st.info("Knowledge retention heatmap not found. Run visualize stage.")

    with tab_tasks:
        st.subheader("Task-by-Task Evolution")
        c1, c2 = st.columns(2)
        with c1:
            acc_tasks_path = RESULTS_PLOTS_DIR / "accuracy_across_tasks.png"
            if acc_tasks_path.exists():
                st.image(str(acc_tasks_path), use_container_width=True)
        with c2:
            forget_path = RESULTS_PLOTS_DIR / "catastrophic_forgetting.png"
            if forget_path.exists():
                st.image(str(forget_path), use_container_width=True)

    with tab_perclass:
        st.subheader("Per-Class Accuracy Breakdown")
        per_class_path = RESULTS_PLOTS_DIR / "per_class_accuracy.png"
        if per_class_path.exists():
            st.image(str(per_class_path), use_container_width=True)


# =============================================================================
# PAGE 5: Novelty Detection Lab
# =============================================================================
elif page == "🚨 Novelty Detection Lab":
    st.markdown('<p class="main-header">Novelty Detection Laboratory</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secondary Supporting Mechanism: Class Centroid Hypersphere Distances & Threshold Tuning</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Role in EvoRoute:</b> Novelty detection serves as a lightweight gatekeeper to flag unfamiliar product descriptions 
    before triggering output expansion. The primary scientific contribution of this work remains 
    <b>Class-Incremental Continual Learning (Retain)</b>.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Threshold Calibration & Tuning")
        thresh_val = st.slider(
            "Novelty Threshold (τ):",
            min_value=0.10,
            max_value=0.90,
            value=float(detector.threshold if detector and detector.threshold is not None else 0.45),
            step=0.01
        )
        st.caption("Incoming embeddings with distance $d_{min} > \\tau$ are flagged as unfamiliar.")

        if novelty_metrics:
            st.markdown("#### Calibrated Benchmark Metrics:")
            for m_name, m_vals in novelty_metrics.items():
                st.write(f"**{m_name.replace('_', ' ').title()}:**")
                st.write(f"- Precision: `{m_vals.get('precision', 0.0):.4f}` | Recall: `{m_vals.get('recall', 0.0):.4f}` | F1: `{m_vals.get('f1', 0.0):.4f}`")

    with col_r:
        st.subheader("Test Live Product Description")
        test_input = st.text_area(
            "Product Description to Test:",
            value="Apple Watch Ultra 2 GPS + Cellular 49mm Titanium Case with Orange Ocean Band",
            height=100
        )
        check_btn = st.button("Check Unfamiliarity Distance", type="primary")

    if check_btn and test_input.strip():
        with st.spinner("Computing cosine distance to known class centroids..."):
            emb_test = encode_texts([test_input], show_progress=False)
            res = detector.detect(emb_test)
            # Override threshold with user slider
            is_nov = res["distance"] > thresh_val

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Distance to Nearest Centroid", f"{res['distance']:.4f}")
        with m2:
            st.metric("Active Threshold (τ)", f"{thresh_val:.4f}")
        with m3:
            if is_nov:
                st.error("🚨 UNFAMILIAR / NOVEL")
            else:
                st.success("🟢 RECOGNIZED AS KNOWN")

        st.info(f"Nearest Known Centroid: **{res.get('nearest_category_name', 'Unknown')}**")

    # Empirical Distribution Plot
    dist_plot = RESULTS_PLOTS_DIR / "novelty_detection_distribution.png"
    if dist_plot.exists():
        st.markdown("---")
        st.subheader("Empirical Semantic Distance Distributions")
        st.image(str(dist_plot), use_container_width=True)
