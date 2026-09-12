import random
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS_PER_TASK,
    REPLAY_MEMORY_BUDGET,
    REPLAY_SAMPLE_RATIO,
    EWC_LAMBDA,
    LWF_TEMPERATURE,
    LWF_LAMBDA,
    TASK_CLASSES,
    TASK_CUMULATIVE_CLASSES,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RANDOM_SEED,
    BASELINE_CHECKPOINT_PATH,
    EVOROUTE_BR_CHECKPOINT_PATH,
    EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH,
    METRIC_TOLERANCE,
    MIN_NEWEST_CLASS_ACCURACY,
    MIN_OLD_CLASS_ACCURACY,
    MAX_FORGETTING_INCREASE
)
from src.model import EvoMLP
from src.tasks import get_task_data, get_task_loader, get_joint_loader
from src.replay import ReplayBuffer, construct_incremental_batch
from src.ewc import EWC
from src.lwf import LwF
from src.evaluate import (
    evaluate_model_on_classes,
    evaluate_task_accuracies,
    compute_forgetting,
    compute_average_forgetting,
    compute_final_average_task_accuracy,
    compute_prediction_distribution,
    validate_checkpoint
)
from src.recency_bias import compute_dataset_aware_recency_bias
from src.calibration import run_post_hoc_calibration, CalibratedModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seed across torch, numpy, and python for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_task(
    model: EvoMLP,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = EPOCHS_PER_TASK,
    replay_buffer: ReplayBuffer = None,
    ewc: EWC = None,
    lwf: LwF = None,
    replay_strategy: str = "original",
    device: str = DEVICE
):
    """
    Trains model on a single task.
    Optionally mixes replay exemplars, adds EWC quadratic penalty, or adds LwF distillation loss.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs = batch["embedding"].to(device)
            targets = batch["label"].to(device)

            # Mix with replay buffer if active
            if replay_buffer is not None and replay_buffer.total_samples > 0:
                if replay_strategy == "original":
                    n_replay = max(1, int(inputs.size(0) * replay_buffer.sample_ratio))
                    replay_inputs, replay_targets = replay_buffer.sample(n_replay)
                    if len(replay_inputs) > 0:
                        inputs = torch.cat([inputs, replay_inputs.to(device)], dim=0)
                        targets = torch.cat([targets, replay_targets.to(device)], dim=0)
                else:
                    inputs, targets = construct_incremental_batch(
                        inputs, targets, replay_buffer, strategy=replay_strategy, batch_size=inputs.size(0), device=device
                    )

            optimizer.zero_grad()
            logits = model(inputs)
            cls_loss = criterion(logits, targets)

            # Add continual learning regularization loss
            reg_loss = torch.tensor(0.0, device=device)
            if ewc is not None:
                reg_loss = ewc.penalty(model)
                total_loss = cls_loss + reg_loss
            elif lwf is not None and lwf.has_teacher:
                kd_loss = lwf.compute_distillation_loss(student_logits=logits, inputs=inputs)
                reg_loss = lwf.lwf_lambda * kd_loss
                total_loss = cls_loss + reg_loss
            else:
                total_loss = cls_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_cls = epoch_cls_loss / max(1, num_batches)
        avg_reg = epoch_reg_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            if ewc is not None:
                logger.info(f"  Epoch {epoch+1}/{epochs} - Total: {avg_loss:.4f} (CE: {avg_cls:.4f}, EWC: {avg_reg:.4f})")
            elif lwf is not None and lwf.has_teacher:
                logger.info(f"  Epoch {epoch+1}/{epochs} - Total: {avg_loss:.4f} (CE: {avg_cls:.4f}, LwF KD: {avg_reg:.4f})")
            else:
                logger.info(f"  Epoch {epoch+1}/{epochs} - Total Loss: {avg_loss:.4f}")


def train_continual_method(
    method: str = "replay_ewc",
    memory_budget: int = REPLAY_MEMORY_BUDGET,
    ewc_lambda: float = EWC_LAMBDA,
    lwf_temperature: float = LWF_TEMPERATURE,
    lwf_lambda: float = LWF_LAMBDA,
    epochs: int = EPOCHS_PER_TASK,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Simulates sequential learning across Task 1 (Books, Clothing) -> Task 2 (Electronics) -> Task 3 (Household).
    
    Supported methods:
    - 'naive': Sequential fine-tuning, no replay, no EWC (exhibits catastrophic forgetting).
    - 'ewc': Elastic Weight Consolidation with Fisher regularization.
    - 'replay': Experience Replay bounded by memory budget.
    - 'lwf': Learning without Forgetting via frozen teacher distillation (exemplar-free).
    - 'replay_ewc': Combined Replay + EWC (primary proposed method).
    """
    set_seed(RANDOM_SEED)
    logger.info(f"================ STARTING CONTINUAL LEARNING METHOD: {method.upper()} ================")
    
    # Initialize base model with 2 classes for Task 1
    model = EvoMLP(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    replay_buffer = None
    if "replay" in method.lower():
        replay_buffer = ReplayBuffer(max_budget=memory_budget, sample_ratio=REPLAY_SAMPLE_RATIO)

    ewc = None
    if "ewc" in method.lower():
        ewc = EWC(model, ewc_lambda=ewc_lambda, device=device)

    lwf = None
    if method.lower() == "lwf":
        lwf = LwF(temperature=lwf_temperature, lwf_lambda=lwf_lambda, device=device)

    # Performance matrix: R[i, j] = accuracy on task j after training task i
    R = np.zeros((3, 3))
    task_history = []

    for task_id in [1, 2, 3]:
        logger.info(f"\n--- [Method: {method.upper()}] Starting Task {task_id} ---")
        cumulative_classes = TASK_CUMULATIVE_CLASSES[task_id]
        new_classes = TASK_CLASSES[task_id]
        required_classes = len(cumulative_classes)

        # Expand output layer dynamically if required
        if model.num_classes < required_classes:
            model.expand_classes(required_classes)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Get task training data
        train_loader = get_task_loader(task_id=task_id, split="train", cumulative=False, shuffle=True)
        train_embs, train_lbls, _ = get_task_data(task_id=task_id, split="train", cumulative=False)

        # Train on current task
        train_single_task(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            replay_buffer=replay_buffer if (task_id > 1 and replay_buffer is not None) else None,
            ewc=ewc if (task_id > 1 and ewc is not None) else None,
            lwf=lwf if (task_id > 1 and lwf is not None) else None,
            device=device
        )

        # Post-task actions: update replay buffer, EWC Fisher matrix, or LwF teacher
        if replay_buffer is not None:
            replay_buffer.add_examples(train_embs, train_lbls)
            replay_buffer.log_memory_composition(task_id=task_id)

        if ewc is not None:
            # Estimate Fisher Information on current task using mathematically exact sample gradients
            task_eval_loader = get_task_loader(task_id=task_id, split="train", cumulative=False, shuffle=False)
            ewc.compute_fisher(task_eval_loader, num_samples=500)

        if lwf is not None:
            # Snapshot frozen teacher for knowledge distillation in subsequent tasks
            lwf.set_teacher(model)

        # Save checkpoint after each task
        task_checkpoint_path = MODELS_DIR / f"{method}_task{task_id}.pt"
        torch.save(model.state_dict(), task_checkpoint_path)

        # Evaluate performance on test set for all seen tasks
        acc_dict = evaluate_task_accuracies(model, current_task=task_id, split="test", device=device)
        for t_eval in range(1, task_id + 1):
            R[task_id - 1, t_eval - 1] = acc_dict[t_eval]

        # Overall accuracy on cumulative seen classes
        overall_eval = evaluate_model_on_classes(model, split="test", allowed_classes=cumulative_classes, device=device)
        
        logger.info(f"Task {task_id} Complete -> Overall Seen Accuracy: {overall_eval['accuracy']:.4f}")
        for cat_name, cat_acc in overall_eval["per_class_accuracy"].items():
            logger.info(f"    {cat_name}: {cat_acc:.4f}")

        task_history.append({
            "task_id": task_id,
            "overall_accuracy": overall_eval["accuracy"],
            "per_class_accuracy": overall_eval["per_class_accuracy"],
            "seen_task_accuracies": {f"Task_{t}": R[task_id - 1, t - 1] for t in range(1, task_id + 1)},
            "memory_size": replay_buffer.total_samples if replay_buffer is not None else 0
        })

    # Compute Primary Metrics
    overall_accuracy = task_history[-1]["overall_accuracy"]
    final_avg_task_acc = compute_final_average_task_accuracy(R)
    forgetting_results = compute_forgetting(R)
    avg_forgetting = forgetting_results["average_forgetting"]

    # Compute Empirical Prediction Distribution and Recency Bias on Test Set
    pred_dist_data = compute_prediction_distribution(model, split="test", device=device)
    logger.info(f"\n==================== [Method: {method.upper()} FINAL SUMMARY] ====================")
    logger.info(f"  PRIMARY METRIC 1 (Overall Class Accuracy):      {overall_accuracy * 100:.2f}%")
    logger.info(f"  PRIMARY METRIC 2 (Final Average Task Accuracy): {final_avg_task_acc * 100:.2f}%")
    logger.info(f"  PRIMARY METRIC 3 (Final Average Forgetting):    {avg_forgetting * 100:.2f}%")
    logger.info(f"  RECENCY BIAS (P(Household) - 0.25):             {pred_dist_data.get('recency_bias', 0.0):+.4f}")
    logger.info(f"  Test Predictions: {pred_dist_data.get('percentages', {})}")
    logger.info(f"==================================================================================")

    # Save final model with baseline preservation protection
    save_path = MODELS_DIR / f"{method}_final.pt"
    if method == "replay_ewc" and save_path.exists():
        logger.warning(f"Baseline checkpoint {save_path} is immutable. Saving retrained baseline weights to {MODELS_DIR / 'replay_ewc_retrained.pt'} to protect baseline manifest.")
        save_path = MODELS_DIR / "replay_ewc_retrained.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved {method} model to {save_path}")

    return {
        "method": method,
        "overall_accuracy": overall_accuracy,
        "final_avg_task_accuracy": final_avg_task_acc,
        "final_average_accuracy": overall_accuracy,
        "final_accuracy": overall_accuracy,
        "average_forgetting": avg_forgetting,
        "forgetting": forgetting_results,
        "final_per_class": task_history[-1]["per_class_accuracy"],
        "prediction_distribution": pred_dist_data.get("distribution", {}),
        "prediction_distribution_percentages": pred_dist_data.get("percentages", {}),
        "recency_bias": pred_dist_data.get("recency_bias", 0.0),
        "prediction_distribution_details": pred_dist_data,
        "task_history": task_history,
        "accuracy_matrix": R.tolist(),
        "memory_size": replay_buffer.total_samples if replay_buffer is not None else 0,
        "model_path": str(save_path)
    }


def run_balanced_finetuning(
    model: EvoMLP,
    replay_buffer: ReplayBuffer,
    lr: float = 1e-4,
    epochs: int = 10,
    patience: int = 2,
    device: str = DEVICE
) -> Tuple[EvoMLP, Dict[str, Any]]:
    """
    Post-task balanced fine-tuning on strictly bounded replay exemplars (<= 200 samples).
    Uses a lower learning rate (1e-4) to re-align output decision boundaries without destroying feature representations.
    Employs lexicographic early stopping: monitors (Macro F1, Balanced Accuracy, Overall Accuracy) on validation split.
    """
    if replay_buffer.total_samples == 0:
        return model, {"epochs_trained": 0, "status": "empty_replay_buffer"}

    all_embs, all_lbls = replay_buffer.get_all()
    # Double check budget invariant
    assert len(all_lbls) <= REPLAY_MEMORY_BUDGET, f"Replay buffer size {len(all_lbls)} exceeds budget {REPLAY_MEMORY_BUDGET}!"

    dataset = TensorDataset(all_embs, all_lbls)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    init_val = validate_checkpoint(model, split="val", device=device)
    best_tuple = init_val["lexicographic_tuple"]
    best_epoch = 0
    patience_counter = 0

    history = [{
        "epoch": 0,
        "val_macro_f1": init_val["macro_f1"],
        "val_balanced_acc": init_val["balanced_accuracy"],
        "val_overall_acc": init_val["overall_accuracy"]
    }]

    logger.info(f"Starting post-task balanced fine-tuning on {len(all_lbls)} exemplars (Max {epochs} epochs, LR={lr}). Initial Val Macro F1: {init_val['macro_f1']:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_embs, batch_lbls in loader:
            optimizer.zero_grad()
            logits = model(batch_embs.to(device))
            loss = criterion(logits, batch_lbls.to(device))
            loss.backward()
            optimizer.step()

        # Validate strictly on validation split
        val_res = validate_checkpoint(model, split="val", device=device)
        curr_tuple = val_res["lexicographic_tuple"]

        history.append({
            "epoch": epoch,
            "val_macro_f1": val_res["macro_f1"],
            "val_balanced_acc": val_res["balanced_accuracy"],
            "val_overall_acc": val_res["overall_accuracy"]
        })

        # Anti-collapse safety check: newest accuracy must not crash
        newest_acc = val_res["recency_metrics"].get("newest_class_accuracy", 0.0)
        old_acc = val_res["recency_metrics"].get("old_class_accuracy", 0.0)

        is_better = False
        if newest_acc >= MIN_NEWEST_CLASS_ACCURACY and old_acc >= MIN_OLD_CLASS_ACCURACY:
            if curr_tuple[0] > best_tuple[0] + METRIC_TOLERANCE:
                is_better = True
            elif abs(curr_tuple[0] - best_tuple[0]) <= METRIC_TOLERANCE:
                if curr_tuple[1] > best_tuple[1] + METRIC_TOLERANCE:
                    is_better = True
                elif abs(curr_tuple[1] - best_tuple[1]) <= METRIC_TOLERANCE and curr_tuple[2] > best_tuple[2] + METRIC_TOLERANCE:
                    is_better = True

        if is_better:
            best_tuple = curr_tuple
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"  Epoch {epoch}: New best validation score {best_tuple} (Newest Acc: {newest_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Epoch {epoch}: Early stopping triggered (Patience={patience}). Best epoch: {best_epoch}")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    final_val = validate_checkpoint(model, split="val", device=device)

    return model, {
        "best_epoch": best_epoch,
        "history": history,
        "initial_val": init_val,
        "final_val": final_val
    }


def train_evoroute_br_candidate(
    replay_strategy: str = "class_balanced",
    memory_budget: int = REPLAY_MEMORY_BUDGET,
    ewc_lambda: float = EWC_LAMBDA,
    epochs: int = EPOCHS_PER_TASK,
    run_finetuning: bool = True,
    device: str = DEVICE,
    save_name: str = "evoroute_br_candidate"
) -> Dict[str, Any]:
    """
    Trains an EvoRoute-BR candidate configuration across tasks 1, 2, and 3.
    Replay strategy can be '50_50' or 'class_balanced'.
    Optionally applies post-task balanced fine-tuning on the 200 exemplar buffer.
    All checkpoints are saved to models/ with unique candidate names, never overwriting the baseline.
    """
    set_seed(RANDOM_SEED)
    logger.info(f"================ STARTING EVOROUTE-BR CANDIDATE TRAINING (Strategy: {replay_strategy}, FT: {run_finetuning}) ================")

    model = EvoMLP(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(max_budget=memory_budget, sample_ratio=REPLAY_SAMPLE_RATIO)
    ewc = EWC(model, ewc_lambda=ewc_lambda, device=device)

    R_val = np.zeros((3, 3))
    task_history = []

    for task_id in [1, 2, 3]:
        logger.info(f"\n--- [EvoRoute-BR Candidate] Starting Task {task_id} ---")
        cumulative_classes = TASK_CUMULATIVE_CLASSES[task_id]
        required_classes = len(cumulative_classes)

        if model.num_classes < required_classes:
            model.expand_classes(required_classes)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_loader = get_task_loader(task_id=task_id, split="train", cumulative=False, shuffle=True)
        train_embs, train_lbls, _ = get_task_data(task_id=task_id, split="train", cumulative=False)

        train_single_task(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            replay_buffer=replay_buffer if (task_id > 1) else None,
            ewc=ewc if (task_id > 1) else None,
            replay_strategy=replay_strategy if (task_id > 1) else "original",
            device=device
        )

        replay_buffer.add_examples(train_embs, train_lbls)
        replay_buffer.log_memory_composition(task_id=task_id)

        task_eval_loader = get_task_loader(task_id=task_id, split="train", cumulative=False, shuffle=False)
        ewc.compute_fisher(task_eval_loader, num_samples=500)

        # Save task checkpoint
        task_ckpt = MODELS_DIR / f"{save_name}_task{task_id}.pt"
        torch.save(model.state_dict(), task_ckpt)

        # Evaluate on validation split
        acc_dict = evaluate_task_accuracies(model, current_task=task_id, split="val", device=device)
        for t_eval in range(1, task_id + 1):
            R_val[task_id - 1, t_eval - 1] = acc_dict[t_eval]

        val_eval = evaluate_model_on_classes(model, split="val", allowed_classes=cumulative_classes, device=device)
        logger.info(f"Candidate Task {task_id} Val Accuracy: {val_eval['accuracy']:.4f} (Macro F1: {val_eval['macro_f1']:.4f})")
        task_history.append({
            "task_id": task_id,
            "val_accuracy": val_eval["accuracy"],
            "val_balanced_accuracy": val_eval["balanced_accuracy"],
            "val_macro_f1": val_eval["macro_f1"],
            "per_class_accuracy": val_eval["per_class_accuracy"],
            "memory_size": replay_buffer.total_samples
        })

    ft_info = None
    if run_finetuning:
        model, ft_info = run_balanced_finetuning(
            model=model,
            replay_buffer=replay_buffer,
            lr=1e-4,
            epochs=10,
            patience=2,
            device=device
        )

    # Final validation evaluation
    final_val_diag = validate_checkpoint(model, split="val", device=device)
    val_forgetting = compute_forgetting(R_val)

    save_path = MODELS_DIR / f"{save_name}_final.pt"
    # Never overwrite baseline
    assert str(save_path.resolve()) != str(BASELINE_CHECKPOINT_PATH.resolve()), "Baseline checkpoint cannot be overwritten!"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved EvoRoute-BR candidate model to {save_path}")

    return {
        "candidate_name": save_name,
        "replay_strategy": replay_strategy,
        "finetuning_applied": run_finetuning,
        "validation_metrics": final_val_diag,
        "validation_forgetting": val_forgetting,
        "task_history": task_history,
        "finetuning_details": ft_info,
        "model_path": str(save_path),
        "lexicographic_tuple": final_val_diag["lexicographic_tuple"]
    }


def run_validation_ablation_study(device: str = DEVICE) -> Dict[str, Any]:
    """
    Executes controlled validation-only ablation study across candidate configurations.
    Compares:
      1. Baseline Replay + EWC (loaded from immutable checkpoint)
      2. Candidate A: 50/50 Replay Mixing (no fine-tuning)
      3. Candidate B: Class-Balanced Replay (no fine-tuning)
      4. Candidate C: Class-Balanced Replay + Post-Task Balanced Fine-Tuning (Full Candidate)
    
    Strict Invariant:
      Evaluates strictly on the validation split. Zero test set exposure.
      Enforces anti-collapse constraints:
        - Newest class accuracy >= 0.60
        - Old class accuracy >= 0.40
    """
    set_seed(RANDOM_SEED)
    logger.info("================ STARTING VALIDATION ABLATION STUDY ================")
    
    # 1. Baseline Replay + EWC
    baseline_model = EvoMLP(num_classes=4).to(device)
    assert BASELINE_CHECKPOINT_PATH.exists(), f"Baseline checkpoint missing at {BASELINE_CHECKPOINT_PATH}!"
    baseline_model.load_state_dict(torch.load(BASELINE_CHECKPOINT_PATH, map_location=device, weights_only=True))
    baseline_val = validate_checkpoint(baseline_model, split="val", device=device)
    logger.info(f"Baseline Replay + EWC Val: Macro F1={baseline_val['macro_f1']:.4f}, BalAcc={baseline_val['balanced_accuracy']:.4f}, Acc={baseline_val['overall_accuracy']:.4f}, NewestAcc={baseline_val['recency_metrics']['newest_class_accuracy']:.4f}, OldAcc={baseline_val['recency_metrics']['old_class_accuracy']:.4f}")

    # 2. Candidate A: 50/50 Replay
    cand_a = train_evoroute_br_candidate(
        replay_strategy="50_50",
        run_finetuning=False,
        save_name="candidate_50_50",
        device=device
    )

    # 3. Candidate B: Class-Balanced Replay (no FT)
    cand_b = train_evoroute_br_candidate(
        replay_strategy="class_balanced",
        run_finetuning=False,
        save_name="candidate_class_balanced",
        device=device
    )

    # 4. Candidate C: Class-Balanced Replay + Balanced Fine-Tuning
    cand_c = train_evoroute_br_candidate(
        replay_strategy="class_balanced",
        run_finetuning=True,
        save_name="evoroute_br_candidate",
        device=device
    )

    all_candidates = {
        "baseline_replay_ewc": {
            "name": "Replay + EWC (Baseline)",
            "replay_strategy": "original",
            "finetuning": False,
            "metrics": baseline_val,
            "model_path": str(BASELINE_CHECKPOINT_PATH),
            "lexicographic_tuple": baseline_val["lexicographic_tuple"]
        },
        "candidate_50_50": {
            "name": "EvoRoute-BR (50/50 Replay)",
            "replay_strategy": "50_50",
            "finetuning": False,
            "metrics": cand_a["validation_metrics"],
            "model_path": cand_a["model_path"],
            "lexicographic_tuple": cand_a["lexicographic_tuple"]
        },
        "candidate_class_balanced": {
            "name": "EvoRoute-BR (Class-Balanced Replay)",
            "replay_strategy": "class_balanced",
            "finetuning": False,
            "metrics": cand_b["validation_metrics"],
            "model_path": cand_b["model_path"],
            "lexicographic_tuple": cand_b["lexicographic_tuple"]
        },
        "evoroute_br_candidate": {
            "name": "EvoRoute-BR (Class-Balanced + Fine-Tuned)",
            "replay_strategy": "class_balanced",
            "finetuning": True,
            "metrics": cand_c["validation_metrics"],
            "model_path": cand_c["model_path"],
            "lexicographic_tuple": cand_c["lexicographic_tuple"]
        }
    }

    # Rank candidates by validation lexicographic tuple: (Macro F1, Balanced Acc, Overall Acc)
    # Filter by anti-collapse constraints
    valid_candidates = {}
    for key, c_data in all_candidates.items():
        m = c_data["metrics"]
        rm = m.get("recency_metrics", {})
        newest_acc = rm.get("newest_class_accuracy", 0.0)
        old_acc = rm.get("old_class_accuracy", 0.0)
        if newest_acc >= MIN_NEWEST_CLASS_ACCURACY and old_acc >= MIN_OLD_CLASS_ACCURACY:
            valid_candidates[key] = c_data
        else:
            logger.warning(f"Candidate {key} failed anti-collapse constraint (Newest Acc: {newest_acc:.4f}, Old Acc: {old_acc:.4f})")

    # If all candidates collapsed, fallback to baseline
    if not valid_candidates:
        valid_candidates["baseline_replay_ewc"] = all_candidates["baseline_replay_ewc"]

    # Select best candidate
    best_key = max(valid_candidates.keys(), key=lambda k: valid_candidates[k]["lexicographic_tuple"])
    best_candidate_info = valid_candidates[best_key]

    logger.info(f"\n================ VALIDATION SELECTION RESULT ================")
    logger.info(f"  Selected Winner: {best_candidate_info['name']} ({best_key})")
    logger.info(f"  Val Macro F1:    {best_candidate_info['metrics']['macro_f1']:.4f}")
    logger.info(f"  Val Balanced Acc: {best_candidate_info['metrics']['balanced_accuracy']:.4f}")
    logger.info(f"  Val Overall Acc: {best_candidate_info['metrics']['overall_accuracy']:.4f}")
    logger.info(f"============================================================")

    # Run post-hoc calibration on the winning candidate strictly on validation split
    winner_model = EvoMLP(num_classes=4).to(device)
    winner_model.load_state_dict(torch.load(best_candidate_info["model_path"], map_location=device, weights_only=True))
    
    baseline_newest = baseline_val["recency_metrics"]["newest_class_accuracy"]
    calibrated_wrapper, cal_report = run_post_hoc_calibration(
        winner_model,
        device=device,
        baseline_newest_acc=baseline_newest
    )

    # Save calibrated candidate model state
    cal_save_path = EVOROUTE_BR_CALIBRATED_CHECKPOINT_PATH
    torch.save({
        "base_model_state": winner_model.state_dict(),
        "calibration_state": cal_report["calibration_state"],
        "calibration_report": cal_report
    }, cal_save_path)
    logger.info(f"Saved calibrated candidate model to {cal_save_path}")

    # Evaluate calibrated wrapper on validation split
    with torch.no_grad():
        val_embs, val_lbls, _ = get_task_data(task_id=3, split="val", cumulative=True)
        val_logits = calibrated_wrapper(val_embs.to(device))
        val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        cal_val_metrics = compute_dataset_aware_recency_bias(val_lbls.numpy(), val_preds)

    all_candidates["evoroute_br_calibrated"] = {
        "name": f"{best_candidate_info['name']} + Calibrated",
        "replay_strategy": best_candidate_info["replay_strategy"],
        "finetuning": best_candidate_info["finetuning"],
        "calibrated": True,
        "calibration_report": cal_report,
        "metrics": cal_val_metrics,
        "model_path": str(cal_save_path),
        "lexicographic_tuple": (cal_val_metrics["macro_f1"], cal_val_metrics["balanced_accuracy"], cal_val_metrics["overall_accuracy"])
    }

    ablation_summary = {
        "ablation_study_split": "val",
        "selection_metric": "Lexicographic(Macro F1, Balanced Accuracy, Overall Accuracy)",
        "metric_tolerance": METRIC_TOLERANCE,
        "anti_collapse_thresholds": {
            "min_newest_class_accuracy": MIN_NEWEST_CLASS_ACCURACY,
            "min_old_class_accuracy": MIN_OLD_CLASS_ACCURACY,
            "max_forgetting_increase": MAX_FORGETTING_INCREASE
        },
        "baseline_val_metrics": baseline_val,
        "candidates": all_candidates,
        "selected_winner_uncalibrated": best_key,
        "selected_winner_uncalibrated_name": best_candidate_info["name"],
        "calibrated_winner": "evoroute_br_calibrated",
        "calibration_applied": not cal_report["decision_rebalancing"]["rejection_triggered"],
        "calibration_report": cal_report
    }

    # Save validation ablation results
    study_path = RESULTS_METRICS_DIR / "validation_ablation_study.json"
    with open(study_path, "w") as f:
        json.dump(ablation_summary, f, indent=2, default=str)
    logger.info(f"Saved validation ablation study to {study_path}")

    return ablation_summary


def train_joint_upper_bound(epochs: int = EPOCHS_PER_TASK, device: str = DEVICE) -> Dict[str, Any]:
    """
    Trains the classifier on all 4 classes jointly.
    Acts as the non-continual upper-bound baseline.
    """
    set_seed(RANDOM_SEED)
    logger.info("================ STARTING JOINT TRAINING (UPPER BOUND) ================")
    model = EvoMLP(num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    joint_loader = get_joint_loader(split="train", shuffle=True)
    train_single_task(
        model=model,
        train_loader=joint_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device
    )

    overall_eval = evaluate_model_on_classes(model, split="test", allowed_classes=[0, 1, 2, 3], device=device)
    
    # Compute per-task accuracy on test set
    task_accs = [
        evaluate_model_on_classes(model, split="test", allowed_classes=TASK_CLASSES[t], device=device)["accuracy"]
        for t in [1, 2, 3]
    ]
    final_avg_task_acc = float(np.mean(task_accs))

    # Compute Empirical Prediction Distribution and Recency Bias
    pred_dist_data = compute_prediction_distribution(model, split="test", device=device)

    logger.info(f"Joint Training Complete -> Overall Accuracy: {overall_eval['accuracy']:.4f}, Avg Task Acc: {final_avg_task_acc:.4f}, Recency Bias: {pred_dist_data.get('recency_bias', 0.0):+.4f}")

    save_path = MODELS_DIR / "joint_final.pt"
    torch.save(model.state_dict(), save_path)

    return {
        "method": "joint",
        "overall_accuracy": overall_eval["accuracy"],
        "final_avg_task_accuracy": final_avg_task_acc,
        "final_average_accuracy": overall_eval["accuracy"],
        "final_accuracy": overall_eval["accuracy"],
        "average_forgetting": 0.0,
        "final_per_class": overall_eval["per_class_accuracy"],
        "prediction_distribution": pred_dist_data.get("distribution", {}),
        "prediction_distribution_percentages": pred_dist_data.get("percentages", {}),
        "recency_bias": pred_dist_data.get("recency_bias", 0.0),
        "prediction_distribution_details": pred_dist_data,
        "task_history": [
            {"task_id": 1, "overall_accuracy": overall_eval["accuracy"]},
            {"task_id": 2, "overall_accuracy": overall_eval["accuracy"]},
            {"task_id": 3, "overall_accuracy": overall_eval["accuracy"]}
        ],
        "forgetting": {"average_forgetting": 0.0, "per_task_forgetting": {"Task_1": 0.0, "Task_2": 0.0}},
        "memory_size": "Full Dataset",
        "model_path": str(save_path)
    }
