import random
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
    RANDOM_SEED
)
from src.model import EvoMLP
from src.tasks import get_task_data, get_task_loader, get_joint_loader
from src.replay import ReplayBuffer
from src.ewc import EWC
from src.lwf import LwF
from src.evaluate import (
    evaluate_model_on_classes,
    evaluate_task_accuracies,
    compute_forgetting,
    compute_final_average_task_accuracy
)

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
    device: str = DEVICE
):
    """
    Trains model on a single task.
    Optionally mixes replay exemplars, adds EWC quadratic penalty, or adds LwF distillation loss.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs = batch["embedding"].to(device)
            targets = batch["label"].to(device)

            # Mix with replay buffer if active
            if replay_buffer is not None and replay_buffer.total_samples > 0:
                n_replay = max(1, int(inputs.size(0) * replay_buffer.sample_ratio))
                replay_inputs, replay_targets = replay_buffer.sample(n_replay)
                if len(replay_inputs) > 0:
                    inputs = torch.cat([inputs, replay_inputs.to(device)], dim=0)
                    targets = torch.cat([targets, replay_targets.to(device)], dim=0)

            optimizer.zero_grad()
            logits = model(inputs)
            cls_loss = criterion(logits, targets)

            # Add continual learning regularization loss
            if ewc is not None:
                penalty_loss = ewc.penalty(model)
                total_loss = cls_loss + penalty_loss
            elif lwf is not None and lwf.has_teacher:
                kd_loss = lwf.compute_distillation_loss(student_logits=logits, inputs=inputs)
                total_loss = cls_loss + lwf.lwf_lambda * kd_loss
            else:
                total_loss = cls_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            logger.info(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


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

    logger.info(f"\n==================== [Method: {method.upper()} FINAL SUMMARY] ====================")
    logger.info(f"  PRIMARY METRIC 1 (Overall Class Accuracy):      {overall_accuracy * 100:.2f}%")
    logger.info(f"  PRIMARY METRIC 2 (Final Average Task Accuracy): {final_avg_task_acc * 100:.2f}%")
    logger.info(f"  PRIMARY METRIC 3 (Final Average Forgetting):    {avg_forgetting * 100:.2f}%")
    logger.info(f"==================================================================================")

    # Save final model
    save_path = MODELS_DIR / f"{method}_final.pt"
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
        "task_history": task_history,
        "accuracy_matrix": R.tolist(),
        "memory_size": replay_buffer.total_samples if replay_buffer is not None else 0,
        "model_path": str(save_path)
    }


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

    logger.info(f"Joint Training Complete -> Overall Accuracy: {overall_eval['accuracy']:.4f}, Avg Task Acc: {final_avg_task_acc:.4f}")

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
        "task_history": [
            {"task_id": 1, "overall_accuracy": overall_eval["accuracy"]},
            {"task_id": 2, "overall_accuracy": overall_eval["accuracy"]},
            {"task_id": 3, "overall_accuracy": overall_eval["accuracy"]}
        ],
        "forgetting": {"average_forgetting": 0.0, "per_task_forgetting": {"Task_1": 0.0, "Task_2": 0.0}},
        "memory_size": "Full Dataset",
        "model_path": str(save_path)
    }
