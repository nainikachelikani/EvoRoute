import logging
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from src.config import REPLAY_MEMORY_BUDGET, REPLAY_SAMPLE_RATIO, RANDOM_SEED, ID2CATEGORY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Memory-efficient Experience Replay buffer strictly bounded by a maximum memory budget.
    Stores precomputed 384-D embeddings and integer labels.
    
    Guarantees:
    1. Memory size never exceeds max_budget (default 200).
    2. Capacity is partitioned evenly across all seen classes.
    3. Old exemplars are rebalanced and downsampled when new classes arrive.
    4. Test split data is never added to memory.
    """
    def __init__(self, max_budget: int = REPLAY_MEMORY_BUDGET, sample_ratio: float = REPLAY_SAMPLE_RATIO):
        self.max_budget = max_budget
        self.sample_ratio = sample_ratio
        # Class-indexed storage: {class_id: {"embeddings": torch.Tensor, "labels": torch.Tensor}}
        self.memory: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def total_samples(self) -> int:
        """Current total number of stored exemplars."""
        return sum(len(store["labels"]) for store in self.memory.values())

    @property
    def seen_classes(self) -> List[int]:
        """List of classes currently represented in memory."""
        return sorted(list(self.memory.keys()))

    def log_memory_composition(self, task_id: int = None):
        """Logs exact per-class memory allocation and verifies budget constraint."""
        prefix = f"Task {task_id} " if task_id is not None else ""
        logger.info(f"========== {prefix}Memory Composition (Budget: {self.max_budget}) ==========")
        for c in self.seen_classes:
            cat_name = ID2CATEGORY.get(c, f"Class_{c}")
            count = len(self.memory[c]["labels"])
            logger.info(f"  {cat_name}: {count} exemplars")
        logger.info(f"  Total Exemplars: {self.total_samples} / {self.max_budget}")
        logger.info("================================================================")
        assert self.total_samples <= self.max_budget, f"FATAL: Buffer size {self.total_samples} exceeded budget {self.max_budget}!"

    def rebalance_memory(self):
        """
        Rebalances exemplar allocation evenly across all seen classes so total <= max_budget.
        If C classes are present, each class gets floor(max_budget / C) exemplars.
        """
        num_classes = len(self.memory)
        if num_classes == 0:
            return

        per_class_budget = self.max_budget // num_classes
        logger.info(f"Rebalancing ReplayBuffer: {num_classes} classes, target {per_class_budget} exemplars each (max budget: {self.max_budget}).")

        for class_id in list(self.memory.keys()):
            store = self.memory[class_id]
            current_count = len(store["labels"])
            if current_count > per_class_budget:
                # Downsample exemplars: keep exemplars closest to class centroid
                embs = store["embeddings"]
                norm_embs = F.normalize(embs, p=2, dim=1)
                centroid = norm_embs.mean(dim=0, keepdim=True)
                dists = 1.0 - torch.matmul(norm_embs, centroid.T).squeeze(1)
                
                # Select exemplars with smallest distance to centroid for representative coverage
                _, keep_indices = torch.topk(dists, k=per_class_budget, largest=False)
                self.memory[class_id]["embeddings"] = embs[keep_indices]
                self.memory[class_id]["labels"] = store["labels"][keep_indices]

        assert self.total_samples <= self.max_budget, f"Buffer size {self.total_samples} exceeded budget {self.max_budget}!"
        logger.info(f"Rebalancing complete. Total exemplars in memory: {self.total_samples}")

    def add_examples(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Adds candidate exemplars for newly encountered classes and immediately rebalances memory.
        """
        unique_classes = torch.unique(labels).tolist()
        num_classes_total = len(set(self.seen_classes + unique_classes))
        per_class_budget = max(1, self.max_budget // num_classes_total)

        for c in unique_classes:
            mask = (labels == c)
            class_embs = embeddings[mask]
            class_lbls = labels[mask]

            # Select the most representative exemplars (closest to class centroid)
            norm_embs = F.normalize(class_embs, p=2, dim=1)
            centroid = norm_embs.mean(dim=0, keepdim=True)
            dists = 1.0 - torch.matmul(norm_embs, centroid.T).squeeze(1)

            k = min(len(class_embs), per_class_budget)
            _, selected_indices = torch.topk(dists, k=k, largest=False)

            self.memory[c] = {
                "embeddings": class_embs[selected_indices].cpu(),
                "labels": class_lbls[selected_indices].cpu()
            }

        self.rebalance_memory()

    def sample_per_class(self, class_id: int, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples num_samples exemplars specifically for class_id.
        """
        if class_id not in self.memory or len(self.memory[class_id]["labels"]) == 0 or num_samples <= 0:
            return torch.empty((0, 384)), torch.empty(0, dtype=torch.long)

        store = self.memory[class_id]
        embs = store["embeddings"]
        lbls = store["labels"]
        total_avail = len(lbls)
        replace = (num_samples > total_avail)
        indices = np.random.choice(total_avail, size=num_samples, replace=replace)
        return embs[indices], lbls[indices]

    def sample_class_balanced(
        self,
        num_samples: int,
        allowed_classes: List[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples exemplars evenly partitioned across available or allowed classes.
        Handles remainder deterministically across classes in sorted order.
        """
        target_classes = self.seen_classes if allowed_classes is None else [c for c in allowed_classes if c in self.seen_classes]
        if not target_classes or num_samples <= 0:
            return torch.empty((0, 384)), torch.empty(0, dtype=torch.long)

        k = len(target_classes)
        base_count = num_samples // k
        remainder = num_samples % k

        sampled_embs = []
        sampled_lbls = []

        for idx, c in enumerate(target_classes):
            c_count = base_count + (1 if idx < remainder else 0)
            if c_count > 0:
                e, l = self.sample_per_class(c, c_count)
                sampled_embs.append(e)
                sampled_lbls.append(l)

        if not sampled_embs:
            return torch.empty((0, 384)), torch.empty(0, dtype=torch.long)

        return torch.cat(sampled_embs, dim=0), torch.cat(sampled_lbls, dim=0)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uniformly samples a batch of exemplars across all stored classes.
        Returns empty tensors if buffer is empty.
        """
        if self.total_samples == 0:
            return torch.empty((0, 384)), torch.empty(0, dtype=torch.long)

        all_embs = torch.cat([store["embeddings"] for store in self.memory.values()], dim=0)
        all_lbls = torch.cat([store["labels"] for store in self.memory.values()], dim=0)

        total_avail = len(all_lbls)
        # Sample with replacement if requested samples exceeds available
        replace = (num_samples > total_avail)
        indices = np.random.choice(total_avail, size=num_samples, replace=replace)

        return all_embs[indices], all_lbls[indices]

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns all exemplars currently in memory."""
        if self.total_samples == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        all_embs = torch.cat([store["embeddings"] for store in self.memory.values()], dim=0)
        all_lbls = torch.cat([store["labels"] for store in self.memory.values()], dim=0)
        return all_embs, all_lbls


def construct_incremental_batch(
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    replay_buffer: ReplayBuffer = None,
    strategy: str = "original",
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs an incremental training batch adhering strictly to the specified replay strategy.

    Strategies:
      - "original": Exact baseline behavior. Adds int(inputs.size(0) * sample_ratio) replay exemplars.
                    Batch size can expand up to 76 (64 new + 12 replay).
      - "50_50": Fixed 50% new task data (batch_size // 2) and 50% replay buffer exemplars (batch_size // 2).
                 Total batch size strictly bounded at batch_size (64).
      - "class_balanced": Dynamically partitions batch_size evenly across ALL seen classes (current + historical).
                          Every seen category receives equal representation in every gradient update.
                          Total batch size strictly bounded at batch_size (64).
    """
    if replay_buffer is None or replay_buffer.total_samples == 0:
        return current_inputs[:batch_size].to(device), current_targets[:batch_size].to(device)

    if strategy == "original":
        # Exact Baseline Replay + EWC mixing
        n_replay = max(1, int(current_inputs.size(0) * replay_buffer.sample_ratio))
        replay_inputs, replay_targets = replay_buffer.sample(n_replay)
        if len(replay_inputs) > 0:
            inputs = torch.cat([current_inputs.to(device), replay_inputs.to(device)], dim=0)
            targets = torch.cat([current_targets.to(device), replay_targets.to(device)], dim=0)
            return inputs, targets
        return current_inputs.to(device), current_targets.to(device)

    elif strategy == "50_50":
        n_replay = batch_size // 2
        n_current = batch_size - n_replay
        replay_inputs, replay_targets = replay_buffer.sample(n_replay)
        curr_in = current_inputs[:n_current].to(device)
        curr_tg = current_targets[:n_current].to(device)
        if len(replay_inputs) > 0:
            inputs = torch.cat([curr_in, replay_inputs.to(device)], dim=0)
            targets = torch.cat([curr_tg, replay_targets.to(device)], dim=0)
            return inputs, targets
        return curr_in, curr_tg

    elif strategy == "class_balanced":
        curr_classes = sorted(torch.unique(current_targets).tolist())
        old_classes = sorted([c for c in replay_buffer.seen_classes if c not in curr_classes])
        all_classes = sorted(list(set(old_classes + curr_classes)))
        num_classes = len(all_classes)

        base_count = batch_size // num_classes
        remainder = batch_size % num_classes

        batch_inputs = []
        batch_targets = []

        for idx, c in enumerate(all_classes):
            needed = base_count + (1 if idx < remainder else 0)
            if needed <= 0:
                continue

            if c in curr_classes:
                # Sample from current task batch
                mask = (current_targets == c)
                c_in = current_inputs[mask]
                c_tg = current_targets[mask]
                avail = len(c_in)
                if avail > 0:
                    replace = (needed > avail)
                    sel_idx = np.random.choice(avail, size=needed, replace=replace)
                    batch_inputs.append(c_in[sel_idx].to(device))
                    batch_targets.append(c_tg[sel_idx].to(device))
            else:
                # Sample from replay buffer
                e, l = replay_buffer.sample_per_class(c, needed)
                if len(e) > 0:
                    batch_inputs.append(e.to(device))
                    batch_targets.append(l.to(device))

        if batch_inputs:
            inputs = torch.cat(batch_inputs, dim=0)
            targets = torch.cat(batch_targets, dim=0)
            return inputs, targets

        return current_inputs[:batch_size].to(device), current_targets[:batch_size].to(device)

    else:
        raise ValueError(f"Unknown replay mixing strategy: {strategy}")
