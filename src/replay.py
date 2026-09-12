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
