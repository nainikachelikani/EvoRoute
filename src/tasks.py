import logging
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    BATCH_SIZE,
    TASK_DEFINITIONS,
    TASK_CLASSES,
    TASK_CUMULATIVE_CLASSES,
    CATEGORY2ID,
    ID2CATEGORY
)
from src.embeddings import load_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for precomputed embeddings and labels."""
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, texts: List[str] = None):
        self.embeddings = embeddings
        self.labels = labels
        self.texts = texts if texts is not None else [""] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "text": self.texts[idx]
        }


def get_task_data(task_id: int, split: str = "train", cumulative: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Retrieves embeddings, labels, and texts for a specific task.
    If cumulative=True, returns all classes seen up to and including task_id.
    Otherwise returns only the new classes introduced in task_id.
    """
    raw_data = load_embeddings(split)
    all_embeddings = raw_data["embeddings"]
    all_labels = raw_data["labels"]
    all_texts = raw_data["texts"]

    if cumulative:
        target_classes = TASK_CUMULATIVE_CLASSES[task_id]
    else:
        target_classes = TASK_CLASSES[task_id]

    mask = torch.isin(all_labels, torch.tensor(target_classes))
    task_embeddings = all_embeddings[mask]
    task_labels = all_labels[mask]
    task_texts = [all_texts[i] for i in range(len(all_labels)) if mask[i].item()]

    return task_embeddings, task_labels, task_texts


def get_task_loader(
    task_id: int,
    split: str = "train",
    cumulative: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True
) -> DataLoader:
    """Creates a PyTorch DataLoader for a specific task and split."""
    embeddings, labels, texts = get_task_data(task_id, split=split, cumulative=cumulative)
    dataset = EmbeddingDataset(embeddings, labels, texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_joint_loader(split: str = "train", batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader containing all 4 categories for joint training upper bound."""
    raw_data = load_embeddings(split)
    dataset = EmbeddingDataset(raw_data["embeddings"], raw_data["labels"], raw_data["texts"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
