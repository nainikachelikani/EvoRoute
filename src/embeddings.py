import os
import logging
from pathlib import Path
from typing import Dict, List, Union
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    TRAIN_EMB_PATH,
    VAL_EMB_PATH,
    TEST_EMB_PATH,
    CLEANED_CSV_PATH
)
from src.data import clean_dataset, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cached model instance
_EMBEDDING_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    """Loads and caches the frozen SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on {DEVICE}...")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        # Freeze weights
        for param in _EMBEDDING_MODEL.parameters():
            param.requires_grad = False
        _EMBEDDING_MODEL.eval()
    return _EMBEDDING_MODEL


def encode_texts(texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE, show_progress: bool = True) -> torch.Tensor:
    """Encodes a list of product descriptions into normalized 384-D PyTorch tensors."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=DEVICE
    )
    return embeddings.cpu()


def generate_and_save_embeddings(force_recompute: bool = False):
    """
    Checks if train, val, and test embeddings exist. If not or if forced,
    generates embeddings from cleaned_data.csv and saves them to data/processed/.
    """
    if not force_recompute and TRAIN_EMB_PATH.exists() and VAL_EMB_PATH.exists() and TEST_EMB_PATH.exists():
        logger.info("Cached embeddings found in data/processed/. Skipping generation.")
        return

    logger.info("Preparing data splits for embedding generation...")
    if not CLEANED_CSV_PATH.exists():
        df_clean = clean_dataset()
    else:
        df_clean = pd.read_csv(CLEANED_CSV_PATH)

    train_df, val_df, test_df = split_data(df_clean)

    splits = {
        "train": (train_df, TRAIN_EMB_PATH),
        "val": (val_df, VAL_EMB_PATH),
        "test": (test_df, TEST_EMB_PATH)
    }

    for split_name, (df, save_path) in splits.items():
        logger.info(f"Generating embeddings for '{split_name}' set ({len(df)} samples)...")
        texts = df["description"].tolist()
        labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
        categories = df["category"].tolist()

        embeddings = encode_texts(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress=True)

        data_dict = {
            "embeddings": embeddings,
            "labels": labels,
            "categories": categories,
            "texts": texts
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_dict, save_path)
        logger.info(f"Saved {split_name} embeddings to {save_path} (shape: {embeddings.shape})")


def load_embeddings(split: str = "train") -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Loads cached embeddings dictionary for 'train', 'val', or 'test'."""
    split_paths = {
        "train": TRAIN_EMB_PATH,
        "val": VAL_EMB_PATH,
        "test": TEST_EMB_PATH
    }
    if split not in split_paths:
        raise ValueError(f"Unknown split '{split}'. Must be one of: {list(split_paths.keys())}")

    path = split_paths[split]
    if not path.exists():
        logger.info(f"Embeddings file {path} not found. Generating now...")
        generate_and_save_embeddings()

    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    generate_and_save_embeddings(force_recompute=True)
    sample = load_embeddings("train")
    print(f"Verified train embeddings shape: {sample['embeddings'].shape}")
