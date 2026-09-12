import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    RAW_CSV_PATH,
    CLEANED_CSV_PATH,
    CATEGORIES,
    CATEGORY2ID,
    MAX_SAMPLES_PER_CLASS,
    MIN_TEXT_LENGTH,
    SPLIT_RATIOS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def auto_detect_columns(df: pd.DataFrame):
    """
    Inspects DataFrame to dynamically identify category and text columns.
    Checks column values against known target categories.
    """
    category_col = None
    text_col = None

    target_cats_lower = {c.lower(): c for c in CATEGORIES}

    # Inspect each column
    for col in df.columns:
        # Check if values match target categories
        sample_vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
        matches = sum(1 for val in sample_vals if val in target_cats_lower)
        if matches >= 2:
            category_col = col
            break

    # Text column is typically the other column with longest mean string length
    remaining_cols = [c for c in df.columns if c != category_col]
    if remaining_cols:
        lengths = {
            col: df[col].dropna().astype(str).str.len().mean()
            for col in remaining_cols
        }
        text_col = max(lengths, key=lengths.get)
    else:
        raise ValueError("Could not find suitable text column in dataset.")

    if category_col is None:
        raise ValueError("Could not identify category column matching target categories.")

    logger.info(f"Auto-detected columns -> Category: '{category_col}', Text: '{text_col}'")
    return category_col, text_col


def clean_dataset(raw_csv_path: Path = RAW_CSV_PATH, max_per_class: int = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Loads raw CSV, auto-detects columns, cleans nulls, removes duplicates,
    filters short descriptions, normalizes categories, and subsamples if requested.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    logger.info(f"Loading raw dataset from {raw_csv_path}...")
    try:
        # First try reading with header
        df_test = pd.read_csv(raw_csv_path, nrows=5)
        # If first row looks like data rather than headers (e.g. Household in col 0)
        first_val = str(df_test.iloc[0, 0]).strip().lower()
        target_cats_lower = [c.lower() for c in CATEGORIES]
        
        has_header = not any(first_val == cat for cat in target_cats_lower)
        if has_header and any(str(c).lower() in ["category", "label", "target"] for c in df_test.columns):
            df = pd.read_csv(raw_csv_path)
        else:
            df = pd.read_csv(raw_csv_path, header=None)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

    logger.info(f"Raw shape: {df.shape}")
    cat_col, text_col = auto_detect_columns(df)

    # Extract and rename columns
    df_clean = pd.DataFrame({
        "category": df[cat_col].astype(str).str.strip(),
        "description": df[text_col].astype(str).str.strip()
    })

    # Drop nulls / empty
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    df_clean = df_clean[(df_clean["category"] != "") & (df_clean["description"] != "")]
    logger.info(f"Dropped {initial_count - len(df_clean)} null or empty records.")

    # Standardize category strings
    cat_mapping = {c.lower(): c for c in CATEGORIES}
    df_clean["category_norm"] = df_clean["category"].str.lower().map(cat_mapping)
    df_clean = df_clean.dropna(subset=["category_norm"])
    df_clean["category"] = df_clean["category_norm"]
    df_clean = df_clean.drop(columns=["category_norm"])

    # Remove duplicates
    prev_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["description"])
    logger.info(f"Removed {prev_len - len(df_clean)} duplicate descriptions. Remaining: {len(df_clean)}")

    # Remove short descriptions
    prev_len = len(df_clean)
    df_clean = df_clean[df_clean["description"].str.len() >= MIN_TEXT_LENGTH]
    logger.info(f"Removed {prev_len - len(df_clean)} descriptions shorter than {MIN_TEXT_LENGTH} chars.")

    # Assign integer labels
    df_clean["label"] = df_clean["category"].map(CATEGORY2ID)

    # Subsample per class if requested
    if max_per_class is not None and max_per_class > 0:
        logger.info(f"Subsampling up to {max_per_class} records per category for hackathon speed (seed={RANDOM_SEED})...")
        sampled_dfs = []
        for cat in CATEGORIES:
            cat_subset = df_clean[df_clean["category"] == cat]
            n_samples = min(len(cat_subset), max_per_class)
            sampled = cat_subset.sample(n=n_samples, random_state=RANDOM_SEED)
            sampled_dfs.append(sampled)
        df_clean = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle
    df_clean = df_clean.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info("Category distribution in cleaned dataset:")
    for cat, count in df_clean["category"].value_counts().items():
        logger.info(f"  {cat}: {count}")

    # Save to data/processed/cleaned_data.csv
    CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)
    logger.info(f"Cleaned dataset saved to {CLEANED_CSV_PATH} (Total: {len(df_clean)} rows)")

    return df_clean


def split_data(df: pd.DataFrame):
    """
    Stratified split into Train (80%), Val (10%), Test (10%).
    Ensures exact reproducibility via random_state = 42.
    """
    logger.info("Splitting dataset into Train (80%), Val (10%), Test (10%)...")
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    # Second split: 50% of temp into val (10% total), 50% into test (10% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    df_clean = clean_dataset()
    train_df, val_df, test_df = split_data(df_clean)
    print("Preprocessing completed successfully.")
