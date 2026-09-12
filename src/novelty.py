import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from src.config import (
    CATEGORY2ID,
    ID2CATEGORY,
    NOVELTY_CALIBRATION_PERCENTILE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NoveltyDetector:
    """
    Centroid-based Cosine Distance Novelty Detector.
    
    Lifecycle:
    1. Known categories are registered.
    2. Class centroids are computed from normalized semantic embeddings.
    3. Distance threshold is calibrated dynamically on validation data.
    4. Unfamiliar incoming products with distance > threshold are flagged as NOVEL.
    """
    def __init__(self, percentile: float = NOVELTY_CALIBRATION_PERCENTILE):
        self.percentile = percentile
        self.known_classes: List[int] = []
        self.centroids: Dict[int, torch.Tensor] = {}
        self.threshold: float = 0.5

    def update_known_classes(
        self,
        new_classes: List[int],
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None
    ):
        """
        Updates registered known classes, recomputes normalized class centroids,
        and recalibrates the novelty threshold.
        """
        for c in new_classes:
            if c not in self.known_classes:
                self.known_classes.append(c)

        logger.info(f"NoveltyDetector updating: currently known classes -> {[ID2CATEGORY[c] for c in self.known_classes]}")
        self.compute_centroids(embeddings, labels)

        if val_embeddings is not None and val_labels is not None:
            self.calibrate_threshold(val_embeddings, val_labels)

    def compute_centroids(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Computes and normalizes class centroids for all currently known classes."""
        embeddings = F.normalize(embeddings, p=2, dim=1)
        for c in self.known_classes:
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_embs = embeddings[mask]
            # Mean embedding normalized to unit hypersphere
            mean_emb = class_embs.mean(dim=0, keepdim=True)
            norm_centroid = F.normalize(mean_emb, p=2, dim=1)
            self.centroids[c] = norm_centroid.squeeze(0)
        logger.info(f"Computed centroids for {len(self.centroids)} known classes.")

    def compute_distances(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cosine distance from each embedding to the closest known centroid.
        Returns:
            min_distances: Tensor of shape (N,) with cosine distances in [0, 2]
            nearest_classes: Tensor of shape (N,) with the ID of the nearest known category
        """
        if not self.centroids:
            raise ValueError("No centroids available. Call update_known_classes first.")

        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroid_keys = list(self.centroids.keys())
        stacked_centroids = torch.stack([self.centroids[c] for c in centroid_keys], dim=0) # (C, D)

        # Cosine similarity matrix: (N, C)
        sim_matrix = torch.matmul(embeddings, stacked_centroids.T)
        # Cosine distance = 1 - cosine similarity
        dist_matrix = 1.0 - sim_matrix

        min_distances, min_indices = torch.min(dist_matrix, dim=1)
        nearest_classes = torch.tensor([centroid_keys[idx.item()] for idx in min_indices], dtype=torch.long)

        return min_distances, nearest_classes

    def calibrate_threshold(self, val_embeddings: torch.Tensor, val_labels: torch.Tensor):
        """
        Calibrates the novelty threshold tau on validation data.
        tau is set to the configured percentile of cosine distances of known validation examples.
        Any sample having minimum distance > tau will be flagged as novel/unfamiliar.
        """
        # Filter validation embeddings to known classes only
        known_mask = torch.isin(val_labels, torch.tensor(self.known_classes))
        if known_mask.sum() == 0:
            logger.warning("No known classes found in validation set for calibration. Keeping default threshold.")
            return

        known_val_embs = val_embeddings[known_mask]
        min_dists, _ = self.compute_distances(known_val_embs)
        calibrated_tau = float(np.percentile(min_dists.numpy(), self.percentile))
        self.threshold = calibrated_tau
        logger.info(f"Calibrated novelty threshold tau = {self.threshold:.4f} (at {self.percentile}th percentile of known validation samples).")

    def detect(self, embedding: torch.Tensor) -> Dict[str, Union[float, str, bool]]:
        """
        Evaluates a single embedding for novelty.
        Returns:
            dict containing distance, threshold, is_novel, nearest_category, confidence
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        min_dist, nearest_cls = self.compute_distances(embedding)
        dist = float(min_dist[0].item())
        nearest_id = int(nearest_cls[0].item())
        nearest_name = ID2CATEGORY.get(nearest_id, f"Class {nearest_id}")
        is_novel = bool(dist > self.threshold)

        return {
            "distance": dist,
            "threshold": self.threshold,
            "is_novel": is_novel,
            "nearest_category_id": nearest_id,
            "nearest_category_name": nearest_name,
            "decision": "UNKNOWN / NOVEL CATEGORY" if is_novel else "KNOWN CATEGORY"
        }

    def get_all_distances(self, embedding: torch.Tensor) -> Dict[str, float]:
        """Returns distance from an embedding to every known class centroid."""
        if not self.centroids:
            return {}
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        norm_e = F.normalize(embedding, p=2, dim=1)
        distances = {}
        for c, cent in self.centroids.items():
            dist = float(1.0 - torch.sum(norm_e * cent).item())
            distances[ID2CATEGORY.get(c, f"Class {c}")] = dist
        return distances

    def evaluate_novelty(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluates precision, recall, and F1 score for novelty detection on a mixed dataset.
        Ground truth: is_novel = (label not in known_classes).
        """
        min_dists, _ = self.compute_distances(embeddings)
        pred_novel = (min_dists > self.threshold).numpy().astype(int)
        true_novel = (~np.isin(labels.numpy(), self.known_classes)).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_novel, pred_novel, average="binary", zero_division=0
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_samples": len(labels),
            "novel_samples": int(true_novel.sum()),
            "detected_novel": int(pred_novel.sum()),
            "threshold": self.threshold
        }
