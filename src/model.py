import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT_RATE, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EvoMLP(nn.Module):
    """
    Lightweight PyTorch MLP Classifier supporting dynamic incremental class expansion.
    
    Architecture:
    384-D Input -> Linear(256) -> ReLU -> Dropout(0.2) -> Linear(128) -> ReLU -> Dropout(0.2) -> Output Layer
    """
    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dims: List[int] = HIDDEN_DIMS, num_classes: int = 2, dropout_rate: float = DROPOUT_RATE):
        super(EvoMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Shared feature extractor layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        # Dynamic output classification head
        self.head = nn.Linear(hidden_dims[1], num_classes)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Kaiming normal initialization."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate layer 128-dimensional representations."""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and current classification head."""
        feats = self.extract_features(x)
        logits = self.head(feats)
        return logits

    def expand_classes(self, new_num_classes: int):
        """
        Dynamically expands the output layer from C_old to C_new classes.
        Existing learned weights and biases for previously known classes are strictly preserved.
        New class weights are initialized without disrupting previous representations.
        """
        if new_num_classes <= self.num_classes:
            return

        old_num_classes = self.num_classes
        logger.info(f"Expanding output classification layer: {old_num_classes} -> {new_num_classes} classes.")

        old_head = self.head
        new_head = nn.Linear(self.hidden_dims[-1], new_num_classes).to(old_head.weight.device)

        # Initialize new head
        nn.init.xavier_normal_(new_head.weight)
        nn.init.constant_(new_head.bias, 0.0)

        # Copy over weights and biases for existing classes exactly
        with torch.no_grad():
            new_head.weight.data[:old_num_classes, :] = old_head.weight.data.clone()
            new_head.bias.data[:old_num_classes] = old_head.bias.data.clone()

        # Rigorous verification: assert old weights and biases are strictly identical
        assert torch.allclose(new_head.weight.data[:old_num_classes, :], old_head.weight.data), (
            "Weight preservation assertion failed! Old weights were corrupted during dynamic expansion."
        )
        assert torch.allclose(new_head.bias.data[:old_num_classes], old_head.bias.data), (
            "Bias preservation assertion failed! Old biases were corrupted during dynamic expansion."
        )

        self.head = new_head
        self.num_classes = new_num_classes
        logger.info(f"Dynamic expansion complete. Preserved {old_num_classes} classes with strict weight invariance verified.")
