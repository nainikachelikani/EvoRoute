import logging
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import EWC_LAMBDA, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation for continual learning.
    
    Computes the empirical Fisher Information Matrix diagonal to quantify parameter
    importance on previous tasks. Applies a quadratic penalty during future tasks
    to safeguard critical weights against catastrophic forgetting.
    """
    def __init__(self, model: nn.Module, ewc_lambda: float = EWC_LAMBDA, device: str = DEVICE):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        
        # Stored reference parameters and Fisher diagonal for each past task
        # Dict[str, torch.Tensor]
        self.reference_params: Dict[str, torch.Tensor] = {}
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_fisher(self, dataloader: DataLoader, num_samples: int = 1000):
        """
        Estimates the empirical Fisher Information Matrix diagonal using squared gradients
        of log-likelihood on the completed task.
        """
        self.model.eval()
        fisher_accum: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data)

        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            inputs = batch["embedding"].to(self.device)
            targets = batch["label"].to(self.device)

            for i in range(len(targets)):
                if samples_seen >= num_samples:
                    break

                self.model.zero_grad()
                logits = self.model(inputs[i:i+1])
                log_probs = F.log_softmax(logits, dim=1)
                nll = F.nll_loss(log_probs, targets[i:i+1])
                nll.backward()

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_accum[name] += (param.grad.data ** 2)

                samples_seen += 1

        self.model.zero_grad()
        samples_seen = max(1, samples_seen)
        for name in fisher_accum:
            fisher_accum[name] /= samples_seen

        # Consolidate into cumulative Fisher matrix and record current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.fisher_matrix:
                    # If parameter shapes match (e.g. feature extractor), accumulate Fisher
                    if self.fisher_matrix[name].shape == param.shape:
                        self.fisher_matrix[name] += fisher_accum[name]
                        self.reference_params[name] = param.data.clone().to(self.device)
                    else:
                        # Output head may have expanded; copy old part and keep new part
                        old_shape = self.fisher_matrix[name].shape
                        new_fisher = fisher_accum[name].clone()
                        new_fisher[:old_shape[0]] += self.fisher_matrix[name]
                        self.fisher_matrix[name] = new_fisher
                        self.reference_params[name] = param.data.clone().to(self.device)
                else:
                    self.fisher_matrix[name] = fisher_accum[name]
                    self.reference_params[name] = param.data.clone().to(self.device)

        self.task_count += 1
        logger.info(f"EWC Fisher Information Matrix estimated and stored for task {self.task_count} (processed {samples_seen} samples).")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Calculates the quadratic EWC penalty:
        loss_penalty = (lambda / 2) * sum_i F_i * (theta_i - theta_i^*)^2
        """
        if not self.fisher_matrix:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in self.fisher_matrix and name in self.reference_params:
                fisher = self.fisher_matrix[name].to(self.device)
                ref_param = self.reference_params[name].to(self.device)

                # Handle dynamic shape expansions in head weights/biases
                if param.shape == ref_param.shape:
                    diff = param - ref_param
                    loss += (fisher * (diff ** 2)).sum()
                else:
                    # Compare only the overlapping rows corresponding to previously learned classes
                    min_rows = min(param.shape[0], ref_param.shape[0])
                    diff = param[:min_rows] - ref_param[:min_rows]
                    fish = fisher[:min_rows]
                    loss += (fish * (diff ** 2)).sum()

        return (self.ewc_lambda / 2.0) * loss
