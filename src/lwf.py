import copy
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LWF_TEMPERATURE, LWF_LAMBDA, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def freeze_teacher(teacher_model: nn.Module):
    """
    Sets teacher model to eval mode and disables gradients for all parameters.
    Ensures the teacher model never receives gradient updates.
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False


def create_teacher_model(model: nn.Module) -> nn.Module:
    """
    Creates a deep copy of the model to serve as a frozen teacher for knowledge distillation.
    """
    teacher = copy.deepcopy(model)
    freeze_teacher(teacher)
    return teacher


def compute_lwf_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_old_classes: int,
    temperature: float = LWF_TEMPERATURE
) -> torch.Tensor:
    """
    Computes temperature-scaled Knowledge Distillation loss (KL divergence)
    restricted strictly to previously learned output classes.

    Args:
        student_logits: Logits from current student model (N, C_new)
        teacher_logits: Logits from frozen teacher model (N, C_old)
        num_old_classes: Number of classes known by the teacher model
        temperature: Distillation temperature T (scales logits before softmax)

    Returns:
        Scalar KL divergence loss scaled by T^2
    """
    assert teacher_logits.shape[1] == num_old_classes, (
        f"Expected teacher logits to have {num_old_classes} columns, got {teacher_logits.shape[1]}"
    )
    assert student_logits.shape[1] >= num_old_classes, (
        f"Expected student logits to have at least {num_old_classes} columns, got {student_logits.shape[1]}"
    )
    if student_logits.shape[1] > num_old_classes:
        assert student_logits.shape[1] > teacher_logits.shape[1], (
            f"Student classes ({student_logits.shape[1]}) must exceed teacher classes ({teacher_logits.shape[1]}) on new task"
        )

    # Distill ONLY over previous class outputs
    teacher_old_logits = teacher_logits[:, :num_old_classes]
    student_old_logits = student_logits[:, :num_old_classes]

    # Temperature-scaled soft probabilities
    teacher_probs = F.softmax(teacher_old_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_old_logits / temperature, dim=1)

    # KL Divergence with batchmean reduction
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Rescale by temperature^2 to match gradient magnitude
    return kd_loss * (temperature ** 2)


class LwF:
    """
    Learning without Forgetting (LwF) continual learning manager.
    Maintains a frozen teacher snapshot of the previous task model and
    applies temperature-scaled distillation to retain historical representations.
    Exemplar-free: stores 0 past training embeddings.
    """
    def __init__(
        self,
        temperature: float = LWF_TEMPERATURE,
        lwf_lambda: float = LWF_LAMBDA,
        device: str = DEVICE
    ):
        self.temperature = temperature
        self.lwf_lambda = lwf_lambda
        self.device = device
        self.teacher: Optional[nn.Module] = None
        self.num_old_classes: int = 0

    @property
    def has_teacher(self) -> bool:
        """Returns True if a frozen teacher model exists for distillation."""
        return self.teacher is not None and self.num_old_classes > 0

    def set_teacher(self, model: nn.Module):
        """
        Snapshots the student model as the frozen teacher after completing a task.
        """
        self.teacher = create_teacher_model(model).to(self.device)
        self.num_old_classes = model.num_classes
        logger.info(f"LwF Teacher snapshot updated: {self.num_old_classes} classes frozen.")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes inputs through the frozen teacher and computes distillation loss against student logits.
        """
        if not self.has_teacher:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        return compute_lwf_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            num_old_classes=self.num_old_classes,
            temperature=self.temperature
        )
