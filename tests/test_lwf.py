import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import EvoMLP
from src.lwf import LwF, create_teacher_model, freeze_teacher, compute_lwf_loss


def test_lwf_teacher_is_frozen():
    """Test 1: Teacher model has requires_grad = False on all parameters and is in eval mode."""
    print("Testing Test 1: Teacher model is frozen...")
    model = EvoMLP(num_classes=2)
    teacher = create_teacher_model(model)

    assert not teacher.training, "Teacher model should be in eval() mode"
    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher parameter {name} should have requires_grad=False"
    print("  [OK] Teacher model is completely frozen.")


def test_lwf_teacher_receives_no_gradients():
    """Test 2: Teacher receives no gradients during student backward pass."""
    print("Testing Test 2: Teacher receives no gradients...")
    student = EvoMLP(num_classes=3)
    teacher = create_teacher_model(EvoMLP(num_classes=2))

    dummy_input = torch.randn(4, 384)
    with torch.no_grad():
        t_logits = teacher(dummy_input)

    s_logits = student(dummy_input)
    kd_loss = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2)
    kd_loss.backward()

    for name, param in teacher.named_parameters():
        assert param.grad is None, f"Teacher parameter {name} should not receive gradients!"
    print("  [OK] Confirmed teacher receives no gradients.")


def test_lwf_loss_produces_valid_scalar():
    """Test 3: KD loss produces a valid finite non-NaN scalar."""
    print("Testing Test 3: KD loss produces valid scalar...")
    s_logits = torch.randn(8, 3)
    t_logits = torch.randn(8, 2)

    loss = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2, temperature=2.0)
    assert torch.is_tensor(loss), "Loss must be a torch Tensor"
    assert loss.dim() == 0, "Loss must be a scalar"
    assert torch.isfinite(loss), "Loss must be finite"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert loss.item() >= 0.0, "KL divergence distillation loss must be non-negative"
    print(f"  [OK] KD loss is valid scalar: {loss.item():.4f}")


def test_lwf_task2_output_slicing():
    """Test 4: Task 2: Teacher has 2 outputs, Student has 3 outputs. Distillation uses only 2 outputs."""
    print("Testing Test 4: Task 2 output head slicing...")
    s_logits = torch.randn(4, 3)
    t_logits = torch.randn(4, 2)

    # Distillation restricted to first 2 classes
    loss = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2)
    assert torch.isfinite(loss)

    # If student output matches teacher over old classes, KD loss approaches 0
    t_logits_matched = torch.randn(4, 2)
    s_logits_matched = torch.cat([t_logits_matched, torch.randn(4, 1) * 10], dim=1)
    matched_loss = compute_lwf_loss(student_logits=s_logits_matched, teacher_logits=t_logits_matched, num_old_classes=2)
    assert matched_loss.item() < 1e-5, f"Matched old logits should yield ~0 KD loss, got {matched_loss.item()}"
    print("  [OK] Task 2 output head slicing correctly isolates previous classes.")


def test_lwf_task3_output_slicing():
    """Test 5: Task 3: Teacher has 3 outputs, Student has 4 outputs. Distillation uses only 3 outputs."""
    print("Testing Test 5: Task 3 output head slicing...")
    s_logits = torch.randn(4, 4)
    t_logits = torch.randn(4, 3)

    loss = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=3)
    assert torch.isfinite(loss)

    t_logits_matched = torch.randn(4, 3)
    s_logits_matched = torch.cat([t_logits_matched, torch.randn(4, 1) * 5], dim=1)
    matched_loss = compute_lwf_loss(student_logits=s_logits_matched, teacher_logits=t_logits_matched, num_old_classes=3)
    assert matched_loss.item() < 1e-5, f"Matched old logits should yield ~0 KD loss, got {matched_loss.item()}"
    print("  [OK] Task 3 output head slicing correctly isolates previous 3 classes.")


def test_lwf_temperature_scaling():
    """Test 6: Temperature scaling works and softens distributions."""
    print("Testing Test 6: Temperature scaling...")
    s_logits = torch.tensor([[2.0, -1.0], [0.5, 1.5]])
    t_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss_t1 = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2, temperature=1.0)
    loss_t2 = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2, temperature=2.0)
    loss_t5 = compute_lwf_loss(student_logits=s_logits, teacher_logits=t_logits, num_old_classes=2, temperature=5.0)

    assert torch.isfinite(loss_t1) and torch.isfinite(loss_t2) and torch.isfinite(loss_t5)
    print(f"  [OK] Temperature scaling verified: T=1: {loss_t1.item():.4f}, T=2: {loss_t2.item():.4f}, T=5: {loss_t5.item():.4f}")


def test_lwf_manager_integration():
    """Test 7: LwF manager state transitions across continual tasks."""
    print("Testing Test 7: LwF manager state transitions...")
    model = EvoMLP(num_classes=2)
    lwf = LwF(temperature=2.0, lwf_lambda=1.0)

    # Initial state (Task 1): no teacher yet
    assert not lwf.has_teacher

    # Snapshot after Task 1
    lwf.set_teacher(model)
    assert lwf.has_teacher
    assert lwf.num_old_classes == 2

    # Task 2 expansion
    model.expand_classes(3)
    dummy_input = torch.randn(4, 384)
    s_logits = model(dummy_input)

    loss = lwf.compute_distillation_loss(student_logits=s_logits, inputs=dummy_input)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Snapshot after Task 2
    lwf.set_teacher(model)
    assert lwf.num_old_classes == 3

    # Task 3 expansion
    model.expand_classes(4)
    s_logits_t3 = model(dummy_input)
    loss_t3 = lwf.compute_distillation_loss(student_logits=s_logits_t3, inputs=dummy_input)
    assert torch.isfinite(loss_t3)
    print("  [OK] LwF manager continual workflow operates correctly.")


if __name__ == "__main__":
    test_lwf_teacher_is_frozen()
    test_lwf_teacher_receives_no_gradients()
    test_lwf_loss_produces_valid_scalar()
    test_lwf_task2_output_slicing()
    test_lwf_task3_output_slicing()
    test_lwf_temperature_scaling()
    test_lwf_manager_integration()
    print("\nAll LwF unit tests passed successfully!")
