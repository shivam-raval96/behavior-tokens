import pytest
import torch

from steering_suffix_optimization.trajectory_imitation import (
    aggregate_losses,
    condition_gate_passes,
    condition_name,
    projection_retention,
    trajectory_loss,
)


def test_trajectory_loss_is_zero_for_exact_teacher_match():
    teacher_hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    baseline_hidden = torch.zeros_like(teacher_hidden)
    logits = torch.zeros(2, 3)
    loss, parts = trajectory_loss(
        teacher_hidden,
        logits,
        teacher_hidden,
        logits,
        baseline_hidden,
        {"normalized_mse": 0.5, "cosine_distance": 0.3, "forward_kl": 0.2},
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(parts["hidden_cosine"]) == pytest.approx(1.0)


def test_projection_retention_reports_fraction_of_teacher_effect():
    baseline = torch.zeros(2, 2)
    teacher = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    student = teacher * 0.5
    retention = projection_retention(
        student, teacher, baseline, torch.tensor([1.0, 0.0])
    )
    assert retention.tolist() == pytest.approx([0.5, 0.5])


def test_aggregate_losses_and_condition_name():
    assert aggregate_losses([{"loss": 1.0}, {"loss": 3.0}]) == {"loss": 2.0}
    assert condition_name(32) == "horizon_32"


def test_success_gate_requires_same_condition_to_pass_both_components():
    evaluations = [
        {"horizon": 8, "summary": {"projection_retention": {"16": 0.6}}},
        {"horizon": 32, "summary": {"projection_retention": {"16": 0.4}}},
    ]
    conditions = {
        "horizon_1": {"successes": 0},
        "horizon_8": {"successes": 1},
        "horizon_32": {"successes": 2},
    }
    result = condition_gate_passes(
        evaluations,
        conditions,
        [1, 8, 32],
        {"min_position_16_retention": 0.5, "min_success_gain": 2},
    )
    assert result == {"horizon_8": False, "horizon_32": False}
