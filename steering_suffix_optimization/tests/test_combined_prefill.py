import pytest
import torch

from steering_suffix_optimization.combined_prefill import (
    combined_objective,
    full_forward_kl,
)


def test_forward_kl_zero_for_equal_logits():
    logits = torch.randn(3, 7)
    assert torch.allclose(full_forward_kl(logits, logits), torch.zeros(3), atol=1e-6)


def test_combined_objective_zero_at_teacher_target():
    baseline = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    logits = torch.randn(2, 5)
    total, parts = combined_objective(
        target,
        logits,
        target,
        logits,
        baseline,
        torch.tensor([1.0, 0.0, 0.0]),
        {
            "normalized_activation_weight": 1.0,
            "normalized_projection_weight": 1.0,
            "position_weighted_kl_weight": 0.1,
            "position_0_fraction": 0.5,
        },
    )
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert all(float(value) == pytest.approx(0.0, abs=1e-6) for value in parts.values())


def test_position_zero_receives_configured_half_weight():
    baseline = torch.zeros(2, 2)
    target_hidden = torch.ones(2, 2)
    target_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    student_logits = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    _, parts = combined_objective(
        target_hidden,
        student_logits,
        target_hidden,
        target_logits,
        baseline,
        torch.tensor([1.0, 0.0]),
        {
            "normalized_activation_weight": 0.0,
            "normalized_projection_weight": 0.0,
            "position_weighted_kl_weight": 1.0,
            "position_0_fraction": 0.5,
        },
    )
    assert float(parts["later_position_kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(parts["position_weighted_kl"]) == pytest.approx(
        float(parts["position_0_kl"]) * 0.5
    )
