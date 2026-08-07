import torch

from steering_suffix_optimization.optimizer import ActivationTarget, direction_metrics


def test_target_is_normalized_and_scaled():
    target = ActivationTarget(3, torch.tensor([3.0, 4.0]), scale=2.0)
    assert torch.allclose(target.normalized(torch.device("cpu"), torch.float32),
                          torch.tensor([1.2, 1.6]))


def test_direction_metrics_detect_alignment():
    metrics = direction_metrics(torch.tensor([[2.0, 0.0], [4.0, 0.0]]), torch.tensor([1.0, 0.0]))
    assert metrics["cosine"] == 1.0
    assert metrics["projection"] == 3.0
    assert metrics["delta_norm"] == 3.0
