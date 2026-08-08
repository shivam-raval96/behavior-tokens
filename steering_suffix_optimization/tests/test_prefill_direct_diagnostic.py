import torch

from steering_suffix_optimization.layout import PromptLayout
from steering_suffix_optimization.prefill_direct_diagnostic import (
    aggregate,
    response_only_hook,
    trajectory_metrics,
)


def test_direct_hook_uses_response_boundary_not_user_boundary():
    layout = PromptLayout(torch.tensor([1, 2, 3]), torch.tensor([9]), torch.tensor([4, 5]), steer_start=1)
    hook = response_only_hook({"module_index": 9, "direct_additive_norm": -3.0}, torch.ones(2), layout)
    assert hook[4] == 4
    assert hook[4] != layout.steer_start


def test_trajectory_metrics_identical_target():
    baseline = torch.zeros(4, 3)
    direct = torch.ones(4, 3)
    soft = direct.clone()
    logits = torch.randn(4, 7)
    metrics = trajectory_metrics(baseline, soft, direct, logits, logits, torch.tensor([1., 0., 0.]))
    assert abs(metrics["trajectory_activation_cosine"] - 1) < 1e-6
    assert metrics["mean_activation_mse"] == 0
    assert abs(metrics["mean_forward_kl"]) < 1e-6


def test_aggregate_three_conditions():
    row = {"metrics": {key: 1.0 for key in (
        "mean_activation_cosine", "trajectory_activation_cosine", "mean_activation_mse",
        "mean_soft_projection", "mean_direct_projection", "mean_forward_kl", "position_0_kl",
        "positions_1_8_kl", "positions_9_32_kl", "positions_33_plus_kl")},
        "judgments": {"baseline": {"success": False}, "direct": {"success": True}, "soft_prefill": {"success": False}}}
    result = aggregate([row])
    assert result["direct_asr"] == 1
    assert result["baseline_asr"] == result["soft_prefill_asr"] == 0
