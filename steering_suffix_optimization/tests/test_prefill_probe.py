import pytest
import torch

from steering_suffix_optimization.layout import PromptLayout, splice_embeddings
from steering_suffix_optimization.prefill_probe import diagnostic, position_metrics


def test_response_logit_start_is_last_clean_prompt_position():
    layout = PromptLayout(
        prefix_ids=torch.tensor([1, 2, 3]),
        suffix_marker_ids=torch.tensor([9, 9]),
        assistant_header_ids=torch.tensor([4, 5]),
        steer_start=1,
    )
    assert layout.response_logit_start == 4
    assert layout.response_logit_start > layout.steer_start


def test_fp32_master_suffix_is_cast_and_receives_gradient():
    embedding = torch.nn.Embedding(20, 4, dtype=torch.bfloat16)
    layout = PromptLayout(torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4]))
    suffix = torch.nn.Parameter(torch.randn(1, 4, dtype=torch.float32))
    inputs, _ = splice_embeddings(embedding, layout, suffix, torch.tensor([5, 6]))
    assert inputs.dtype == torch.bfloat16
    inputs.float().sum().backward()
    assert suffix.grad is not None and suffix.grad.dtype == torch.float32


def test_position_metrics_cover_32_positions():
    result = position_metrics(torch.arange(32, dtype=torch.float32))
    assert result == {
        "position_0_kl": 0.0,
        "positions_1_8_kl": 4.5,
        "positions_9_32_kl": 20.0,
    }


def test_perfect_trajectory_has_unit_cosine_and_zero_mse():
    baseline = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    logits = torch.zeros(2, 5)
    result = diagnostic(
        target, logits, target, logits, baseline, torch.tensor([1.0, 0.0, 0.0])
    )
    assert result["activation_mse"] == 0.0
    assert result["activation_cosine"] == pytest.approx(1.0)
    assert result["mean_forward_kl"] == 0.0
