import torch

from suffix_optimization_algorithm.fixed_rollout_ce import (
    continuation_nll,
    splice_suffix,
)


def test_splice_suffix_is_explicit_and_stable():
    assert splice_suffix("behavior", "") == "behavior"
    assert splice_suffix("behavior", "abc") == "behavior\n\nabc"


def test_continuation_nll_masks_prefix_and_scores_first_target():
    # Prefix ids [4, 5], continuation [1, 2]. The prediction logits for labels
    # 1 and 2 live at positions 1 and 2 respectively.
    logits = torch.zeros(1, 4, 6)
    logits[0, 1, 1] = 8
    logits[0, 2, 2] = 8
    nll_sum, count = continuation_nll(logits, [4, 5, 1, 2], prefix_length=2)
    assert count == 2
    assert float(nll_sum) < 0.01


def test_continuation_nll_rejects_empty_target():
    logits = torch.zeros(1, 1, 3)
    try:
        continuation_nll(logits, [1], prefix_length=1)
    except ValueError as error:
        assert "continuation" in str(error)
    else:
        raise AssertionError("expected ValueError")
