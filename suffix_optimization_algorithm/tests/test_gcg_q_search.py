import torch

from suffix_optimization_algorithm.gcg_q_search import continuation_losses, position_decomposition


def test_continuation_slice_accounts_for_suffix_and_prediction_offset():
    logits = torch.full((1, 5, 4), -20.0)
    logits[0, 2, 1] = 20.0
    logits[0, 3, 2] = 20.0
    losses = continuation_losses(logits, 2, 1, torch.tensor([1, 2]))
    assert losses.shape == (1, 2)
    assert float(losses.max()) < 1e-6


def test_position_decomposition_is_record_mean_then_token_weighted_tail():
    result = position_decomposition([[10.0, 1.0, 2.0], [20.0, 3.0]])
    assert result == {"position_1_ce": 15.0, "tail_ce": 2.0}
