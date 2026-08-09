from types import SimpleNamespace

import torch

from suffix_optimization_algorithm.gcg_q_search import Search, continuation_losses, position_decomposition


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


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 5)
        self.output = torch.nn.Linear(5, 11, bias=False)

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.output(inputs_embeds))


def test_rollout_batched_candidate_losses_match_serial_reference():
    torch.manual_seed(3)
    model = TinyModel()
    cache = [
        {"prefix_ids": [1, 2], "continuation_ids": [3, 4, 5]},
        {"prefix_ids": [2, 1, 3], "continuation_ids": [6, 7]},
        {"prefix_ids": [4], "continuation_ids": [8, 9, 2, 1]},
    ]
    config = {"evaluation_microbatch_size": 2, "evaluation_record_batch_size": 2}
    search = Search(model, None, cache, [], config, torch.zeros(5))
    candidates = torch.tensor([[1, 2], [2, 3], [4, 5]])
    serial = search.candidate_losses_serial(candidates)
    batched = search.candidate_losses(candidates)
    torch.testing.assert_close(batched, serial, atol=1e-6, rtol=1e-6)
