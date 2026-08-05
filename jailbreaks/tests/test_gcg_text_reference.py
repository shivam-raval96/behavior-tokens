import torch

from jailbreaks.gcg_text_reference import filter_text_candidates


class ToyTokenizer:
    """Makes reference filtering semantics observable without a model download."""

    def decode(self, ids, skip_special_tokens=True):
        return {1: "old", 2: "valid", 3: "short", 4: "valid"}[ids[0]]

    def __call__(self, text, add_special_tokens=False):
        mapping = {"old": [7, 8], "valid": [9, 10], "short": [11]}
        return type("Encoded", (), {"input_ids": mapping[text]})()


def test_reference_filter_uses_retokenized_length_not_id_equality():
    candidates = torch.tensor([[1, 8], [2, 8], [3, 8], [4, 8]])
    filtered = filter_text_candidates(ToyTokenizer(), candidates, "old", expected_length=2)
    assert filtered == ["valid", "valid", "valid", "valid"]
