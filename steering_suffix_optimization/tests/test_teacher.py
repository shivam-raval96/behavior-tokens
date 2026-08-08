from types import SimpleNamespace

import torch

from steering_suffix_optimization.teacher import steering_hook


class Layer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([Layer()]))


def test_response_hook_preserves_prompt_and_covers_prediction_positions():
    model = Toy()
    hidden = torch.zeros(1, 6, 2)
    with steering_hook(model, 0, torch.tensor([1.0, 0.0]), 2.0, "plain", 2):
        result = model.model.layers[0](hidden)
    assert torch.equal(result[0, :2], torch.zeros(2, 2))
    assert torch.equal(result[0, 2:], torch.tensor([[2.0, 0.0]]).repeat(4, 1))
