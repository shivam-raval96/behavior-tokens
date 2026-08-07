from __future__ import annotations

from contextlib import contextmanager

import torch


def transformer_layers(model):
    for path in (("model", "layers"), ("transformer", "h")):
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return value
    raise TypeError("unsupported causal-LM layer structure")


@contextmanager
def steering_hook(
    model,
    layer_index: int,
    vector: torch.Tensor,
    alpha: float,
    mode: str,
    start_position: int,
):
    unit = vector / vector.norm().clamp_min(1e-12)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden.clone()
        segment = steered[:, start_position:, :]
        scale = (
            segment.float().norm(dim=-1, keepdim=True)
            if mode == "activation_norm"
            else 1.0
        )
        segment.add_(alpha * scale * unit.to(segment.device, segment.dtype))
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = transformer_layers(model)[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
