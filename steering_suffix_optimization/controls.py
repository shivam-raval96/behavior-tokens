"""Causal direct-steering positive controls for a candidate vector/sign."""
from __future__ import annotations

from contextlib import contextmanager
import torch


@contextmanager
def add_residual_direction(model, module_index: int, direction: torch.Tensor, scale: float):
    """Add a direction to the selected decoder-block output during generation."""
    vector=(scale*direction/direction.norm()).to(next(model.parameters()).device)
    def hook(_module,_inputs,output):
        hidden=output[0] if isinstance(output,tuple) else output
        steered=hidden+vector.to(hidden.dtype)
        return (steered,*output[1:]) if isinstance(output,tuple) else steered
    handle=model.model.layers[module_index].register_forward_hook(hook)
    try: yield
    finally: handle.remove()


def hidden_state_index_to_module_index(hidden_state_index: int) -> int:
    if hidden_state_index<1: raise ValueError("embedding hidden state has no decoder module")
    return hidden_state_index-1
