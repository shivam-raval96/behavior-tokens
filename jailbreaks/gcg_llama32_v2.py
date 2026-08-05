"""Token-native building blocks for the Llama-3.2 GCG v2 path.

Unlike the legacy-compatible path, candidate validity is defined by the token
IDs actually inserted into the model prompt. Decoding a suffix is retained as
an artifact only; it is never used to decide whether a candidate is valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PhaseBest:
    """Best suffix for one fixed active-behavior objective."""

    active_goals: int
    loss: float = float("inf")
    suffix: torch.Tensor | None = None

    def consider(self, suffix: torch.Tensor, loss: float) -> bool:
        if loss >= self.loss:
            return False
        self.loss, self.suffix = loss, suffix.detach().clone()
        return True


def validate_token_native_candidates(candidates: torch.Tensor, current: torch.Tensor,
                                     special_ids: set[int]) -> None:
    """Require exactly one non-special token mutation, without text round-trips."""
    if candidates.ndim != 2 or candidates.shape[1] != current.numel():
        raise AssertionError("candidate shape does not match the control token sequence")
    changed = (candidates != current.unsqueeze(0)).sum(dim=1)
    if not torch.all(changed == 1):
        raise AssertionError("every candidate must change exactly one control token")
    if any(int(token) in special_ids for token in candidates.flatten()):
        raise AssertionError("candidate includes a tokenizer special token")
