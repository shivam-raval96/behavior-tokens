"""GCG-style coordinate search against a residual-stream steering target."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ActivationTarget:
    layer: int
    vector: torch.Tensor
    scale: float = 1.0

    def normalized(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = self.vector.to(device=device, dtype=dtype)
        return self.scale * F.normalize(value, dim=-1)


def direction_metrics(delta: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Metrics for the mean suffix-induced activation delta."""
    mean = delta.mean(0)
    unit = F.normalize(target, dim=-1)
    return {
        "cosine": float(F.cosine_similarity(mean.float(), target.float(), dim=0)),
        "projection": float(mean.float() @ unit.float()),
        "delta_norm": float(mean.float().norm()),
        "target_norm": float(target.float().norm()),
        "mse": float(F.mse_loss(mean.float(), target.float())),
    }


class SteeringSuffixOptimizer:
    """HotFlip/GCG search maximizing a suffix-induced activation displacement.

    The loss matches `h(prompt + suffix) - h(prompt)` to a supplied steering
    vector. Candidate tokens are proposed from the gradient with respect to the
    suffix embeddings and selected using exact forward evaluations.
    """

    def __init__(self, model, tokenizer, target: ActivationTarget, *, top_k: int = 64,
                 candidate_batch_size: int = 128, eval_chunk_size: int = 16,
                 norm_weight: float = 0.1):
        self.model, self.tokenizer, self.target = model, tokenizer, target
        self.top_k, self.candidate_batch_size = top_k, candidate_batch_size
        self.eval_chunk_size, self.norm_weight = eval_chunk_size, norm_weight
        self.device = next(model.parameters()).device
        self.embedding = model.get_input_embeddings()
        self.special_ids = set(tokenizer.all_special_ids)

    def _hidden(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.model(inputs_embeds=embeds, attention_mask=mask,
                         output_hidden_states=True, use_cache=False)
        return out.hidden_states[self.target.layer][:, -1, :]

    @torch.no_grad()
    def baseline(self, prompt_ids: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [self._hidden(self.embedding(ids[None].to(self.device)),
                             torch.ones(1, len(ids), device=self.device, dtype=torch.long))[0]
                for ids in prompt_ids]

    def _loss(self, suffix_embeds: torch.Tensor, prompt_ids: Sequence[torch.Tensor],
              baselines: Sequence[torch.Tensor]) -> torch.Tensor:
        losses = []
        for ids, base in zip(prompt_ids, baselines):
            prefix = self.embedding(ids[None].to(self.device)).detach()
            joined = torch.cat((prefix, suffix_embeds), dim=1)
            hidden = self._hidden(joined, torch.ones(joined.shape[:2], device=self.device, dtype=torch.long))
            delta = hidden[0] - base
            target = self.target.normalized(delta.device, delta.dtype)
            cosine = F.cosine_similarity(delta, target, dim=0)
            radial = (delta.norm() - target.norm()).square() / target.numel()
            losses.append(1 - cosine + self.norm_weight * radial)
        return torch.stack(losses).mean()

    def token_gradient(self, suffix: torch.Tensor, prompts: Sequence[torch.Tensor],
                       baselines: Sequence[torch.Tensor]) -> tuple[torch.Tensor, float]:
        embeds = self.embedding(suffix.to(self.device))[None].detach().requires_grad_(True)
        loss = self._loss(embeds, prompts, baselines)
        loss.backward()
        # dL/d(one_hot) = dL/d(embed) @ embedding_matrix.T
        grad = embeds.grad[0].float() @ self.embedding.weight.detach().float().T
        return grad, float(loss.detach())

    def candidates(self, suffix: torch.Tensor, grad: torch.Tensor,
                   generator: torch.Generator) -> torch.Tensor:
        scores = -grad
        if self.special_ids:
            scores[:, list(self.special_ids)] = -torch.inf
        choices = scores.topk(min(self.top_k, scores.shape[1]), dim=1).indices
        rows = suffix.to(self.device).repeat(self.candidate_batch_size, 1)
        positions = torch.arange(self.candidate_batch_size, device=self.device) % suffix.numel()
        ranks = torch.randint(choices.shape[1], (self.candidate_batch_size,), generator=generator,
                              device=self.device)
        rows[torch.arange(rows.shape[0], device=self.device), positions] = choices[positions, ranks]
        return rows

    @torch.no_grad()
    def evaluate(self, candidates: torch.Tensor, prompts: Sequence[torch.Tensor],
                 baselines: Sequence[torch.Tensor]) -> torch.Tensor:
        values = []
        for start in range(0, len(candidates), self.eval_chunk_size):
            for row in candidates[start:start + self.eval_chunk_size]:
                values.append(self._loss(self.embedding(row)[None], prompts, baselines))
        return torch.stack(values)

    def step(self, suffix: torch.Tensor, prompts: Sequence[torch.Tensor],
             baselines: Sequence[torch.Tensor], generator: torch.Generator):
        grad, current_loss = self.token_gradient(suffix, prompts, baselines)
        candidates = self.candidates(suffix, grad, generator)
        losses = self.evaluate(candidates, prompts, baselines)
        index = int(losses.argmin())
        if float(losses[index]) < current_loss:
            return candidates[index].detach(), float(losses[index]), True
        return suffix.detach(), current_loss, False
