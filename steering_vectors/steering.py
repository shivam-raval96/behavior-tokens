"""Contrastive steering-vector construction.

Given collected pooled activations for positive (concept) and negative
conversations, the steering vector is the difference of class means
(a.k.a. CAA / diff-of-means):

    v = mean(A_pos) - mean(A_neg)

optionally unit-normalized. The pooling method (mean/last/attention) is
applied upstream in `SteeringModel.collect_activation`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from steering_vectors.config import Config
from steering_vectors.data import Conversation
from steering_vectors.model import SteeringModel


@dataclass
class SteeringVector:
    vector: torch.Tensor       # [hidden]
    layer: int
    pooling: str
    concept: str
    raw_norm: float            # norm of the diff-of-means before normalization
    n_pos: int
    n_neg: int


def collect_activations(model: SteeringModel, convs: list[Conversation],
                        layer: int, pooling: str, batch_size: int = 16,
                        progress=None, desc=None) -> torch.Tensor:
    """Stack pooled activations for a list of conversations -> [N, hidden]."""
    pairs = [(c.prompt, c.response) for c in convs]
    return model.collect_activations_batch(pairs, layer, pooling, batch_size, progress, desc)


def build_steering_vector(model: SteeringModel, cfg: Config,
                          pos: list[Conversation], neg: list[Conversation],
                          progress=None, tqdm=False):
    """Return (SteeringVector, A_pos, A_neg) — activations reused by the classifier."""
    def prog(tag):
        return (lambda d, t: progress(tag, d, t)) if progress else None
    A_pos = collect_activations(model, pos, cfg.layer, cfg.pooling, cfg.batch_size,
                                prog("pos"), desc="activations[pos]" if tqdm else None)
    A_neg = collect_activations(model, neg, cfg.layer, cfg.pooling, cfg.batch_size,
                                prog("neg"), desc="activations[neg]" if tqdm else None)

    diff = A_pos.mean(0) - A_neg.mean(0)
    raw_norm = float(diff.norm())
    vec = diff / (raw_norm + 1e-8) if cfg.normalize else diff

    sv = SteeringVector(
        vector=vec, layer=cfg.layer, pooling=cfg.pooling, concept=cfg.concept,
        raw_norm=raw_norm, n_pos=len(pos), n_neg=len(neg),
    )
    return sv, A_pos, A_neg


if __name__ == "__main__":
    from steering_vectors.data import load_conversations, split_by_label

    cfg = Config.from_yaml("steering/configs/rude.yaml")
    cfg.device = "cpu"
    model = SteeringModel(cfg)
    convs = load_conversations(cfg, n=20)
    pos, neg = split_by_label(convs, cfg)
    sv, A_pos, A_neg = build_steering_vector(model, cfg, pos, neg)
    print(f"steering vector: shape={tuple(sv.vector.shape)} raw_norm={sv.raw_norm:.3f} "
          f"norm={sv.vector.norm():.3f} pos={sv.n_pos} neg={sv.n_neg}")
