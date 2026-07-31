"""Data loading for contrastive steering.

A `Conversation` is one (prompt, response, label) triple. Positive-label rows
carry the concept (e.g. rude), negative-label rows don't. Load from the HF Hub
or from a local JSON file (list of objects with the same keys).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from steering_vectors.config import Config


@dataclass
class Conversation:
    prompt: str
    response: str
    label: int          # +1 concept present, -1 absent


def _rows_from_hf(cfg: Config) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name, split=cfg.split)
    return [dict(r) for r in ds]


def _rows_from_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):                # allow {"data": [...]} wrappers
        data = data.get("data", data.get("conversations", []))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of conversation objects")
    return data


def load_conversations(cfg: Config, n: int | None = None) -> list[Conversation]:
    """Return up to `n` conversations (defaults to cfg.n_samples).

    Rows are balanced across pos/neg labels so the contrast is not skewed.
    """
    rows = _rows_from_json(cfg.data_file) if cfg.data_file else _rows_from_hf(cfg)
    n = cfg.n_samples if n is None else n

    pos, neg = [], []
    for r in rows:
        conv = Conversation(
            prompt=str(r[cfg.prompt_key]),
            response=str(r[cfg.response_key]),
            label=int(r[cfg.label_key]),
        )
        if conv.label == cfg.pos_label:
            pos.append(conv)
        elif conv.label == cfg.neg_label:
            neg.append(conv)

    half = n // 2
    take = min(half, len(pos), len(neg))
    balanced = pos[:take] + neg[:take]
    if not balanced:
        raise ValueError("No conversations matched pos/neg labels — check config keys/labels")
    return balanced


def split_by_label(convs: list[Conversation], cfg: Config):
    """Split into (positive, negative) lists."""
    pos = [c for c in convs if c.label == cfg.pos_label]
    neg = [c for c in convs if c.label == cfg.neg_label]
    return pos, neg


if __name__ == "__main__":
    cfg = Config.from_yaml("steering/configs/rude.yaml")
    convs = load_conversations(cfg, n=20)
    pos, neg = split_by_label(convs, cfg)
    print(f"loaded {len(convs)} conversations | pos={len(pos)} neg={len(neg)}")
    print("pos[0]:", pos[0].prompt[:60], "->", pos[0].response[:60])
    print("neg[0]:", neg[0].prompt[:60], "->", neg[0].response[:60])
