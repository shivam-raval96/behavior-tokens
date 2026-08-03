"""Steering-curve evaluation.

Sweep the steering multiplier across a range, and at each scale generate
responses under steering, re-encode them, and score with the concept
classifier. Produces a curve of concept-rate vs. steering scale — the core
evidence that the vector controls the behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from steering_vectors.classifier import ConceptClassifier
from steering_vectors.config import Config
from steering_vectors.data import Conversation
from steering_vectors.model import SteeringModel
from steering_vectors.steering import SteeringVector


@dataclass
class CurvePoint:
    scale: float
    concept_rate: float        # fraction of generations classified as concept-present
    mean_prob: float           # mean P(concept) over generations


def _score_generations(model: SteeringModel, clf: ConceptClassifier,
                       prompts: list[str], gens: list[str], batch_size: int) -> tuple[float, float]:
    """Re-encode (prompt, generation) pairs and classify them."""
    pairs = list(zip(prompts, gens))
    acts = model.collect_activations_batch(pairs, clf.layer, clf.pooling, batch_size)
    preds = clf.predict(acts)
    probs = clf.prob_pos(acts)
    return float((preds == 1).mean()), float(probs.mean())


def steering_curve(model: SteeringModel, cfg: Config, sv: SteeringVector,
                   clf: ConceptClassifier, prompts: list[str],
                   skip_scales: set[float] | None = None,
                   on_point=None) -> list[CurvePoint]:
    """Generate under each steering scale and score the outputs.

    Resumable: scales in `skip_scales` (already computed) are skipped, and
    `on_point(CurvePoint)` is called after each new point so callers can
    persist it immediately.
    """
    skip = {round(s, 4) for s in (skip_scales or set())}
    curve = []
    for scale in cfg.curve_scales():
        if round(scale, 4) in skip:
            continue
        model.add_steering(sv.vector, sv.layer, scale)
        gens = model.generate_batch(prompts, cfg.batch_size)
        model.clear_steering()
        rate, mprob = _score_generations(model, clf, prompts, gens, cfg.batch_size)
        pt = CurvePoint(scale=scale, concept_rate=rate, mean_prob=mprob)
        curve.append(pt)
        print(f"  scale={scale:+.1f}  concept_rate={rate:.3f}  mean_prob={mprob:.3f}")
        if on_point is not None:
            on_point(pt)
        _free_memory()                            # release KV/MPS cache between scales
    return curve


def _free_memory():
    """Free cached device memory so long curves don't creep into swap."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_prompts_from(convs: list[Conversation], n: int) -> list[str]:
    """Dedupe prompts (dataset pairs share prompts) and take the first n."""
    seen, out = set(), []
    for c in convs:
        if c.prompt not in seen:
            seen.add(c.prompt)
            out.append(c.prompt)
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    # Tiny smoke test: 2 prompts, 3 scales, verify curve monotone-ish.
    from steering_vectors.data import load_conversations, split_by_label
    from steering_vectors.steering import build_steering_vector
    from steering_vectors.classifier import train_classifier

    cfg = Config.from_yaml("steering/configs/rude.yaml")
    cfg.device = "cpu"
    cfg.max_new_tokens = 24
    cfg.curve_min, cfg.curve_max, cfg.curve_step = -4.0, 4.0, 4.0   # scales -4,0,4
    model = SteeringModel(cfg)
    convs = load_conversations(cfg, n=40)
    pos, neg = split_by_label(convs, cfg)
    sv, A_pos, A_neg = build_steering_vector(model, cfg, pos, neg)
    clf = train_classifier(cfg, A_pos, A_neg)
    print(f"clf test_acc={clf.test_acc:.3f}")
    prompts = eval_prompts_from(convs, 3)
    curve = steering_curve(model, cfg, sv, clf, prompts)
    print("curve:", [(p.scale, round(p.concept_rate, 2)) for p in curve])
