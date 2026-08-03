"""Add steering-curve points to an existing artifact without a full rerun.

Reuses the saved steering vector + classifier from a run artifact, evaluates
only the requested (missing) scales, and merges them into the curve (sorted).

Usage:
    python -m steering.add_scales <artifact.json> <scale> [<scale> ...]
    python -m steering.add_scales steering/outputs/steering_rude_L8_mean.json 3
"""
from __future__ import annotations

import json
import sys

from steering_vectors import checkpoint as ck
from steering_vectors.config import Config
from steering_vectors.data import load_conversations
from steering_vectors.evaluate import eval_prompts_from, steering_curve
from steering_vectors.model import SteeringModel


def add_scales(artifact_path: str, scales: list[float]) -> str:
    with open(artifact_path) as f:
        art = json.load(f)
    cfg = Config(**art["config"])
    sv = ck.sv_from_dict(art["steering_vector"])
    clf = ck.clf_from_dict(art, cfg.layer, cfg.pooling)

    existing = {round(p["scale"], 4) for p in art["steering_curve"]}
    todo = [s for s in scales if round(s, 4) not in existing]
    if not todo:
        print("all requested scales already present:", scales)
        return artifact_path
    print(f"evaluating new scales: {todo}")

    model = SteeringModel(cfg)
    prompts = eval_prompts_from(load_conversations(cfg), cfg.eval_n_prompts)

    # steering_curve iterates cfg.curve_scales(); point it at just the new scales.
    cfg.curve_min, cfg.curve_max, cfg.curve_step = min(todo), max(todo), (
        min(abs(b - a) for a, b in zip(sorted(todo), sorted(todo)[1:])) if len(todo) > 1 else 1.0
    )
    new_pts = [p for p in steering_curve(model, cfg, sv, clf, prompts)
               if round(p.scale, 4) in {round(s, 4) for s in todo}]

    merged = art["steering_curve"] + [
        {"scale": p.scale, "concept_rate": p.concept_rate, "mean_prob": p.mean_prob}
        for p in new_pts
    ]
    art["steering_curve"] = sorted(merged, key=lambda p: p["scale"])
    with open(artifact_path, "w") as f:
        json.dump(art, f, indent=2)

    print(f"\nupdated curve ({len(art['steering_curve'])} points) -> {artifact_path}")
    for p in art["steering_curve"]:
        print(f"  {p['scale']:+.1f}  {cfg.concept}_rate={p['concept_rate']:.2f}  mean_prob={p['mean_prob']:.2f}")
    try:
        from steering_vectors.plot import plot_curve
        print("plot ->", plot_curve(artifact_path))
    except Exception as e:
        print("plot skipped:", e)
    return artifact_path


if __name__ == "__main__":
    path = sys.argv[1]
    scales = [float(x) for x in sys.argv[2:]]
    add_scales(path, scales)
