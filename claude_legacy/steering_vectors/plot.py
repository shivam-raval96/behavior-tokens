"""Plot steering performance from a steering-vector artifact."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curve(artifact_path: str, out_path: str | None = None) -> str:
    """Concept-rate + mean-probability vs. steering scale."""
    with open(artifact_path) as f:
        art = json.load(f)
    curve = sorted(art["steering_curve"], key=lambda p: p["scale"])
    concept = art["steering_vector"]["concept"]
    layer = art["steering_vector"]["layer"]

    scales = [p["scale"] for p in curve]
    rate = [p["concept_rate"] for p in curve]
    prob = [p["mean_prob"] for p in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(scales, rate, "o-", color="#c1121f", label=f"{concept} rate (classified)")
    ax.plot(scales, prob, "s--", color="#2a6f97", label="mean P(concept)")
    ax.axvline(0, color="0.7", lw=1, zorder=0)
    ax.axhline(0.5, color="0.85", lw=1, ls=":", zorder=0)
    ax.set_xlabel("steering scale  (α · v)")
    ax.set_ylabel("fraction / probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Steering performance — '{concept}' vector, layer {layer}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(artifact_path), "steering_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("saved ->", plot_curve(sys.argv[1]))
