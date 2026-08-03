"""Overlay steering curves from several vector artifacts on one axis.

Usage:
    python -m steering_vectors.plot_compare <out.png> <label>=<artifact.json> ...
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#c1121f", "#2a6f97", "#5a189a", "#588157", "#e07a00"]


def plot_compare(out_path: str, named_artifacts: list[tuple[str, str]]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    concept = layer = None
    for i, (label, path) in enumerate(named_artifacts):
        art = json.load(open(path))
        curve = sorted(art["steering_curve"], key=lambda p: p["scale"])
        concept = art["steering_vector"]["concept"]
        layer = art["steering_vector"]["layer"]
        ax.plot([p["scale"] for p in curve], [p["concept_rate"] for p in curve],
                "o-", color=COLORS[i % len(COLORS)], lw=2, ms=6, label=label)
    ax.axvline(0, color="0.8", lw=1, zorder=0)
    ax.set_xlabel("steering scale  (α · v)")
    ax.set_ylabel(f"{concept} rate (classified)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Steering direction comparison — '{concept}', layer {layer}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1]
    pairs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[2:]]
    print("saved ->", plot_compare(out, pairs))
