"""Plots for token-optimization (GCG) runs."""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gcg(artifact_path: str, out_path: str | None = None) -> str:
    """GCG result: loss-vs-step curve + concept-rate bars (clean/steering/suffix)."""
    with open(artifact_path) as f:
        art = json.load(f)
    concept = art["concept"]
    hist = art["loss_history"]
    ev = art.get("eval")

    fig, axes = plt.subplots(1, 2 if ev else 1, figsize=(11 if ev else 6, 4.2))
    ax0 = axes[0] if ev else axes
    ax0.plot(range(len(hist)), hist, color="#5a189a")
    ax0.set_xlabel("GCG step")
    ax0.set_ylabel(r"activation-match loss  $\||h_{suffix}-(h_{clean}+\alpha v)\||^2$")
    ax0.set_title(f"GCG convergence — '{concept}', layer {art['layer']}, α={art['target_scale']}")
    ax0.grid(True, alpha=0.3)

    if ev:
        conds = ["clean", "steering", "suffix"]
        rates = [ev[c]["concept_rate"] for c in conds]
        colors = ["#8d99ae", "#2a6f97", "#c1121f"]
        labels = ["clean", f"steering α={ev['target_scale']}", "GCG suffix"]
        ax1 = axes[1]
        bars = ax1.bar(labels, rates, color=colors)
        ax1.set_ylabel(f"{concept} rate (classified)")
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"Behavioral effect ({ev['n_prompts']} prompts)")
        for b, r in zip(bars, rates):
            ax1.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center")
        ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(artifact_path), "gcg_result.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_length_sweep(sweep_dir: str, out_path: str | None = None) -> str:
    """Suffix length vs. behavioral concept-rate, over gcg_len*.json in a dir."""
    pts, steer, concept = [], None, "concept"
    for p in glob.glob(os.path.join(sweep_dir, "gcg_len*.json")):
        with open(p) as f:
            a = json.load(f)
        ev = a.get("eval")
        if not ev:
            continue
        pts.append((a["suffix_len"], ev["suffix"]["concept_rate"]))
        steer = ev["steering"]["concept_rate"]
        concept = a["concept"]
    pts.sort()
    lens = [x[0] for x in pts]
    rates = [x[1] for x in pts]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if steer is not None:
        ax.axhline(steer, color="#2a6f97", ls="--", lw=1.3,
                   label=f"activation steering ({steer:.2f})")
    ax.plot(lens, rates, "o-", color="#c1121f", lw=2, ms=7, label="GCG suffix")
    if rates:
        i = max(range(len(rates)), key=lambda k: rates[k])
        ax.annotate(f"peak: {lens[i]} tok → {rates[i]:.2f}", (lens[i], rates[i]),
                    textcoords="offset points", xytext=(8, -16), fontsize=9, color="#c1121f")
    ax.set_xlabel("suffix length (tokens)")
    ax.set_ylabel(f"{concept} rate (classified)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Behavior-token attack vs. suffix length — '{concept}'")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(sweep_dir, "length_sweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = sys.argv[1]
    print("saved ->", (plot_length_sweep if os.path.isdir(path) else plot_gcg)(path))
