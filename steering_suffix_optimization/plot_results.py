from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(run_dir: Path) -> Path:
    payload = json.loads((run_dir / "results.json").read_text())
    rows = payload["results"]
    main = [row for row in rows if not row["positive_control"]]
    figure, axis = plt.subplots(figsize=(9, 6))
    for length in sorted({row["suffix_length"] for row in main}):
        for constraint in sorted({row["constraint"] for row in main}):
            selected = [
                row
                for row in main
                if row["suffix_length"] == length and row["constraint"] == constraint
            ]
            xs = sorted({row["alpha_multiplier"] for row in selected})
            ys = [
                np.mean(
                    [
                        row["heldout_normalized_kl"]
                        for row in selected
                        if row["alpha_multiplier"] == x
                    ]
                )
                for x in xs
            ]
            axis.plot(xs, ys, marker="o", label=f"soft k={length}, {constraint}")
    xs = sorted({row["alpha_multiplier"] for row in main})
    for key, label, style in (
        ("natural_language_normalized_kl", "natural-language", "--"),
        ("random_normalized_kl", "random tokens", ":"),
    ):
        ys = [
            np.mean([row[key] for row in main if row["alpha_multiplier"] == x])
            for x in xs
        ]
        axis.plot(xs, ys, style, linewidth=2, label=label)
    controls = [row["heldout_normalized_kl"] for row in rows if row["positive_control"]]
    if controls:
        axis.axhline(
            min(controls), color="black", linestyle="-.", label="C1 positive control"
        )
    axis.axhline(0.3, color="gray", alpha=0.5, label="decision gate")
    axis.set(
        xlabel=r"$\alpha / \alpha_0$",
        ylabel="normalized forward KL",
        yscale="log",
        title="Steered-teacher soft-suffix reachability",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    target = run_dir / "normalized_kl.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
