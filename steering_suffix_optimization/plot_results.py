"""Generate a compact optimization/ASR summary plot from a completed run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt


def escaped_suffix(text: str, width: int = 78) -> str:
    visible = text.encode("unicode_escape").decode("ascii")
    return "\n".join(textwrap.wrap(visible, width=width, break_long_words=True,
                                    break_on_hyphens=False))


def plot_run(run_dir: Path, output: Path | None = None) -> Path:
    result = json.loads((run_dir / "results.json").read_text())
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history = result["history"]
    steps = [row["step"] for row in history]
    losses = [row["loss"] for row in history]
    cosines = [row["cosine"] for row in history]
    projections = [row["projection"] for row in history]
    judged_path = run_dir / "harmbench_results.json"
    asr = json.loads(judged_path.read_text()) if judged_path.exists() else result["asr"]
    judged = judged_path.exists()
    output = output or run_dir / "optimization_and_asr.png"

    fig = plt.figure(figsize=(12, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(3.2, 1.15))
    axis = fig.add_subplot(grid[0, 0])
    metric_axis = axis.twinx()
    axis.plot(steps, losses, color="#355f8a", linewidth=2, label="Activation-match loss")
    metric_axis.plot(steps, cosines, color="#2a9d6f", linewidth=2, label="Cosine")
    metric_axis.plot(steps, projections, color="#e08b32", linewidth=2, label="Projection")
    axis.set(xlabel="Optimization step", ylabel="Loss", title="Steering-vector matching")
    metric_axis.set_ylabel("Alignment metric")
    axis.grid(alpha=0.22)
    lines = axis.lines + metric_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=False)
    axis.annotate(f"best loss {min(losses):.4f}", xy=(steps[losses.index(min(losses))], min(losses)),
                  xytext=(-8, 14), textcoords="offset points", ha="right", fontsize=9)

    asr_axis = fig.add_subplot(grid[0, 1])
    values = [asr["baseline_asr"], asr["suffix_asr"]]
    bars = asr_axis.bar(["Baseline", "Best suffix"], values,
                        color=["#7b8794", "#c4554d"], width=0.62)
    asr_axis.set_ylim(0, max(0.4, max(values) * 1.35))
    asr_axis.set(ylabel="HarmBench ASR" if judged else "Diagnostic ASR",
                 title=f"Paired {'HarmBench ' if judged else ''}ASR (n={asr['n']})")
    asr_axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        asr_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                      f"{value:.0%}", ha="center", va="bottom", fontsize=11)
    delta = asr["delta"] if judged else asr["asr_delta"]
    asr_axis.text(0.5, 0.03, f"delta {delta:+.0%}",
                  transform=asr_axis.transAxes, ha="center", va="bottom", fontsize=11)

    suffix_axis = fig.add_subplot(grid[1, :])
    suffix_axis.axis("off")
    suffix_axis.text(0, 0.96, "Best suffix", transform=suffix_axis.transAxes,
                     fontsize=11, fontweight="bold", va="top")
    suffix_axis.text(0, 0.73, escaped_suffix(result["suffix_text"]),
                     transform=suffix_axis.transAxes, family="monospace", fontsize=9, va="top")
    suffix_axis.text(0, 0.12, "Token IDs: " + " ".join(map(str, result["suffix_ids"])),
                     transform=suffix_axis.transAxes, family="monospace", fontsize=8, va="bottom")
    title = ("Negative-direction steering suffix comparison"
             if float(config.get("vector_scale", 1.0)) < 0
             else "Steering suffix jailbreak pilot")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(plot_run(args.run_dir, args.output))


if __name__ == "__main__":
    main()
