"""Re-score a finished results.json with the current asr.py — no GPU rerun.

Generations are saved per behavior, so ASR can be recomputed offline whenever the
refusal filter improves (e.g. adding non-Latin-script / new-language refusal
detection). Rewrites summary + per-behavior success flags in place and regenerates
the plot. Keeps a copy of the pre-rescore summary under `summary_prev`.

  .venv/bin/python -m jailbreaks.gcg_bench.rescore jailbreaks/outputs/jbb_run/results.json
"""
from __future__ import annotations

import json
import sys

from .asr import is_success
from .benchmark import make_plot


def rescore(path: str):
    with open(path) as f:
        d = json.load(f)

    prev = {k: v.copy() for k, v in d["summary"].items()}
    for ds, rows in d["per_behavior"].items():
        atk = [r["attacked_gen"] for r in rows]
        base = [r["baseline_gen"] for r in rows]
        for r in rows:
            r["attacked_success"] = is_success(r["attacked_gen"])
            r["baseline_success"] = is_success(r["baseline_gen"])
        n = len(rows)
        d["summary"][ds] = {
            "n": n,
            "asr_baseline": round(sum(is_success(g) for g in base) / n, 4) if n else 0.0,
            "asr_gcg": round(sum(is_success(g) for g in atk) / n, 4) if n else 0.0,
        }
    d["summary_prev"] = prev
    d.setdefault("config", {})["metric"] = (
        "refusal-string-match ASR + non-Latin-script guard (rescored)")

    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    plot_path = path.rsplit("/", 1)[0] + "/asr.png"
    make_plot(d["summary"], plot_path, d["config"].get("model", "model"))

    print(f"[rescore] {path}")
    for ds in d["summary"]:
        p, c = prev.get(ds, {}), d["summary"][ds]
        print(f"  {ds}: ASR_gcg {p.get('asr_gcg')} -> {c['asr_gcg']}   "
              f"baseline {p.get('asr_baseline')} -> {c['asr_baseline']}")
    print(f"[write] {plot_path}")
    return d


if __name__ == "__main__":
    rescore(sys.argv[1] if len(sys.argv) > 1 else "jailbreaks/outputs/jbb_run/results.json")
