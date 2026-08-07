from __future__ import annotations

import json
from pathlib import Path


def write_summary(run_dir: Path) -> Path:
    payload = json.loads((run_dir / "results.json").read_text())
    rows = payload["results"]
    c1 = [r for r in rows if r["positive_control"]]
    at_reference = [
        r
        for r in rows
        if not r["positive_control"]
        and r["alpha_multiplier"] == 1.0
        and r["constraint"] == "free"
    ]
    c1_best = min((r["heldout_normalized_kl"] for r in c1), default=float("nan"))
    reference_best = min(
        (r["heldout_normalized_kl"] for r in at_reference), default=float("nan")
    )
    if c1_best >= 0.3:
        decision = (
            "C1 failed: audit optimizer/splicing before interpreting reachability."
        )
    elif reference_best < 0.3:
        decision = (
            "Unconstrained suffix clears the 0.3 gate at alpha0: proceed to Phase 2."
        )
    else:
        passing = sorted(
            {
                r["alpha_multiplier"]
                for r in rows
                if not r["positive_control"]
                and r["constraint"] == "free"
                and r["heldout_normalized_kl"] < 0.3
            }
        )
        decision = (
            f"Reachability threshold alpha*={max(passing):g} alpha0; suggested working coefficient={0.8 * max(passing):g} alpha0."
            if passing
            else "No tested unconstrained condition clears the reachability gate."
        )
    text = (
        "# Results\n\n"
        f"- Run: `{payload['run_id']}`\n"
        f"- Best C1 held-out normalized KL: {c1_best:.4g}\n"
        f"- Best free k sweep at alpha0: {reference_best:.4g}\n"
        f"- Decision: {decision}\n\n"
        "See `metrics.jsonl`, `results.json`, and `normalized_kl.png` for the complete diagnostics.\n"
    )
    target = run_dir / "RESULTS.md"
    target.write_text(text)
    return target
