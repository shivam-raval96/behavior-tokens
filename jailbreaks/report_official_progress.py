#!/usr/bin/env python3
"""Render a trustworthy progress snapshot from an official GCG JSON log.

The upstream progressive runner prints nested tqdm loops.  Those loop counters
are not a run-level progress signal.  Its JSON log is durable and receives one
control/loss/test snapshot at each configured ``test_steps`` boundary, so this
tool derives the global step solely from that artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text())
    params = result["params"]
    snapshots = len(result.get("controls", []))
    test_steps = int(params["test_steps"])
    completed = snapshots * test_steps
    total = int(params["n_steps"])
    latest_test = result.get("tests", [{}])[-1]

    status = {
        "source": str(args.result_json),
        "global_step": min(completed, total),
        "configured_steps": total,
        "progress_fraction": round(min(completed / total, 1.0), 4),
        "evaluation_snapshots": snapshots,
        "latest_optimizer_loss": result.get("losses", [None])[-1],
        "active_train_behaviors": latest_test.get("total", [None])[0],
        "held_out_behaviors": latest_test.get("total", [None])[1],
        "upstream_passed": latest_test.get("n_passed"),
        "upstream_exact_match": latest_test.get("n_em"),
        "upstream_avg_target_loss": latest_test.get("n_loss"),
        "note": (
            "Upstream passed/exact-match fields use its original refusal-prefix "
            "criterion. Final ASR is evaluated separately with the repository's "
            "trimmed-response-starts-with-Sure criterion."
        ),
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
