#!/usr/bin/env python3
"""Fit source-specific probes from a previously cached activation tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = ("claude-opus-4.6", "gemini-3.1-pro", "gpt-5.4")
DISPLAY = {"claude-opus-4.6": "Claude", "gemini-3.1-pro": "Gemini", "gpt-5.4": "GPT"}
TRAIN_SOURCES = ("gemini-3.1-pro", "gpt-5.4")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def checkpoint(run: Path, started: float, payload: dict) -> None:
    snapshot = {
        "run_id": run.name,
        "elapsed_seconds": round(time.time() - started, 3),
        "throughput_layers_per_second": round(payload.get("completed", 0) / max(time.time() - started, 1e-9), 4),
        "error_count": 0,
        "retry_count": 0,
        **payload,
    }
    write_json(run / "progress.json", snapshot)
    with (run / "dashboard_history.jsonl").open("a") as f:
        f.write(json.dumps(snapshot, sort_keys=True) + "\n")
    html = f"""<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5>
<title>Cross-source probe fitting</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;background:#101418;color:#e8eef2}}progress{{width:100%}}pre{{background:#182028;padding:18px;overflow:auto}}</style>
<h1>Cached-activation cross-source probes</h1><h2>{snapshot['phase']}</h2>
<progress value="{snapshot.get('completed', 0)}" max="{snapshot.get('total', 64)}"></progress>
<pre>{json.dumps(snapshot, indent=2, sort_keys=True)}</pre>"""
    (run / "dashboard.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    source_run = args.source_run.resolve()
    run = args.output_run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()

    config = {
        "run_id": run.name,
        "trial_type": "cached-activation follow-up comparison",
        "source_run": str(source_run),
        "activation_file": str(source_run / "activations/all_layers.float16.npy"),
        "manifest_file": str(source_run / "manifest.jsonl"),
        "train_sources": list(TRAIN_SOURCES),
        "evaluation_sources": list(SOURCES),
        "split": "existing deterministic task-level 70/30 split",
        "probe": {"standard_scaler": True, "C": 1.0, "class_weight": "balanced", "max_iter": 5000, "random_state": 42},
        "selection": "highest same-source test accuracy; earliest-layer tie break; descriptive",
        "activation_extraction": False,
    }
    config["config_fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(run / "resolved_config.json", config)
    checkpoint(run, started, {"phase": "initialized", "completed": 0, "total": 64, "latest_objective": None, "best_metric": None, "best_layer": None, "config_fingerprint": config["config_fingerprint"]})

    rows = [json.loads(line) for line in (source_run / "manifest.jsonl").read_text().splitlines()]
    x = np.load(source_run / "activations/all_layers.float16.npy", mmap_mode="r")
    if x.shape[0] != len(rows):
        raise ValueError(f"activation rows {x.shape[0]} != manifest rows {len(rows)}")
    labels = np.array([row["label"] for row in rows], dtype=np.int8)
    sources = np.array([row["source_model"] for row in rows])
    splits = np.array([row["split"] for row in rows])
    expected_partitions = [(source, split) for source in SOURCES for split in ("train", "test")]
    empty = [(source, split) for source, split in expected_partitions if not np.any((sources == source) & (splits == split))]
    if empty:
        raise ValueError(f"empty manifest partitions: {empty}; observed sources={sorted(set(sources))}")
    all_results: dict[str, dict] = {}
    completed = 0

    for train_source in TRAIN_SOURCES:
        train_mask = (sources == train_source) & (splits == "train")
        test_mask = (sources == train_source) & (splits == "test")
        metrics = []
        probe_dir = run / f"probes_{DISPLAY[train_source].lower()}"
        probe_dir.mkdir()
        for layer in range(x.shape[1]):
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42))
            probe.fit(x[train_mask, layer].astype(np.float32), labels[train_mask])
            evaluations = {}
            for eval_source in SOURCES:
                for split in ("train", "test"):
                    mask = (sources == eval_source) & (splits == split)
                    pred = probe.predict(x[mask, layer].astype(np.float32))
                    evaluations[f"{eval_source}|{split}"] = {
                        "accuracy": float(accuracy_score(labels[mask], pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(labels[mask], pred)),
                        "confusion_matrix": confusion_matrix(labels[mask], pred, labels=[0, 1]).tolist(),
                        "n": int(mask.sum()),
                    }
            same_test = evaluations[f"{train_source}|test"]["accuracy"]
            same_train = evaluations[f"{train_source}|train"]["accuracy"]
            metrics.append({"layer": layer, "train_accuracy": same_train, "test_accuracy": same_test, "evaluations": evaluations})
            joblib.dump(probe, probe_dir / f"probe_layer_{layer:02d}.joblib")
            completed += 1
            current_best = max(metrics, key=lambda m: (m["test_accuracy"], -m["layer"]))
            checkpoint(run, started, {"phase": f"fitting_{DISPLAY[train_source].lower()}", "completed": completed, "total": 64, "latest_objective": same_test, "best_metric": current_best["test_accuracy"], "best_layer": current_best["layer"], "active_train_source": train_source, "config_fingerprint": config["config_fingerprint"]})

        best = max(metrics, key=lambda m: (m["test_accuracy"], -m["layer"]))
        best_layer = best["layer"]
        positions = np.arange(len(SOURCES)); width = 0.36
        train_scores = [best["evaluations"][f"{s}|train"]["accuracy"] for s in SOURCES]
        test_scores = [best["evaluations"][f"{s}|test"]["accuracy"] for s in SOURCES]
        fig, ax = plt.subplots(figsize=(8, 5))
        a = ax.bar(positions - width / 2, train_scores, width, label="Train tasks")
        b = ax.bar(positions + width / 2, test_scores, width, label="Test tasks")
        ax.bar_label(a, fmt="%.3f"); ax.bar_label(b, fmt="%.3f")
        ax.set_xticks(positions, [DISPLAY[s] for s in SOURCES]); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.08)
        ax.set_title(f"{DISPLAY[train_source]}-trained probe transfer at layer {best_layer}")
        ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
        plot_name = f"{DISPLAY[train_source].lower()}_trained_best_layer_cross_model_accuracy.png"
        fig.savefig(run / plot_name, dpi=180); plt.close(fig)
        all_results[train_source] = {"best_layer": best_layer, "best_layer_metrics": best, "per_layer": metrics, "plot": plot_name, "train_examples": int(train_mask.sum()), "test_examples": int(test_mask.sum())}
        write_json(run / "partial_results.json", all_results)

    result = {"status": "complete", "config_fingerprint": config["config_fingerprint"], "source_activation_shape": list(x.shape), "results": all_results}
    write_json(run / "results.json", result)
    lines = ["# Cached-activation cross-source probes", "", "No generations or activation extraction were performed.", ""]
    for source in TRAIN_SOURCES:
        item = all_results[source]; best = item["best_layer_metrics"]
        lines += [f"## {DISPLAY[source]}-trained probe", "", f"- Best layer: {item['best_layer']}", f"- Same-source test accuracy: {best['test_accuracy']:.4f}", "", "| Evaluation source | Train-task accuracy | Test-task accuracy |", "|---|---:|---:|"]
        for eval_source in SOURCES:
            lines.append(f"| {DISPLAY[eval_source]} | {best['evaluations'][eval_source + '|train']['accuracy']:.4f} | {best['evaluations'][eval_source + '|test']['accuracy']:.4f} |")
        lines.append("")
    (run / "RESULTS.md").write_text("\n".join(lines))
    checkpoint(run, started, {"phase": "complete", "completed": 64, "total": 64, "latest_objective": None, "best_metric": None, "best_layer": None, "config_fingerprint": config["config_fingerprint"]})


if __name__ == "__main__":
    main()
