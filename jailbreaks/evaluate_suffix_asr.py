"""Resumable ASR evaluation for an already-found jailbreak suffix."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm.auto import tqdm

from jailbreaks.asr import is_success
from claude_legacy.jailbreaks.gcg_bench.datasets_hf import load
from claude_legacy.jailbreaks.gcg_bench.gcg_zou import GCGConfig
from jailbreaks.gcg_small_scale import CheckpointedGCG, load_model, run_paths


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("stage") != "stage_d_suffix_asr":
        raise ValueError("config must select stage_d_suffix_asr")
    if config.get("dataset") != "advbench" or not isinstance(config.get("n_behaviors"), int):
        raise ValueError("stage D requires an AdvBench behavior count")
    if config["n_behaviors"] < 2 or config["n_behaviors"] > 100:
        raise ValueError("stage D supports 2–100 behaviors")
    if config.get("run_mode", "fresh") not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    if not isinstance(config.get("source_result"), str):
        raise ValueError("source_result is required")
    return config


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def run(config_path: Path, output_base: Path,
        checkpoint_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    config = read_config(config_path)
    paths = run_paths(config, output_base)
    fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint = None
    if config["run_mode"] == "resume":
        if not paths["checkpoint"].exists():
            raise FileNotFoundError("cannot resume without checkpoint.json")
        checkpoint = json.loads(paths["checkpoint"].read_text())
        if checkpoint.get("config_sha256") != fingerprint:
            raise ValueError("checkpoint configuration does not match the YAML")

    source_path = output_base / config["source_result"]
    source = json.loads(source_path.read_text())
    suffix_ids = source.get("suffix", {}).get("token_ids")
    if not isinstance(suffix_ids, list) or not suffix_ids:
        raise ValueError("source result does not contain a suffix token sequence")

    selected = load(config["dataset"], n=config["skip_first"] + config["n_behaviors"], seed=config["seed"])
    selected = selected[config["skip_first"]:]
    source_goals = {row["goal"] for group in source.get("private_artifacts", {}).values() for row in group}
    if any(row["goal"] in source_goals for row in selected):
        raise ValueError("evaluation sample overlaps the source run")

    model, tokenizer, device = load_model(config)
    attack = CheckpointedGCG(
        model, tokenizer,
        GCGConfig(suffix_len=len(suffix_ids), steps=1, topk=1, search_batch=1,
                  eval_chunk=1, seed=config["seed"]),
    )
    rows = list(checkpoint.get("behaviors", [])) if checkpoint else []
    started = time.monotonic()
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if not (paths["directory"] / "config.yaml").exists():
        shutil.copyfile(config_path, paths["directory"] / "config.yaml")

    def save(status: str) -> None:
        payload = {
            "status": status,
            "config_sha256": fingerprint,
            "metadata": {
                "model_id": config["model_id"], "device": device,
                "source_result": config["source_result"], "dataset": config["dataset"],
                "seed": config["seed"], "skip_first": config["skip_first"],
                "n_behaviors": config["n_behaviors"], "max_new_tokens": config["max_new_tokens"],
            },
            "suffix": {"token_ids": suffix_ids, "decoded": tokenizer.decode(suffix_ids)},
            "completed_behaviors": len(rows),
            "asr": {
                "baseline": _rate(rows, "baseline_success"),
                "gcg": _rate(rows, "attacked_success"),
                "delta": _rate(rows, "attacked_success") - _rate(rows, "baseline_success"),
            },
            "private_artifacts": {"behaviors": rows},
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        _write(paths["checkpoint"], payload)
        _write(paths["progress"], payload)
        if checkpoint_callback:
            checkpoint_callback()

    try:
        for index in tqdm(range(len(rows), len(selected)), desc="Stage D ASR", unit="behavior"):
            row = selected[index]
            baseline = attack.generate(row["goal"], None, config["max_new_tokens"])
            attacked = attack.generate(row["goal"], suffix_ids, config["max_new_tokens"])
            rows.append({
                "behavior_id": hashlib.sha256(row["goal"].encode()).hexdigest()[:12],
                "goal": row["goal"], "target": row["target"],
                "baseline_response": baseline, "attacked_response": attacked,
                "baseline_success": is_success(baseline), "attacked_success": is_success(attacked),
            })
            if len(rows) % config["progress_every"] == 0 or len(rows) == len(selected):
                print("METRIC " + json.dumps({"type": "suffix_asr_progress", "completed": len(rows),
                    "total": len(selected), "baseline_asr": _rate(rows, "baseline_success"),
                    "gcg_asr": _rate(rows, "attacked_success")}, sort_keys=True), flush=True)
            if len(rows) % config["checkpoint_every"] == 0 or len(rows) == len(selected):
                save("running")
    except KeyboardInterrupt:
        save("stopped")
        raise

    save("complete")
    result = json.loads(paths["checkpoint"].read_text())
    _write(paths["result"], result)
    paths["summary"].write_text(
        "# Stage D suffix ASR results\n\nStatus: completed\n\n"
        f"- Behaviors: {result['completed_behaviors']}\n"
        f"- Baseline ASR: {result['asr']['baseline']:.3f}\n"
        f"- GCG-suffix ASR: {result['asr']['gcg']:.3f}\n"
        f"- ASR change: {result['asr']['delta']:+.3f}\n"
    )
    if checkpoint_callback:
        checkpoint_callback()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("jailbreaks/configs/stage_d_suffix_asr_25_advbench.yaml"))
    parser.add_argument("--output-base", type=Path, default=Path())
    args = parser.parse_args()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        run(args.config, args.output_base)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
