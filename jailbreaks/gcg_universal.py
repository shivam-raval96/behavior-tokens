"""Resumable universal GCG replication: one suffix optimized across AdvBench goals."""
from __future__ import annotations

import argparse, hashlib, json, os, signal, time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from tqdm.auto import tqdm

from claude_legacy.jailbreaks.gcg_bench.asr import is_success
from claude_legacy.jailbreaks.gcg_bench.datasets_hf import load
from jailbreaks.gcg_small_scale import CheckpointedGCG, load_model, run_paths
from claude_legacy.jailbreaks.gcg_bench.gcg_zou import GCGConfig


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("stage") not in {"stage_c_universal_advbench_llama2_7b", "stage_c_single_behavior_advbench_llama2_7b"}:
        raise ValueError("config must select the universal or one-behavior AdvBench protocol")
    required = {"dataset": "advbench", "suffix_length": 20,
                "steps": 500, "top_k": 256, "candidate_batch_size": 512}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("config must retain the paper-matched universal Llama-2 settings")
    if cfg["stage"] == "stage_c_universal_advbench_llama2_7b" and (cfg.get("train_behaviors"), cfg.get("test_behaviors")) != (25, 100):
        raise ValueError("the full protocol requires 25 training and 100 held-out behaviors")
    if cfg["stage"] == "stage_c_single_behavior_advbench_llama2_7b" and (cfg.get("train_behaviors"), cfg.get("test_behaviors")) != (1, 1):
        raise ValueError("the one-behavior protocol requires one train and one held-out behavior")
    if cfg.get("run_mode", "fresh") not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    return cfg


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def run(config_path: Path, output_base: Path, commit: Callable[[], None] | None = None) -> dict[str, Any]:
    cfg = read_config(config_path)
    paths = run_paths(cfg, output_base)
    fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    old = json.loads(paths["checkpoint"].read_text()) if cfg["run_mode"] == "resume" and paths["checkpoint"].exists() else None
    if cfg["run_mode"] == "resume" and old is None:
        raise FileNotFoundError("cannot resume without checkpoint.json")
    if old and old["config_sha256"] != fingerprint:
        raise ValueError("checkpoint does not match config")
    records = load("advbench", n=cfg["train_behaviors"] + cfg["test_behaviors"], seed=cfg["seed"])
    train, test = records[:cfg["train_behaviors"]], records[cfg["train_behaviors"]:]
    model, tokenizer, device = load_model(cfg)
    gcfg = GCGConfig(suffix_len=cfg["suffix_length"], steps=cfg["steps"], topk=cfg["top_k"],
                     search_batch=cfg["candidate_batch_size"], eval_chunk=cfg["evaluation_chunk_size"], seed=cfg["seed"])
    attack = CheckpointedGCG(model, tokenizer, gcfg)
    layouts = [attack._layout(row["goal"], row["target"]) for row in train]
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    if old:
        suffix = torch.tensor(old["suffix_ids"], device=device, dtype=torch.long)
        generator.set_state(torch.tensor(old["generator_state"], dtype=torch.uint8))
        history, start, active = old["loss_history"], old["next_step"], old["active_goals"]
    else:
        suffix, history, start, active = attack._init_suffix(), [], 0, 1
    started = time.monotonic()

    def losses(control: torch.Tensor, count: int) -> torch.Tensor:
        return torch.stack([attack._eval(control.unsqueeze(0), *layouts[i])[0] for i in range(count)])

    def save(status: str, metric: dict[str, Any] | None = None) -> None:
        payload = {"status": status, "config_sha256": fingerprint, "next_step": start,
                   "active_goals": active, "suffix_ids": suffix.tolist(), "loss_history": history,
                   "generator_state": generator.get_state().cpu().tolist(), "latest_metric": metric,
                   "elapsed_seconds": round(time.monotonic() - started, 1)}
        _write(paths["checkpoint"], payload); _write(paths["progress"], payload)
        if commit: commit()

    try:
        for step in tqdm(range(start, cfg["steps"]), desc="Stage C universal GCG", unit="step"):
            aggregate = torch.zeros(cfg["suffix_length"], attack.vocab, device=device, dtype=attack.W.dtype)
            for before, after, target in layouts[:active]:
                grad, _ = attack._token_gradients(suffix, before, after, target)
                aggregate += grad / grad.norm().clamp_min(1e-12)
            candidates = attack._sample(suffix, aggregate, generator)
            candidate_losses = torch.zeros(candidates.shape[0], device=device)
            for layout in layouts[:active]:
                candidate_losses += attack._eval(candidates, *layout)
            candidate_losses /= active
            suffix = candidates[int(candidate_losses.argmin())].clone()
            mean_loss = float(candidate_losses.min())
            history.append(mean_loss); start = step + 1
            active_losses = losses(suffix, active)
            if cfg["progressive_goals"] and active < len(train) and bool(torch.all(active_losses < cfg["success_loss_threshold"])):
                active += 1
            metric = {"type": "universal_gcg_progress", "step": start, "total_steps": cfg["steps"],
                      "active_goals": active, "mean_loss": mean_loss,
                      "max_active_loss": float(active_losses.max())}
            if start % cfg["progress_every"] == 0 or start == cfg["steps"]:
                print("METRIC " + json.dumps(metric, sort_keys=True), flush=True)
            if start % cfg["checkpoint_every"] == 0 or start == cfg["steps"]:
                save("running", metric)
    except KeyboardInterrupt:
        save("stopped"); raise

    def evaluate(rows: list[dict[str, str]]) -> tuple[list[bool], list[dict[str, Any]]]:
        flags, artifacts = [], []
        for row in tqdm(rows, desc="Stage C ASR", unit="behavior"):
            baseline = attack.generate(row["goal"], None, cfg["max_new_tokens"])
            attacked = attack.generate(row["goal"], suffix.tolist(), cfg["max_new_tokens"])
            flags.append(is_success(attacked))
            artifacts.append({"goal": row["goal"], "target": row["target"], "baseline_response": baseline,
                              "attacked_response": attacked, "baseline_success": is_success(baseline),
                              "attacked_success": is_success(attacked)})
        return flags, artifacts
    train_attack, train_artifacts = evaluate(train)
    test_attack, test_artifacts = evaluate(test)
    train_base = [row["baseline_success"] for row in train_artifacts]
    test_base = [row["baseline_success"] for row in test_artifacts]
    result = {"status": "complete", "config_sha256": fingerprint,
              "metadata": {"model_id": cfg["model_id"], "device": device, "active_goals_final": active,
                           "steps": cfg["steps"], "train_behaviors": len(train), "test_behaviors": len(test)},
              "optimization": {"initial_mean_loss": history[0], "final_mean_loss": history[-1], "loss_history": history},
              "suffix": {"token_ids": suffix.tolist(), "decoded": tokenizer.decode(suffix.tolist())},
              "asr": {"train_baseline": _rate(train_base), "train_gcg": _rate(train_attack),
                      "test_baseline": _rate(test_base), "test_gcg": _rate(test_attack)},
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    if cfg.get("store_raw_artifacts"):
        result["private_artifacts"] = {"train": train_artifacts, "test": test_artifacts}
    result["asr"]["test_delta"] = result["asr"]["test_gcg"] - result["asr"]["test_baseline"]
    _write(paths["result"], result); _write(paths["checkpoint"], result)
    paths["summary"].write_text("# Stage C universal GCG results\n\nStatus: completed\n\n"
        f"- Train ASR: {result['asr']['train_baseline']:.3f} → {result['asr']['train_gcg']:.3f}\n"
        f"- Held-out ASR: {result['asr']['test_baseline']:.3f} → {result['asr']['test_gcg']:.3f} "
        f"({result['asr']['test_delta']:+.3f})\n"
        f"- Mean optimization loss: {history[0]:.6f} → {history[-1]:.6f}\n\n"
        "Raw prompts, completions, and decoded suffix text are intentionally omitted.\n")
    if commit: commit()
    return result


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=Path("jailbreaks/configs/stage_c_universal_advbench_llama2_7b.yaml")); p.add_argument("--output-base", type=Path, default=Path())
    args = p.parse_args(); previous = signal.getsignal(signal.SIGTERM); signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try: run(args.config, args.output_base)
    finally: signal.signal(signal.SIGTERM, previous)

if __name__ == "__main__": main()
