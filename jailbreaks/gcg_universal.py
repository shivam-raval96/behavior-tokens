"""Resumable universal GCG replication: one suffix optimized across AdvBench goals."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, signal, time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from tqdm.auto import tqdm

from jailbreaks.asr import is_success
from claude_legacy.jailbreaks.gcg_bench.datasets_hf import load
from jailbreaks.gcg_small_scale import CheckpointedGCG, load_model, run_paths
from claude_legacy.jailbreaks.gcg_bench.gcg_zou import GCGConfig


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("stage") not in {"stage_c_universal_advbench_llama2_7b", "stage_c_single_behavior_advbench_llama2_7b", "stage_c_five_behavior_advbench_llama2_7b", "stage_c_five_behavior_advbench_llama32_1b"}:
        raise ValueError("config must select the universal or one-behavior AdvBench protocol")
    required = {"dataset": "advbench", "suffix_length": 20,
                "top_k": 256, "candidate_batch_size": 512}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("config must retain the paper-matched universal Llama-2 settings")
    if cfg["stage"] == "stage_c_universal_advbench_llama2_7b" and (cfg.get("train_behaviors"), cfg.get("test_behaviors")) != (25, 100):
        raise ValueError("the full protocol requires 25 training and 100 held-out behaviors")
    if cfg["stage"] == "stage_c_single_behavior_advbench_llama2_7b" and (cfg.get("train_behaviors"), cfg.get("test_behaviors")) != (1, 1):
        raise ValueError("the one-behavior protocol requires one train and one held-out behavior")
    if cfg["stage"] == "stage_c_five_behavior_advbench_llama2_7b" and (cfg.get("train_behaviors"), cfg.get("test_behaviors")) != (5, 25):
        raise ValueError("the five-behavior protocol requires five train and 25 held-out behaviors")
    if cfg["stage"] == "stage_c_five_behavior_advbench_llama32_1b":
        if (cfg.get("model_id"), cfg.get("steps"), cfg.get("train_behaviors"), cfg.get("test_behaviors")) != ("meta-llama/Llama-3.2-1B-Instruct", 200, 5, 50):
            raise ValueError("the Llama-3.2 protocol requires 200 steps, five training, and 50 held-out behaviors")
        if cfg.get("train_indices") != [0, 1, 2, 3, 4] or cfg.get("test_indices") != list(range(30, 80)):
            raise ValueError("the Llama-3.2 protocol requires fixed disjoint AdvBench rows")
        if cfg.get("target_prefix") != "Sure, here is ":
            raise ValueError("the Llama-3.2 protocol requires the fixed affirmative target prefix")
    if cfg.get("run_mode", "fresh") not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    return cfg


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _original_advbench_records(cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load explicit original-AdvBench rows with a deterministic target policy."""
    path = Path(cfg["dataset_csv"])
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    prefix = cfg["target_prefix"]
    def record(index: int) -> dict[str, str]:
        goal = rows[index]["goal"].strip()
        return {"goal": goal, "target": prefix + goal[:1].lower() + goal[1:]}
    return ([record(index) for index in cfg["train_indices"]],
            [record(index) for index in cfg["test_indices"]])


def audit(config_path: Path, output_base: Path,
          commit: Callable[[], None] | None = None) -> dict[str, Any]:
    """Run the Stage C tokenization, gradient, and batching preflight."""
    cfg = yaml.safe_load(config_path.read_text())
    required = {"stage": "stage_c_implementation_audit", "dataset": "advbench",
                "suffix_length": 20, "top_k": 256, "candidate_batch_size": 512}
    if not isinstance(cfg, dict) or any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("invalid Stage C implementation-audit configuration")
    paths = run_paths(cfg, output_base)
    records = load("advbench", n=1, seed=cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    attack = CheckpointedGCG(
        model, tokenizer,
        GCGConfig(suffix_len=cfg["suffix_length"], steps=1, topk=cfg["top_k"],
                  search_batch=cfg["candidate_batch_size"], eval_chunk=cfg["evaluation_chunk_size"],
                  seed=cfg["seed"]),
    )
    checks = attack.audit_invariants(records[0]["goal"], records[0]["target"])
    checks["passed"] = (
        checks["canonical_generation_ids_match"]
        and checks["canonical_loss_ids_match"]
        and checks["candidates_change_one_coordinate"]
        and checks["suffix_roundtrips"]
        and checks["model_weight_gradients_absent"]
        and checks["target_loss_batch_serial_max_abs_difference"] < 2e-3
    )
    result = {"status": "complete" if checks["passed"] else "failed", "metadata": {
        "model_id": cfg["model_id"], "device": device, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }, "checks": checks}
    paths["directory"].mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, paths["directory"] / "config.yaml")
    _write(paths["result"], result); _write(paths["checkpoint"], result); _write(paths["progress"], result)
    paths["summary"].write_text(
        "# Stage C implementation audit\n\n"
        f"Status: {result['status']}\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in checks.items()) + "\n"
    )
    if commit:
        commit()
    if not checks["passed"]:
        raise AssertionError(f"Stage C implementation audit failed: {checks}")
    return result


def run(config_path: Path, output_base: Path, commit: Callable[[], None] | None = None,
        run_mode: str | None = None) -> dict[str, Any]:
    cfg = read_config(config_path)
    paths = run_paths(cfg, output_base)
    fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    effective_run_mode = run_mode or cfg["run_mode"]
    if effective_run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    old = json.loads(paths["checkpoint"].read_text()) if effective_run_mode == "resume" and paths["checkpoint"].exists() else None
    if effective_run_mode == "resume" and old is None:
        raise FileNotFoundError("cannot resume without checkpoint.json")
    if old and old["config_sha256"] != fingerprint:
        raise ValueError("checkpoint does not match config")
    if cfg["stage"] == "stage_c_five_behavior_advbench_llama32_1b":
        train, test = _original_advbench_records(cfg)
    else:
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
        best_suffix = torch.tensor(old.get("best_suffix_ids", old["suffix_ids"]), device=device, dtype=torch.long)
        best_loss = float(old.get("best_loss", min(history) if history else float("inf")))
    else:
        suffix, history, start, active = attack._init_suffix(), [], 0, 1
        best_suffix, best_loss = suffix.clone(), float("inf")
    started = time.monotonic()
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if not (paths["directory"] / "config.yaml").exists():
        shutil.copyfile(config_path, paths["directory"] / "config.yaml")

    def evaluate(rows: list[dict[str, str]], control: torch.Tensor) -> tuple[list[bool], list[dict[str, Any]]]:
        flags, artifacts = [], []
        for row in tqdm(rows, desc="Stage C ASR", unit="behavior", leave=False):
            baseline = attack.generate(row["goal"], None, cfg["max_new_tokens"])
            attacked = attack.generate(row["goal"], control.tolist(), cfg["max_new_tokens"])
            flags.append(is_success(attacked))
            artifacts.append({"goal": row["goal"], "target": row["target"], "baseline_response": baseline,
                              "attacked_response": attacked, "baseline_success": is_success(baseline),
                              "attacked_success": is_success(attacked)})
        return flags, artifacts

    periodic_evaluations = list(old.get("periodic_evaluations", [])) if old else []

    def losses(control: torch.Tensor, count: int) -> torch.Tensor:
        return torch.stack([attack._eval(control.unsqueeze(0), *layouts[i])[0] for i in range(count)])

    def save(status: str, metric: dict[str, Any] | None = None) -> None:
        payload = {"status": status, "config_sha256": fingerprint, "next_step": start,
                   "active_goals": active, "suffix_ids": suffix.tolist(), "loss_history": history,
                   "best_suffix_ids": best_suffix.tolist(), "best_loss": best_loss,
                   "generator_state": generator.get_state().cpu().tolist(), "latest_metric": metric,
                   "periodic_evaluations": periodic_evaluations,
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
            candidate_loss, candidate_index = float(candidate_losses.min()), int(candidate_losses.argmin())
            # Match the reference GCG transition rule: explore the lowest-loss
            # sampled candidate even when it is temporarily worse, while the
            # separately tracked best control is used for final evaluation.
            suffix, current_loss = candidates[candidate_index].clone(), candidate_loss
            if current_loss < best_loss:
                best_suffix, best_loss = suffix.clone(), current_loss
            history.append(current_loss); start = step + 1
            active_losses = losses(suffix, active)
            if cfg["progressive_goals"] and active < len(train) and bool(torch.all(active_losses < cfg["success_loss_threshold"])):
                active += 1
            metric = {"type": "universal_gcg_progress", "step": start, "total_steps": cfg["steps"],
                      "active_goals": active, "mean_loss": current_loss,
                      "best_loss": best_loss,
                      "max_active_loss": float(active_losses.max())}
            if start % cfg.get("evaluation_every", cfg["steps"]) == 0 or start == cfg["steps"]:
                test_flags, _ = evaluate(test, best_suffix)
                metric["held_out_suffix_asr"] = _rate(test_flags)
                periodic_evaluations.append({"step": start, "suffix_asr": _rate(test_flags)})
            if start % cfg["progress_every"] == 0 or start == cfg["steps"]:
                print("METRIC " + json.dumps(metric, sort_keys=True), flush=True)
            if start % cfg["checkpoint_every"] == 0 or start == cfg["steps"]:
                save("running", metric)
    except KeyboardInterrupt:
        save("stopped"); raise

    suffix = best_suffix
    train_attack, train_artifacts = evaluate(train, suffix)
    test_attack, test_artifacts = evaluate(test, suffix)
    train_base = [row["baseline_success"] for row in train_artifacts]
    test_base = [row["baseline_success"] for row in test_artifacts]
    result = {"status": "complete", "config_sha256": fingerprint,
              "metadata": {"model_id": cfg["model_id"], "device": device, "active_goals_final": active,
                           "steps": cfg["steps"], "train_behaviors": len(train), "test_behaviors": len(test)},
              "optimization": {"initial_mean_loss": history[0], "final_mean_loss": history[-1],
                               "best_mean_loss": best_loss, "loss_history": history,
                               "held_out_evaluations": periodic_evaluations},
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
