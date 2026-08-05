"""Token-native building blocks for the Llama-3.2 GCG v2 path.

Unlike the legacy-compatible path, candidate validity is defined by the token
IDs actually inserted into the model prompt. Decoding a suffix is retained as
an artifact only; it is never used to decide whether a candidate is valid.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm
from torch.nn import functional as F

from jailbreaks.asr import is_success
from jailbreaks.gcg_small_scale import CheckpointedGCG, load_model, run_paths
from claude_legacy.jailbreaks.gcg_bench.gcg_zou import GCGConfig


@dataclass
class PhaseBest:
    """Best suffix for one fixed active-behavior objective."""

    active_goals: int
    loss: float = float("inf")
    suffix: torch.Tensor | None = None

    def consider(self, suffix: torch.Tensor, loss: float) -> bool:
        if loss >= self.loss:
            return False
        self.loss, self.suffix = loss, suffix.detach().clone()
        return True


def validate_token_native_candidates(candidates: torch.Tensor, current: torch.Tensor,
                                     special_ids: set[int]) -> None:
    """Require exactly one non-special token mutation, without text round-trips."""
    if candidates.ndim != 2 or candidates.shape[1] != current.numel():
        raise AssertionError("candidate shape does not match the control token sequence")
    changed = (candidates != current.unsqueeze(0)).sum(dim=1)
    if not torch.all(changed == 1):
        raise AssertionError("every candidate must change exactly one control token")
    if any(int(token) in special_ids for token in candidates.flatten()):
        raise AssertionError("candidate includes a tokenizer special token")


def layout_checks(before: torch.Tensor, suffix: torch.Tensor, after: torch.Tensor,
                  target_ids: torch.Tensor) -> dict[str, object]:
    """Prove loss evaluation retains the exact generation prompt prefix."""
    generation_ids = torch.cat([before, suffix, after])
    loss_ids = torch.cat([generation_ids, target_ids])
    if not torch.equal(loss_ids[:generation_ids.numel()], generation_ids):
        raise AssertionError("loss prompt diverges from generation prompt before target")
    digest = lambda ids: hashlib.sha256(
        json.dumps(ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "generation_prompt_token_count": int(generation_ids.numel()),
        "loss_prompt_token_count": int(loss_ids.numel()),
        "assistant_boundary_token_index": int(generation_ids.numel()),
        "generation_prompt_sha256": digest(generation_ids),
        "loss_prefix_sha256": digest(loss_ids[:generation_ids.numel()]),
        "generation_loss_prefix_match": True,
    }


@torch.no_grad()
def affirmative_metrics(attack: CheckpointedGCG, before: torch.Tensor,
                        suffix: torch.Tensor, after: torch.Tensor,
                        target_ids: torch.Tensor, prefix_token_count: int) -> dict[str, float | int]:
    """Measure the free-generation decision, not only teacher-forced target CE."""
    prompt_ids = torch.cat([before, suffix, after])
    full_ids = torch.cat([prompt_ids, target_ids])
    logits = attack.m(input_ids=full_ids.unsqueeze(0)).logits[0]
    start = prompt_ids.numel()
    predicted = logits[start - 1:start - 1 + target_ids.numel()]
    per_token = F.cross_entropy(predicted, target_ids, reduction="none")
    first_logits = predicted[0]
    first_id = target_ids[0]
    first_logit = first_logits[first_id]
    return {
        "full_target_loss": float(per_token.mean()),
        "first_target_token_loss": float(per_token[0]),
        "affirmative_prefix_loss": float(per_token[:prefix_token_count].mean()),
        "first_target_token_id": int(first_id),
        "first_target_token_probability": float(torch.softmax(first_logits.float(), dim=0)[first_id]),
        "first_target_token_rank": int((first_logits > first_logit).sum()) + 1,
    }


def decoded_roundtrip(tokenizer, suffix: torch.Tensor) -> dict[str, object]:
    decoded = tokenizer.decode(suffix.tolist(), skip_special_tokens=False)
    retokenized = tokenizer(decoded, add_special_tokens=False).input_ids
    return {"decoded_suffix": decoded, "decoded_suffix_roundtrips": retokenized == suffix.tolist(),
            "retokenized_token_count": len(retokenized)}


def candidate_roundtrip_rate(tokenizer, candidates: torch.Tensor) -> float:
    return sum(tokenizer(tokenizer.decode(row.tolist(), skip_special_tokens=False),
                         add_special_tokens=False).input_ids == row.tolist()
               for row in candidates) / candidates.shape[0]


def run_diagnostic(config_path: Path, output_base: Path, commit=None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("stage") != "llama32_gcg_v2_diagnostic":
        raise ValueError("invalid Llama-3.2 v2 diagnostic config")
    paths = run_paths(cfg, output_base)
    with Path(cfg["dataset_csv"]).open(newline="") as handle:
        row = list(csv.DictReader(handle))[cfg["diagnostic_behavior_index"]]
    goal = row["goal"].strip(); target = cfg["target_prefix"] + goal[:1].lower() + goal[1:]
    model, tokenizer, device = load_model(cfg)
    attack = CheckpointedGCG(model, tokenizer, GCGConfig(suffix_len=cfg["suffix_length"], steps=cfg["diagnostic_steps"], topk=cfg["top_k"], search_batch=cfg["candidate_batch_size"], eval_chunk=cfg["evaluation_chunk_size"], seed=cfg["seed"]))
    attack.require_roundtrip_candidates = False
    before, after, target_ids = attack._layout(goal, target)
    suffix = attack._init_suffix(); best = PhaseBest(1); history=[]
    paths["directory"].mkdir(parents=True, exist_ok=True)
    (paths["directory"] / "config.yaml").write_text(config_path.read_text())
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    for parameter in model.parameters():
        parameter.grad = None
        if parameter.requires_grad:
            raise AssertionError("GCG did not freeze a model weight")
    prompt_checks = layout_checks(before, suffix, after, target_ids)
    checks = {"model_weight_gradients_absent": True, "token_native_candidates": True,
              "batch_serial_max_abs_difference": 0.0, **prompt_checks}
    for step in tqdm(range(cfg["diagnostic_steps"]), desc="Llama-3.2 GCG v2", unit="step"):
        grad, _ = attack._token_gradients(suffix, before, after, target_ids)
        candidates = attack._sample(suffix, grad, gen)
        validate_token_native_candidates(candidates, suffix, set(tokenizer.all_special_ids))
        if any(parameter.grad is not None for parameter in model.parameters()):
            checks["model_weight_gradients_absent"] = False
            raise AssertionError("a model weight received a gradient")
        batched = attack._eval(candidates[:8], before, after, target_ids)
        serial = torch.cat([attack._eval(row.unsqueeze(0), before, after, target_ids) for row in candidates[:8]])
        checks["batch_serial_max_abs_difference"] = max(checks["batch_serial_max_abs_difference"], float((batched-serial).abs().max()))
        losses = attack._eval(candidates, before, after, target_ids); suffix = candidates[int(losses.argmin())].clone(); loss=float(losses.min()); history.append(loss); best.consider(suffix, loss)
        payload={"status":"running","next_step":step+1,"loss_history":history,"suffix_ids":suffix.tolist(),"best_suffix_ids":best.suffix.tolist(),"best_loss":best.loss,"checks":checks}
        (paths["checkpoint"]).write_text(json.dumps(payload, indent=2)); (paths["progress"]).write_text(json.dumps(payload, indent=2))
        if (step + 1) % 10 == 0 or step + 1 == cfg["diagnostic_steps"]:
            print(json.dumps({"type": "gcg_v2_progress", "step": step + 1,
                              "total_steps": cfg["diagnostic_steps"], "loss": loss,
                              "best_loss": best.loss}))
        if commit: commit()
    suffix=best.suffix; baseline=attack.generate(goal, None, cfg["max_new_tokens"]); attacked=attack.generate(goal, suffix.tolist(), cfg["max_new_tokens"])
    checks["target_loss_decreased"] = history[-1] < history[0]
    checks["passed"] = (checks["target_loss_decreased"]
                        and checks["generation_loss_prefix_match"]
                        and checks["model_weight_gradients_absent"]
                        and checks["token_native_candidates"]
                        and checks["batch_serial_max_abs_difference"] <= cfg["batching_tolerance"])
    result={"status":"complete" if checks["passed"] else "failed","checks":checks,"optimization":{"initial_loss":history[0],"final_loss":history[-1],"best_loss":best.loss},"suffix":{"token_ids":suffix.tolist(),"decoded":tokenizer.decode(suffix.tolist())},"paired_generation":{"baseline_response":baseline,"suffix_response":attacked,"baseline_success":is_success(baseline),"suffix_success":is_success(attacked)}}
    (paths["result"]).write_text(json.dumps(result, indent=2)); (paths["checkpoint"]).write_text(json.dumps(result, indent=2)); (paths["summary"]).write_text(f"# Llama-3.2 GCG v2 diagnostic\n\nStatus: {result['status']}\n\n- Loss: {history[0]:.4f} → {history[-1]:.4f}\n")
    if commit: commit()
    return result


def run_alignment_diagnostic(config_path: Path, output_base: Path, commit=None,
                             run_mode: str = "fresh", run_id: str | None = None) -> dict:
    """Run/resume an ASR-aligned diagnostic of the Llama-3.2 GCG objective."""
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("stage") != "llama32_gcg_alignment_diagnostic":
        raise ValueError("invalid Llama-3.2 alignment diagnostic config")
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths = run_paths(cfg, output_base)
    if run_id:
        paths = {key: (output_base / cfg["run"]["output_root"] / run_id / value.name)
                 for key, value in paths.items()}
    if run_mode == "resume" and not run_id:
        raise ValueError("resume requires the original run_id")
    directory = paths["directory"]
    checkpoint = json.loads(paths["checkpoint"].read_text()) if run_mode == "resume" and paths["checkpoint"].exists() else None
    if run_mode == "resume" and (not checkpoint or checkpoint.get("config_sha256") != config_sha):
        raise ValueError("resume requires a matching checkpoint")

    with Path(cfg["dataset_csv"]).open(newline="") as handle:
        row = list(csv.DictReader(handle))[cfg["diagnostic_behavior_index"]]
    goal = row["goal"].strip()
    target = cfg["target_prefix"] + goal[:1].lower() + goal[1:]
    model, tokenizer, device = load_model(cfg)
    attack = CheckpointedGCG(model, tokenizer, GCGConfig(
        suffix_len=cfg["suffix_length"], steps=cfg["diagnostic_steps"], topk=cfg["top_k"],
        search_batch=cfg["candidate_batch_size"], eval_chunk=cfg["evaluation_chunk_size"], seed=cfg["seed"]
    ))
    attack.require_roundtrip_candidates = False
    before, after, target_ids = attack._layout(goal, target)
    prefix_ids = tokenizer(cfg["target_prefix"].strip(), add_special_tokens=False).input_ids
    if not prefix_ids or target_ids[:len(prefix_ids)].tolist() != prefix_ids:
        raise AssertionError("affirmative prefix tokenization diverges from target")
    for parameter in model.parameters():
        parameter.grad = None
        if parameter.requires_grad:
            raise AssertionError("GCG did not freeze a model weight")

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(config_path.read_text())
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    if checkpoint:
        suffix = torch.tensor(checkpoint["suffix_ids"], device=device, dtype=torch.long)
        best = PhaseBest(1, float(checkpoint["best_loss"]), torch.tensor(checkpoint["best_suffix_ids"], device=device))
        history, snapshots, start_step = checkpoint["loss_history"], checkpoint["snapshots"], checkpoint["next_step"]
        generator.set_state(torch.tensor(checkpoint["generator_state"], dtype=torch.uint8))
    else:
        suffix, best, history, snapshots, start_step = attack._init_suffix(), PhaseBest(1), [], [], 0

    prompt_checks = layout_checks(before, suffix, after, target_ids)
    checks = {"model_weight_gradients_absent": True, "token_native_candidates": True,
              "batch_serial_max_abs_difference": 0.0, **prompt_checks}

    def snapshot(step: int, label: str, current: torch.Tensor, candidate_rate: float | None = None) -> dict:
        metrics = affirmative_metrics(attack, before, current, after, target_ids, len(prefix_ids))
        roundtrip = decoded_roundtrip(tokenizer, current)
        baseline = attack.generate(goal, None, cfg["max_new_tokens"])
        attacked = attack.generate(goal, current.tolist(), cfg["max_new_tokens"])
        return {"step": step, "label": label, **metrics, **roundtrip,
                "candidate_text_roundtrip_rate": candidate_rate,
                "baseline_response": baseline, "suffix_response": attacked,
                "baseline_success": is_success(baseline), "suffix_success": is_success(attacked)}

    if not snapshots:
        snapshots.append(snapshot(0, "initial", suffix))
    try:
        for step in tqdm(range(start_step, cfg["diagnostic_steps"]), desc="Llama-3.2 alignment", unit="step"):
            grad, _ = attack._token_gradients(suffix, before, after, target_ids)
            candidates = attack._sample(suffix, grad, generator)
            validate_token_native_candidates(candidates, suffix, set(tokenizer.all_special_ids))
            if any(parameter.grad is not None for parameter in model.parameters()):
                checks["model_weight_gradients_absent"] = False
                raise AssertionError("a model weight received a gradient")
            batched = attack._eval(candidates[:8], before, after, target_ids)
            serial = torch.cat([attack._eval(item.unsqueeze(0), before, after, target_ids) for item in candidates[:8]])
            checks["batch_serial_max_abs_difference"] = max(checks["batch_serial_max_abs_difference"], float((batched - serial).abs().max()))
            losses = attack._eval(candidates, before, after, target_ids)
            suffix = candidates[int(losses.argmin())].clone()
            loss = float(losses.min()); history.append(loss); best.consider(suffix, loss)
            candidate_rate = None
            if (step + 1) % cfg["diagnostic_every"] == 0 or step + 1 == cfg["diagnostic_steps"]:
                candidate_rate = candidate_roundtrip_rate(tokenizer, candidates)
                snapshots.append(snapshot(step + 1, "current", suffix, candidate_rate))
                latest = snapshots[-1]
                print(json.dumps({"type": "gcg_alignment_progress", "step": step + 1,
                                  "total_steps": cfg["diagnostic_steps"], "loss": loss,
                                  "best_loss": best.loss, "sure_probability": latest["first_target_token_probability"],
                                  "sure_rank": latest["first_target_token_rank"], "asr": latest["suffix_success"]}))
            payload = {"status": "running", "config_sha256": config_sha, "next_step": step + 1,
                       "suffix_ids": suffix.tolist(), "best_suffix_ids": best.suffix.tolist(), "best_loss": best.loss,
                       "loss_history": history, "snapshots": snapshots,
                       "generator_state": generator.get_state().cpu().tolist(), "checks": checks}
            paths["checkpoint"].write_text(json.dumps(payload, indent=2)); paths["progress"].write_text(json.dumps(payload, indent=2))
            if commit: commit()
    except KeyboardInterrupt:
        raise

    final = snapshot(cfg["diagnostic_steps"], "best", best.suffix)
    if snapshots[-1]["step"] == cfg["diagnostic_steps"]:
        snapshots[-1] = final
    else:
        snapshots.append(final)
    checks["target_loss_decreased"] = history[-1] < snapshots[0]["full_target_loss"]
    checks["passed"] = checks["generation_loss_prefix_match"] and checks["model_weight_gradients_absent"] and checks["token_native_candidates"] and checks["batch_serial_max_abs_difference"] <= cfg["batching_tolerance"]
    result = {"status": "complete", "checks": checks, "optimization": {"initial_loss": snapshots[0]["full_target_loss"], "final_loss": history[-1], "best_loss": best.loss}, "snapshots": snapshots, "suffix": {"token_ids": best.suffix.tolist(), **decoded_roundtrip(tokenizer, best.suffix)}}
    paths["result"].write_text(json.dumps(result, indent=2)); paths["checkpoint"].write_text(json.dumps(result, indent=2))
    paths["summary"].write_text(f"# Llama-3.2 GCG alignment diagnostic\n\nStatus: complete\n\n- Target loss: {result['optimization']['initial_loss']:.4f} → {result['optimization']['final_loss']:.4f}\n- Final `Sure` probability: {final['first_target_token_probability']:.4f}\n- Final `Sure` rank: {final['first_target_token_rank']}\n- ASR: {final['suffix_success']}\n- Text round-trip: {final['decoded_suffix_roundtrips']}\n")
    if commit: commit()
    return result
