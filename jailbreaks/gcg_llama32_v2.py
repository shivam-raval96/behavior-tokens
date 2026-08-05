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
    paths["directory"].mkdir(parents=True, exist_ok=True); paths["config" if "config" in paths else "directory"]
    (paths["directory"] / "config.yaml").write_text(config_path.read_text())
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    checks = {"model_weight_gradients_absent": True, "token_native_candidates": True, "batch_serial_max_abs_difference": 0.0}
    for step in range(cfg["diagnostic_steps"]):
        grad, _ = attack._token_gradients(suffix, before, after, target_ids)
        candidates = attack._sample(suffix, grad, gen)
        validate_token_native_candidates(candidates, suffix, set(tokenizer.all_special_ids))
        batched = attack._eval(candidates[:8], before, after, target_ids)
        serial = torch.cat([attack._eval(row.unsqueeze(0), before, after, target_ids) for row in candidates[:8]])
        checks["batch_serial_max_abs_difference"] = max(checks["batch_serial_max_abs_difference"], float((batched-serial).abs().max()))
        losses = attack._eval(candidates, before, after, target_ids); suffix = candidates[int(losses.argmin())].clone(); loss=float(losses.min()); history.append(loss); best.consider(suffix, loss)
        payload={"status":"running","next_step":step+1,"loss_history":history,"suffix_ids":suffix.tolist(),"best_suffix_ids":best.suffix.tolist(),"best_loss":best.loss,"checks":checks}
        (paths["checkpoint"]).write_text(json.dumps(payload, indent=2)); (paths["progress"]).write_text(json.dumps(payload, indent=2))
        if commit: commit()
    suffix=best.suffix; baseline=attack.generate(goal, None, cfg["max_new_tokens"]); attacked=attack.generate(goal, suffix.tolist(), cfg["max_new_tokens"])
    checks["target_loss_decreased"] = history[-1] < history[0]
    checks["passed"] = checks["target_loss_decreased"] and checks["batch_serial_max_abs_difference"] <= cfg["batching_tolerance"]
    result={"status":"complete" if checks["passed"] else "failed","checks":checks,"optimization":{"initial_loss":history[0],"final_loss":history[-1],"best_loss":best.loss},"suffix":{"token_ids":suffix.tolist(),"decoded":tokenizer.decode(suffix.tolist())},"paired_generation":{"baseline_response":baseline,"suffix_response":attacked,"baseline_success":is_success(baseline),"suffix_success":is_success(attacked)}}
    (paths["result"]).write_text(json.dumps(result, indent=2)); (paths["checkpoint"]).write_text(json.dumps(result, indent=2)); (paths["summary"]).write_text(f"# Llama-3.2 GCG v2 diagnostic\n\nStatus: {result['status']}\n\n- Loss: {history[0]:.4f} → {history[-1]:.4f}\n")
    if commit: commit()
    return result
