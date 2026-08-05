"""Checkpointed small-scale GCG/AdvBench safety evaluation.

This runner uses the Zou et al. search hyperparameters while reducing only the
number of behaviors and optimization steps. Raw benchmark instructions,
generations, and decoded adversarial suffixes remain in process memory and are
never written to the run artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from tqdm.auto import tqdm

from jailbreaks.asr import is_success
from claude_legacy.jailbreaks.gcg_bench.benchmark import LLAMA2_CHAT_TEMPLATE, resolve_device
from claude_legacy.jailbreaks.gcg_bench.datasets_hf import load
from claude_legacy.jailbreaks.gcg_bench.gcg_zou import GCG, GCGConfig, GCGResult, SENTINEL


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("stage") != "stage_b_small_scale_advbench":
        raise ValueError("this runner only accepts a stage_b_small_scale_advbench config")
    if config.get("dataset") != "advbench":
        raise ValueError("the small-scale replication is scoped to AdvBench")
    required = {"suffix_length": 20, "top_k": 256, "candidate_batch_size": 512}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must remain {expected} for the paper-matched search")
    if not 1 <= config.get("n_behaviors", 0) <= 5 or not 20 <= config.get("steps", 0) <= 100:
        raise ValueError("small-scale checks require 1–5 behaviors and 20–100 steps")
    if config.get("run_mode", "fresh") not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    for key in ("progress_every", "checkpoint_every"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    run = config.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("description"), str):
        raise ValueError("run.description is required")
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in run["description"]):
        raise ValueError("run.description must be a lowercase hyphenated slug")
    return config


def run_paths(config: dict[str, Any], output_base: Path = Path()) -> dict[str, Path]:
    run = config["run"]
    run_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ") if run.get("date", "auto") == "auto" else run["date"]
    directory = output_base / run["output_root"] / f"{run_stamp}_{run['description']}"
    return {
        "directory": directory,
        "result": directory / "results.json",
        "summary": directory / "RESULTS.md",
        "checkpoint": directory / "checkpoint.json",
        "progress": directory / "progress.json",
    }


def load_model(config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; keep it in the Modal secret, not this config")
    device = resolve_device(config["device"])
    dtype = DTYPES[config["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], token=token, use_fast=config.get("use_fast_tokenizer", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], token=token, torch_dtype=dtype
    ).to(device).eval()
    return model, tokenizer, device


class CheckpointedGCG(GCG):
    """The legacy paper-matched GCG core with restartable per-step state."""

    def _layout(self, goal: str, target: str):
        """Place the control span at the canonical Llama-2 template boundary."""
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": goal + SENTINEL}],
            tokenize=False, add_generation_prompt=True,
        )
        before_text, after_text = rendered.split(SENTINEL)
        after_text = after_text.removeprefix(" ")
        encode = lambda text: self.tok(text, add_special_tokens=False).input_ids
        before = torch.tensor(encode(before_text), device=self.device)
        after = torch.tensor(encode(after_text), device=self.device)
        target_ids = torch.tensor(encode(target.strip()), device=self.device)
        return before, after, target_ids

    def _init_suffix(self) -> torch.Tensor:
        """Use the paper's tokenization-stable, space-separated control init."""
        initial_control = " ".join(["!"] * self.cfg.suffix_len)
        token_ids = self.tok(initial_control, add_special_tokens=False).input_ids
        if len(token_ids) != self.cfg.suffix_len:
            raise AssertionError("paper control initialization did not preserve suffix length")
        return torch.tensor(token_ids, device=self.device, dtype=torch.long)

    def _sample(self, suffix: torch.Tensor, grad: torch.Tensor,
                generator: torch.Generator) -> torch.Tensor:
        """Sample only tokenizer-stable, non-special, one-token mutations.

        The reference implementation filters decoded controls that equal the
        current control or change token length. Keeping this invariant in token
        space prevents no-op/special-token candidates from silently consuming a
        GCG step. Valid candidates are repeated to retain the configured batch
        size, matching the reference's post-filter batching behavior.
        """
        top_ids = (-grad).topk(self.cfg.topk, dim=1).indices
        batch_size = self.cfg.search_batch
        special_ids = set(self.tok.all_special_ids)
        valid: list[torch.Tensor] = []
        # Some tokenizer vocabularies have a much smaller round-tripping subset
        # near the gradient top-k. Retry with the same seeded generator instead
        # of treating an empty first draw as an optimizer failure.
        for _ in range(8):
            positions = torch.randint(0, suffix.numel(), (batch_size,), device=self.device, generator=generator)
            choices = torch.randint(0, self.cfg.topk, (batch_size,), device=self.device, generator=generator)
            candidates = suffix.unsqueeze(0).repeat(batch_size, 1)
            candidates[torch.arange(batch_size, device=self.device), positions] = top_ids[positions, choices]
            for candidate in candidates:
                if int((candidate != suffix).sum()) != 1:
                    continue
                if any(int(token) in special_ids for token in candidate):
                    continue
                decoded = self.tok.decode(candidate, skip_special_tokens=False)
                encoded = self.tok(decoded, add_special_tokens=False).input_ids
                if not getattr(self, "require_roundtrip_candidates", True) or encoded == candidate.tolist():
                    valid.append(candidate)
            if valid:
                break
        if not valid:
            raise RuntimeError("GCG sampled no valid one-coordinate candidates")
        valid.extend([valid[-1]] * (batch_size - len(valid)))
        result = torch.stack(valid)
        if not torch.all((result != suffix.unsqueeze(0)).sum(dim=1) == 1):
            raise AssertionError("candidate batch contains a non-mutation")
        return result

    def audit_invariants(self, goal: str, target: str) -> dict[str, Any]:
        """Validate the exact token-level GCG path used by Stage C."""
        suffix = self._init_suffix()
        before, after, target_ids = self._layout(goal, target)
        generation_ids = torch.cat([before, suffix, after])
        canonical_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": goal + " " + self.tok.decode(suffix)}],
            add_generation_prompt=True, return_tensors="pt",
        )[0].to(self.device)
        loss_ids = torch.cat([generation_ids, target_ids])
        canonical_loss_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": goal + " " + self.tok.decode(suffix)},
             {"role": "assistant", "content": target}],
            add_generation_prompt=False, return_tensors="pt",
        )[0].to(self.device)
        shared_loss_length = min(loss_ids.numel(), canonical_loss_ids.numel())
        loss_differences = (loss_ids[:shared_loss_length] != canonical_loss_ids[:shared_loss_length]).nonzero(as_tuple=False)
        first_loss_mismatch = int(loss_differences[0]) if loss_differences.numel() else (
            shared_loss_length if loss_ids.numel() != canonical_loss_ids.numel() else None
        )
        shared_length = min(generation_ids.numel(), canonical_ids.numel())
        differing = (generation_ids[:shared_length] != canonical_ids[:shared_length]).nonzero(as_tuple=False)
        first_mismatch = int(differing[0]) if differing.numel() else (
            shared_length if generation_ids.numel() != canonical_ids.numel() else None
        )
        grad, _ = self._token_gradients(suffix, before, after, target_ids)
        candidate_batch = self._sample(suffix, grad, torch.Generator(device=self.device).manual_seed(self.cfg.seed))
        batched = self._eval(candidate_batch[:8], before, after, target_ids)
        serial = torch.cat([self._eval(row.unsqueeze(0), before, after, target_ids) for row in candidate_batch[:8]])
        return {
            "canonical_generation_ids_match": bool(torch.equal(generation_ids, canonical_ids)),
            "canonical_loss_ids_match": bool(torch.equal(loss_ids, canonical_loss_ids[:loss_ids.numel()])),
            "constructed_loss_token_count": int(loss_ids.numel()),
            "canonical_loss_token_count": int(canonical_loss_ids.numel()),
            "first_loss_token_mismatch": first_loss_mismatch,
            "constructed_tokens_near_loss_boundary": loss_ids[max(0, (first_loss_mismatch or 0) - 2):(first_loss_mismatch or 0) + 4].tolist(),
            "canonical_tokens_near_loss_boundary": canonical_loss_ids[max(0, (first_loss_mismatch or 0) - 2):(first_loss_mismatch or 0) + 4].tolist(),
            "constructed_generation_token_count": int(generation_ids.numel()),
            "canonical_generation_token_count": int(canonical_ids.numel()),
            "first_generation_token_mismatch": first_mismatch,
            "constructed_tokens_near_boundary": generation_ids[max(0, (first_mismatch or 0) - 2):(first_mismatch or 0) + 4].tolist(),
            "canonical_tokens_near_boundary": canonical_ids[max(0, (first_mismatch or 0) - 2):(first_mismatch or 0) + 4].tolist(),
            "candidate_count": int(candidate_batch.shape[0]),
            "candidates_change_one_coordinate": bool(torch.all(
                (candidate_batch != suffix.unsqueeze(0)).sum(dim=1) == 1
            )),
            "suffix_roundtrips": self.tok(self.tok.decode(suffix), add_special_tokens=False).input_ids == suffix.tolist(),
            "target_loss_batch_serial_max_abs_difference": float((batched - serial).abs().max()),
            "model_weight_gradients_absent": not any(parameter.grad is not None for parameter in self.m.parameters()),
            "target_token_count": int(target_ids.numel()),
        }

    def optimize(
        self,
        goal: str,
        target: str,
        state: dict[str, Any] | None,
        save_state: Callable[[dict[str, Any]], None],
        log: Callable[[str], None],
    ) -> GCGResult:
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        before, after, target_ids = self._layout(goal, target)
        if state:
            suffix = torch.tensor(state["suffix_ids"], device=self.device, dtype=torch.long)
            history = list(state["loss_history"])
            start_step = int(state["next_step"])
            generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
            initial_loss = float(state["initial_loss"])
        else:
            suffix = self._init_suffix()
            history = []
            start_step = 0
            initial_loss = float(self._eval(suffix.unsqueeze(0), before, after, target_ids)[0])

        safe = {
            "suffix_ids": suffix.tolist(), "loss_history": list(history), "next_step": start_step,
            "generator_state": generator.get_state().cpu().tolist(), "initial_loss": initial_loss,
        }
        try:
            for step in tqdm(range(start_step, self.cfg.steps), desc="Stage B GCG", unit="step"):
                gradient, _ = self._token_gradients(suffix, before, after, target_ids)
                candidates = self._sample(suffix, gradient, generator)
                losses = self._eval(candidates, before, after, target_ids)
                choice = int(losses.argmin())
                suffix = candidates[choice].clone()
                current_loss = float(losses[choice])
                history.append(current_loss)
                safe = {
                    "suffix_ids": suffix.tolist(), "loss_history": list(history), "next_step": step + 1,
                    "generator_state": generator.get_state().cpu().tolist(), "initial_loss": initial_loss,
                }
                if (step + 1) % self.cfg.progress_every == 0 or step + 1 == self.cfg.steps:
                    log(json.dumps({"type": "gcg_progress", "step": step + 1,
                                    "total_steps": self.cfg.steps, "loss": current_loss}))
                if (step + 1) % self.cfg.checkpoint_every == 0 or step + 1 == self.cfg.steps:
                    save_state(safe)
        except KeyboardInterrupt:
            save_state(safe)
            raise
        return GCGResult(goal=goal, target=target, suffix=self.tok.decode(suffix),
                         suffix_ids=suffix.tolist(), loss=history[-1], loss_history=history), initial_loss

    @torch.no_grad()
    def generate(self, goal: str, suffix_ids: list[int] | None, max_new_tokens: int = 128) -> str:
        """Greedy generation with an explicit mask for Llama's pad/EOS collision."""
        before, after, _ = self._layout(goal, target="x")
        parts = [before]
        if suffix_ids:
            parts.append(torch.tensor(suffix_ids, device=self.device))
        parts.append(after)
        input_ids = torch.cat(parts).unsqueeze(0)
        generation_config = copy.deepcopy(self.m.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        output_ids = self.m.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, generation_config=generation_config,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage B small-scale GCG results", "", "Status: " + payload["status"], "",
        "Raw AdvBench instructions, generated completions, and suffix text are intentionally omitted.", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| Behaviors evaluated | {payload['summary']['n']} |",
        f"| Baseline ASR | {payload['summary']['baseline_asr']:.4f} |",
        f"| GCG-suffix ASR | {payload['summary']['attacked_asr']:.4f} |",
        f"| ASR change | {payload['summary']['asr_delta']:+.4f} |", "",
        "| Behavior ID | Initial loss | Final loss | Loss decreased | Baseline ASR flag | GCG ASR flag |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["behaviors"]:
        lines.append(
            f"| {row['behavior_id']} | {row['initial_loss']:.6f} | {row['final_loss']:.6f} | "
            f"{row['loss_decreased']} | {row['baseline_success']} | {row['attacked_success']} |"
        )
    return "\n".join(lines) + "\n"


def run(config_path: Path, output_base: Path, checkpoint_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    config = read_config(config_path)
    paths = run_paths(config, output_base)
    mode = config.get("run_mode", "fresh")
    checkpoint = json.loads(paths["checkpoint"].read_text()) if mode == "resume" and paths["checkpoint"].exists() else None
    if mode == "resume" and checkpoint is None:
        raise FileNotFoundError("cannot resume: checkpoint.json does not exist")
    if checkpoint and checkpoint.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("checkpoint config does not match the current YAML")

    model, tokenizer, device = load_model(config)
    gcg_config = GCGConfig(suffix_len=config["suffix_length"], steps=config["steps"], topk=config["top_k"],
                           search_batch=config["candidate_batch_size"], eval_chunk=config["evaluation_chunk_size"],
                           seed=config["seed"])
    gcg_config.progress_every = config["progress_every"]
    gcg_config.checkpoint_every = config["checkpoint_every"]
    attack = CheckpointedGCG(model, tokenizer, gcg_config)
    records = load("advbench", n=config["n_behaviors"], seed=config["seed"])
    rows = list(checkpoint.get("behaviors", [])) if checkpoint else []
    started = time.monotonic()

    def save(current: dict[str, Any] | None = None, status: str = "running") -> None:
        base = [row["baseline_success"] for row in rows]
        attacked = [row["attacked_success"] for row in rows]
        baseline_asr = sum(base) / len(base) if base else 0.0
        attacked_asr = sum(attacked) / len(attacked) if attacked else 0.0
        payload = {
            "status": status, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "metadata": {"model_id": config["model_id"], "device": device, "run_mode": mode,
                         "steps": config["steps"], "suffix_length": config["suffix_length"],
                         "top_k": config["top_k"], "candidate_batch_size": config["candidate_batch_size"]},
            "next_behavior": len(rows), "current_optimization": current,
            "behaviors": rows,
            "summary": {"n": len(rows), "baseline_asr": baseline_asr,
                        "attacked_asr": attacked_asr,
                        "asr_delta": attacked_asr - baseline_asr},
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        _write_json(paths["checkpoint"], payload)
        _write_json(paths["progress"], payload)
        if checkpoint_callback:
            checkpoint_callback()

    start_index = len(rows)
    active_state = checkpoint.get("current_optimization") if checkpoint else None
    try:
        for index in range(start_index, len(records)):
            record = records[index]
            behavior_id = hashlib.sha256(record["goal"].encode()).hexdigest()[:12]

            def save_current(state: dict[str, Any]) -> None:
                nonlocal active_state
                active_state = state
                save({"behavior_id": behavior_id, **state})

            result, initial_loss = attack.optimize(record["goal"], record["target"], active_state,
                                                    save_current, print)
            attacked = attack.generate(record["goal"], result.suffix_ids, config["max_new_tokens"])
            baseline = attack.generate(record["goal"], None, config["max_new_tokens"])
            rows.append({"behavior_id": behavior_id, "initial_loss": initial_loss, "final_loss": result.loss,
                         "loss_decreased": result.loss < initial_loss,
                         "baseline_success": is_success(baseline), "attacked_success": is_success(attacked)})
            active_state = None
            save()
    except KeyboardInterrupt:
        save(active_state, status="stopped")
        raise

    save(status="complete")
    payload = json.loads(paths["checkpoint"].read_text())
    _write_json(paths["result"], payload)
    paths["summary"].write_text(_summary(payload))
    if checkpoint_callback:
        checkpoint_callback()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("jailbreaks/configs/stage_b_small_scale_advbench.yaml"))
    parser.add_argument("--output-base", type=Path, default=Path())
    args = parser.parse_args()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signal, _frame: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        run(args.config, args.output_base)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
