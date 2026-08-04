"""Benign Stage A validation for a GCG-style suffix optimizer.

The experiment is controlled by a YAML file in ``jailbreaks/configs/``. It deliberately
accepts only the harmless prompt/target pairs recorded there and performs no
behavioral jailbreak evaluation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date
import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm


SENTINEL = "<|stage_a_control|>"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class Layout:
    before: torch.Tensor
    after: torch.Tensor
    target: torch.Tensor

    def slices(self, suffix_length: int) -> dict[str, list[int]]:
        control_start = int(self.before.numel())
        control_end = control_start + suffix_length
        target_start = control_end + int(self.after.numel())
        target_end = target_start + int(self.target.numel())
        return {
            "prompt": [0, control_start],
            "control": [control_start, control_end],
            "target": [target_start, target_end],
            "loss_logits": [target_start - 1, target_end - 1],
        }


def read_experiment(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("experiment YAML must be a mapping")
    if config.get("stage") != "stage_a_benign_validation":
        raise ValueError("this runner only accepts stage_a_benign_validation")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not 10 <= len(prompts) <= 20:
        raise ValueError("Stage A requires 10–20 prompt/target pairs")
    for row in prompts:
        if not isinstance(row, dict) or not row.get("prompt") or not row.get("target_prefix"):
            raise ValueError("every prompt row needs prompt and target_prefix")
    if config.get("suffix_length") != 5 or not 20 <= config.get("steps", 0) <= 50:
        raise ValueError("Stage A is fixed to a 5-token suffix and 20–50 steps")
    if config.get("run_mode", "fresh") not in {"fresh", "resume"}:
        raise ValueError("run_mode must be either fresh or resume")
    for key in ("progress_every", "checkpoint_every"):
        if not isinstance(config.get(key, 5), int) or config.get(key, 5) < 1:
            raise ValueError(f"{key} must be a positive integer")
    run = config.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("description"), str):
        raise ValueError("run.description is required")
    if not run["description"] or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in run["description"]):
        raise ValueError("run.description must be a lowercase hyphenated slug")
    if not isinstance(run.get("output_root"), str) or not run["output_root"]:
        raise ValueError("run.output_root is required")
    run_date = run.get("date", "auto")
    if run_date != "auto":
        try:
            date.fromisoformat(run_date)
        except (TypeError, ValueError) as error:
            raise ValueError("run.date must be auto or YYYY-MM-DD") from error
    return config


def resolve_run_paths(config: dict[str, Any], output_base: Path = Path()) -> dict[str, Path]:
    """Return one dated, descriptive run directory and its durable artifacts."""
    run = config["run"]
    run_date = date.today().isoformat() if run.get("date", "auto") == "auto" else run["date"]
    directory = output_base / run["output_root"] / f"{run_date}_{run['description']}"
    return {
        "directory": directory,
        "result": directory / "results.json",
        "progress": directory / "progress.json",
        "checkpoint": directory / "checkpoint.json",
        "summary": directory / "RESULTS.md",
    }


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for the gated Llama-2 model. Set it in the environment "
            "and rerun; do not add the token to EXPERIMENT.md or commit it."
        )
    device = resolve_device(config["device"])
    dtype = DTYPES[config["dtype"]]
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], token=token, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], token=token, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer, device, dtype


class StageAValidator:
    """A frozen-model GCG loop with explicit, testable Stage A invariants."""

    def __init__(self, model, tokenizer, config: dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        self.embedding = model.get_input_embeddings()
        self.vocab_size = self.embedding.weight.shape[0]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def layout(self, prompt: str, target_prefix: str) -> Layout:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt} {SENTINEL}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.count(SENTINEL) != 1:
            raise ValueError("chat template did not preserve the single control sentinel")
        before_text, after_text = rendered.split(SENTINEL)
        return Layout(before=self._encode(before_text), after=self._encode(after_text),
                      target=self._encode(" " + target_prefix.strip()))

    def initial_suffix(self) -> torch.Tensor:
        initial_text = self.config["initial_suffix_text"]
        if initial_text == "auto":
            suffix = self._auto_initial_suffix()
        else:
            suffix = self._encode(initial_text)
        if suffix.numel() != self.config["suffix_length"]:
            raise ValueError("initial_suffix_text must tokenize to suffix_length tokens, or be auto")
        if not self.roundtrips(suffix):
            raise ValueError("initial_suffix_text must decode and re-encode identically")
        return suffix

    def _auto_initial_suffix(self) -> torch.Tensor:
        """Pick a deterministic printable, non-special token with a stable decode."""
        special_ids = set(self.tokenizer.all_special_ids)
        for token_id in range(self.vocab_size):
            if token_id in special_ids:
                continue
            suffix = torch.full(
                (self.config["suffix_length"],), token_id, device=self.device, dtype=torch.long
            )
            decoded = self.tokenizer.decode(suffix.tolist(), clean_up_tokenization_spaces=False)
            if decoded.strip() and decoded.isprintable() and self.roundtrips(suffix):
                return suffix
        raise RuntimeError("could not find a tokenizer-stable initial suffix")

    def roundtrips(self, suffix: torch.Tensor) -> bool:
        decoded = self.tokenizer.decode(suffix.tolist(), clean_up_tokenization_spaces=False)
        encoded = self.tokenizer(decoded, add_special_tokens=False).input_ids
        return encoded == suffix.tolist()

    def _logit_slice(self, layout: Layout) -> slice:
        start = layout.before.numel() + self.config["suffix_length"] + layout.after.numel()
        return slice(start - 1, start - 1 + layout.target.numel())

    def target_loss(self, logits: torch.Tensor, layout: Layout) -> torch.Tensor:
        token_logits = logits[..., self._logit_slice(layout), :]
        targets = layout.target.expand(token_logits.shape[0], -1)
        return F.cross_entropy(
            token_logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none"
        ).view(token_logits.shape[0], -1).mean(dim=1)

    def gradient(self, suffix: torch.Tensor, layout: Layout) -> tuple[torch.Tensor, float]:
        one_hot = torch.zeros(
            self.config["suffix_length"], self.vocab_size,
            device=self.device, dtype=self.embedding.weight.dtype,
        )
        one_hot.scatter_(1, suffix.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_embeddings = one_hot @ self.embedding.weight
        sequence = torch.cat(
            [self.embedding(layout.before), suffix_embeddings, self.embedding(layout.after),
             self.embedding(layout.target)]
        ).unsqueeze(0)
        logits = self.model(inputs_embeds=sequence).logits
        loss = self.target_loss(logits, layout).mean()
        gradient = torch.autograd.grad(loss, one_hot, only_inputs=True)[0]
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise AssertionError("model weights received gradients")
        return gradient.detach(), float(loss.detach())

    def sample_candidates(
        self, suffix: torch.Tensor, gradient: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        top_ids = (-gradient).topk(self.config["top_k"], dim=1).indices
        batch_size = self.config["candidate_batch_size"]
        positions = torch.randint(0, suffix.numel(), (batch_size,), device=self.device, generator=generator)
        choices = torch.randint(0, self.config["top_k"], (batch_size,), device=self.device, generator=generator)
        candidates = suffix.unsqueeze(0).repeat(batch_size, 1)
        candidates[torch.arange(batch_size, device=self.device), positions] = top_ids[positions, choices]
        changed = (candidates != suffix.unsqueeze(0)).sum(dim=1)
        candidates = candidates[changed == 1]
        special_ids = set(self.tokenizer.all_special_ids)
        stable = [
            row for row in candidates
            if not any(int(token_id) in special_ids for token_id in row) and self.roundtrips(row)
        ]
        if not stable:
            raise RuntimeError("no valid one-coordinate, tokenizer-stable candidates were sampled")
        candidates = torch.stack(stable)
        if not torch.all((candidates != suffix.unsqueeze(0)).sum(dim=1) == 1):
            raise AssertionError("candidate batch changed more than one coordinate")
        return candidates

    @torch.no_grad()
    def evaluate(self, candidates: torch.Tensor, layout: Layout) -> torch.Tensor:
        fixed_before = self.embedding(layout.before)
        fixed_after = self.embedding(layout.after)
        fixed_target = self.embedding(layout.target)
        losses = []
        for start in range(0, candidates.shape[0], self.config["evaluation_chunk_size"]):
            chunk = candidates[start:start + self.config["evaluation_chunk_size"]]
            size = chunk.shape[0]
            sequence = torch.cat(
                [fixed_before.unsqueeze(0).expand(size, -1, -1), self.embedding(chunk),
                 fixed_after.unsqueeze(0).expand(size, -1, -1),
                 fixed_target.unsqueeze(0).expand(size, -1, -1)],
                dim=1,
            )
            losses.append(self.target_loss(self.model(inputs_embeds=sequence).logits, layout))
        return torch.cat(losses)

    def verify_layout(self, suffix: torch.Tensor, layout: Layout) -> dict[str, list[int]]:
        sequence_ids = torch.cat([layout.before, suffix, layout.after, layout.target])
        slices = layout.slices(suffix.numel())
        assert torch.equal(sequence_ids[slices["control"][0]:slices["control"][1]], suffix)
        assert torch.equal(sequence_ids[slices["target"][0]:slices["target"][1]], layout.target)
        assert slices["loss_logits"][1] - slices["loss_logits"][0] == layout.target.numel()
        return slices

    def verify_batching(self, candidates: torch.Tensor, layout: Layout) -> float:
        subset = candidates[:min(8, candidates.shape[0])]
        batched = self.evaluate(subset, layout)
        serial = torch.cat([self.evaluate(row.unsqueeze(0), layout) for row in subset])
        maximum_difference = float((batched - serial).abs().max())
        if not torch.allclose(batched, serial, rtol=2e-3, atol=2e-3):
            raise AssertionError(f"batched/serial target losses differ by {maximum_difference}")
        return maximum_difference

    def run(
        self,
        prompt: str,
        target_prefix: str,
        *,
        checkpoint_path: Path | None = None,
        progress_path: Path | None = None,
        run_mode: str = "fresh",
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])
        layout = self.layout(prompt, target_prefix)
        started = time.monotonic()
        if run_mode == "resume":
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError("cannot resume: no Stage A checkpoint was found")
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("status") == "complete":
                raise ValueError("checkpoint is already complete; set run_mode: fresh for a new run")
            if checkpoint.get("prompt") != prompt or checkpoint.get("target_prefix") != target_prefix:
                raise ValueError("checkpoint prompt does not match the selected experiment prompt")
            initial_suffix = torch.tensor(
                checkpoint["initial_suffix_token_ids"], device=self.device, dtype=torch.long
            )
            suffix = torch.tensor(checkpoint["suffix_token_ids"], device=self.device, dtype=torch.long)
            if suffix.numel() != self.config["suffix_length"]:
                raise ValueError("checkpoint suffix length does not match the config file")
            if not self.roundtrips(suffix):
                raise ValueError("checkpoint suffix no longer round-trips through the tokenizer")
            generator.set_state(torch.tensor(checkpoint["generator_state"], dtype=torch.uint8))
            initial_loss = float(checkpoint["initial_loss"])
            current_loss = float(checkpoint["current_loss"])
            history = list(checkpoint["loss_history"])
            invariants = dict(checkpoint["invariants"])
            start_step = int(checkpoint["next_step"])
            if not 0 <= start_step < self.config["steps"]:
                raise ValueError("checkpoint has no remaining steps; set run_mode: fresh for a new run")
            slices = self.verify_layout(suffix, layout)
            if invariants.get("slice_indices") != slices:
                raise ValueError("checkpoint slice layout does not match the config file")
            print(f"Resuming Stage A from step {start_step}/{self.config['steps']}", flush=True)
        else:
            suffix = self.initial_suffix()
            initial_suffix = suffix.clone()
            slices = self.verify_layout(suffix, layout)
            initial_loss = float(self.evaluate(suffix.unsqueeze(0), layout)[0])
            current_loss = initial_loss
            history = []
            invariants = {
                "initial_suffix_roundtrips": True,
                "slice_indices": slices,
                "model_weight_gradients_absent": True,
            }
            start_step = 0

        def save_checkpoint(status: str, next_step: int, metric: dict[str, Any] | None = None) -> None:
            if checkpoint_path is None:
                return
            payload = {
                "status": status,
                "next_step": next_step,
                "total_steps": self.config["steps"],
                "prompt": prompt,
                "target_prefix": target_prefix,
                "initial_suffix_token_ids": initial_suffix.tolist(),
                "suffix_token_ids": suffix.tolist(),
                "initial_loss": initial_loss,
                "current_loss": current_loss,
                "loss_history": history,
                "invariants": invariants,
                "generator_state": generator.get_state().cpu().tolist(),
                "latest_metric": metric,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(payload, indent=2) + "\n")
            if checkpoint_callback:
                checkpoint_callback()

        def save_progress(metric: dict[str, Any], status: str = "running") -> None:
            if progress_path is None:
                return
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({
                "status": status, "prompt": prompt, "target_prefix": target_prefix,
                "initial_loss": initial_loss, "current_loss": current_loss,
                "loss_history": history, "latest_metric": metric,
            }, indent=2) + "\n")

        save_checkpoint("running", start_step)
        progress = tqdm(range(start_step, self.config["steps"]), desc="Stage A GCG", unit="step")
        latest_metric: dict[str, Any] | None = None
        # These snapshots make an asynchronous stop restart at the beginning of
        # the last incomplete step, including its exact candidate RNG state.
        safe_suffix = suffix.clone()
        safe_current_loss = current_loss
        safe_history = list(history)
        safe_invariants = copy.deepcopy(invariants)
        safe_next_step = start_step
        safe_generator_state = generator.get_state().clone()
        try:
            for step in progress:
                gradient, gradient_loss = self.gradient(suffix, layout)
                candidates = self.sample_candidates(suffix, gradient, generator)
                if step == 0:
                    invariants["candidate_batch_one_coordinate"] = True
                    invariants["candidate_batch_roundtrips"] = True
                    invariants["batch_serial_max_abs_difference"] = self.verify_batching(candidates, layout)
                candidate_losses = self.evaluate(candidates, layout)
                best_index = int(candidate_losses.argmin())
                best_loss = float(candidate_losses[best_index])
                if best_loss < current_loss:
                    suffix, current_loss = candidates[best_index].clone(), best_loss
                history.append(current_loss)
                safe_suffix = suffix.clone()
                safe_current_loss = current_loss
                safe_history = list(history)
                safe_invariants = copy.deepcopy(invariants)
                safe_next_step = step + 1
                safe_generator_state = generator.get_state().clone()
                progress.set_postfix(loss=f"{current_loss:.4f}", candidates=candidates.shape[0])
                if (step + 1) % self.config["progress_every"] == 0 or step + 1 == self.config["steps"]:
                    metric = {
                        "type": "stage_a_progress",
                        "step": step + 1,
                        "total_steps": self.config["steps"],
                        "current_loss": current_loss,
                        "gradient_loss": gradient_loss,
                        "candidate_count": int(candidates.shape[0]),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                    latest_metric = metric
                    print("METRIC " + json.dumps(metric, sort_keys=True), flush=True)
                    save_progress(metric)
                if (step + 1) % self.config["checkpoint_every"] == 0 or step + 1 == self.config["steps"]:
                    save_checkpoint("running", step + 1, latest_metric)
        except KeyboardInterrupt:
            suffix = safe_suffix
            current_loss = safe_current_loss
            history = safe_history
            invariants = safe_invariants
            generator.set_state(safe_generator_state)
            metric = {
                "type": "stage_a_stopped", "step": safe_next_step,
                "total_steps": self.config["steps"], "current_loss": current_loss,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            print("METRIC " + json.dumps(metric, sort_keys=True), flush=True)
            save_progress(metric, status="stopped")
            save_checkpoint("stopped", safe_next_step, metric)
            raise
        invariants["final_suffix_roundtrips"] = self.roundtrips(suffix)
        invariants["target_loss_decreased"] = current_loss < initial_loss
        if not invariants["target_loss_decreased"]:
            raise AssertionError(f"target loss did not decrease ({initial_loss:.6f} -> {current_loss:.6f})")
        result = {
            "prompt": prompt,
            "target_prefix": target_prefix,
            "initial_suffix_token_ids": initial_suffix.tolist(),
            "initial_suffix_decoded": self.tokenizer.decode(
                initial_suffix.tolist(), clean_up_tokenization_spaces=False
            ),
            "initial_loss": initial_loss,
            "final_loss": current_loss,
            "loss_history": history,
            "final_suffix_token_ids": suffix.tolist(),
            "final_suffix_decoded": self.tokenizer.decode(suffix.tolist(), clean_up_tokenization_spaces=False),
            "invariants": invariants,
        }
        save_checkpoint("complete", self.config["steps"])
        return result


def run_experiment(
    experiment: Path,
    prompt_index: int,
    output: Path | None = None,
    progress_output: Path | None = None,
    checkpoint_output: Path | None = None,
    run_mode: str | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = read_experiment(experiment)
    if not 0 <= prompt_index < len(config["prompts"]):
        raise ValueError("--prompt-index is outside the experiment prompt list")
    mode = run_mode or config.get("run_mode", "fresh")
    if mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be either fresh or resume")
    model, tokenizer, device, dtype = load_model(config)
    row = config["prompts"][prompt_index]
    started = time.monotonic()
    result = StageAValidator(model, tokenizer, config).run(
        row["prompt"], row["target_prefix"], checkpoint_path=checkpoint_output,
        progress_path=progress_output, run_mode=mode, checkpoint_callback=checkpoint_callback,
    )
    result["metadata"] = {
        "stage": config["stage"], "model_id": config["model_id"], "revision": config["revision"],
        "device": device, "dtype": str(dtype), "seed": config["seed"],
        "run_mode": mode,
        "experiment_sha256": hashlib.sha256(experiment.read_bytes()).hexdigest(),
        "prompt_index": prompt_index, "wall_clock_seconds": round(time.monotonic() - started, 2),
    }
    paths = resolve_run_paths(config)
    destination = output or paths["result"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    write_results_summary(destination.parent / "RESULTS.md", result)
    if checkpoint_callback:
        checkpoint_callback()
    return result


def write_results_summary(path: Path, result: dict[str, Any]) -> None:
    """Write the compact, human-readable companion to a completed results.json."""
    metadata = result["metadata"]
    invariants = result["invariants"]
    path.write_text(
        "# Stage A results\n\n"
        "Status: completed\n\n"
        f"- Model: `{metadata['model_id']}` (`{metadata['revision']}`)\n"
        f"- Prompt index: {metadata['prompt_index']}\n"
        f"- Duration: {metadata['wall_clock_seconds']} seconds\n"
        f"- Initial loss: {result['initial_loss']:.6f}\n"
        f"- Final loss: {result['final_loss']:.6f}\n"
        f"- Final suffix: `{result['final_suffix_decoded']}`\n\n"
        "| Validation | Result |\n"
        "| --- | --- |\n"
        f"| Target loss decreased | {invariants['target_loss_decreased']} |\n"
        f"| Initial/final suffix round-trips | {invariants['initial_suffix_roundtrips']} / {invariants['final_suffix_roundtrips']} |\n"
        f"| One-coordinate candidates | {invariants['candidate_batch_one_coordinate']} |\n"
        f"| Model weights frozen | {invariants['model_weight_gradients_absent']} |\n"
        f"| Batch/serial max loss difference | {invariants['batch_serial_max_abs_difference']:.6f} |\n\n"
        "See `results.json` for the full loss history and experiment metadata.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benign GCG Stage A validation")
    parser.add_argument("--experiment", type=Path,
                        default=Path("jailbreaks/configs/stage_a_benign_llama2_7b_chat.yaml"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="override the dated run directory results.json path")
    parser.add_argument("--progress-output", type=Path, default=None,
                        help="write a partial JSON checkpoint at each progress interval")
    parser.add_argument("--checkpoint-output", type=Path, default=None,
                        help="durable resume checkpoint; defaults to the dated run directory")
    parser.add_argument("--run-mode", choices=("fresh", "resume"), default=None,
                        help="override run_mode in the YAML config")
    args = parser.parse_args()
    config = read_experiment(args.experiment)
    paths = resolve_run_paths(config)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop_at_checkpoint(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_at_checkpoint)
    try:
        result = run_experiment(
            args.experiment, args.prompt_index, args.output or paths["result"],
            args.progress_output or paths["progress"], args.checkpoint_output or paths["checkpoint"],
            args.run_mode,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(f"wrote {args.output or paths['result']}")


if __name__ == "__main__":
    main()
