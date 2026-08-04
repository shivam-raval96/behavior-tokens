"""Benign Stage A validation for a GCG-style suffix optimizer.

The experiment is controlled by ``jailbreaks/EXPERIMENT.md``. It deliberately
accepts only the harmless prompt/target pairs recorded there and performs no
behavioral jailbreak evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    text = path.read_text()
    start = text.find("```yaml")
    end = text.find("```", start + len("```yaml"))
    if start < 0 or end < 0:
        raise ValueError(f"{path} must contain exactly one YAML control block")
    config = yaml.safe_load(text[start + len("```yaml"):end])
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
    return config


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
        self, prompt: str, target_prefix: str, progress_path: Path | None = None
    ) -> dict[str, Any]:
        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])
        layout = self.layout(prompt, target_prefix)
        suffix = self.initial_suffix()
        initial_suffix = suffix.clone()
        started = time.monotonic()
        slices = self.verify_layout(suffix, layout)
        initial_loss = float(self.evaluate(suffix.unsqueeze(0), layout)[0])
        current_loss = initial_loss
        history: list[float] = []
        invariants: dict[str, Any] = {
            "initial_suffix_roundtrips": True,
            "slice_indices": slices,
            "model_weight_gradients_absent": True,
        }
        progress = tqdm(range(self.config["steps"]), desc="Stage A GCG", unit="step")
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
                print("METRIC " + json.dumps(metric, sort_keys=True), flush=True)
                if progress_path:
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    progress_path.write_text(json.dumps({
                        "status": "running", "prompt": prompt, "target_prefix": target_prefix,
                        "initial_loss": initial_loss, "loss_history": history, "latest_metric": metric,
                    }, indent=2) + "\n")
        invariants["final_suffix_roundtrips"] = self.roundtrips(suffix)
        invariants["target_loss_decreased"] = current_loss < initial_loss
        if not invariants["target_loss_decreased"]:
            raise AssertionError(f"target loss did not decrease ({initial_loss:.6f} -> {current_loss:.6f})")
        return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benign GCG Stage A validation")
    parser.add_argument("--experiment", type=Path, default=Path("jailbreaks/EXPERIMENT.md"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="override output_path from the experiment Markdown")
    parser.add_argument("--progress-output", type=Path, default=None,
                        help="write a partial JSON checkpoint at each progress interval")
    args = parser.parse_args()
    config = read_experiment(args.experiment)
    if not 0 <= args.prompt_index < len(config["prompts"]):
        raise ValueError("--prompt-index is outside the experiment prompt list")
    model, tokenizer, device, dtype = load_model(config)
    row = config["prompts"][args.prompt_index]
    started = time.monotonic()
    result = StageAValidator(model, tokenizer, config).run(
        row["prompt"], row["target_prefix"], args.progress_output
    )
    result["metadata"] = {
        "stage": config["stage"], "model_id": config["model_id"], "revision": config["revision"],
        "device": device, "dtype": str(dtype), "seed": config["seed"],
        "experiment_sha256": hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
        "prompt_index": args.prompt_index, "wall_clock_seconds": round(time.monotonic() - started, 2),
    }
    output = args.output or Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
