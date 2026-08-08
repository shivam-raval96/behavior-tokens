from __future__ import annotations

import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import file_sha256
from .io_utils import ArtifactWriter
from .layout import build_layout
from .prefill_probe import (
    diagnostic,
    forward_trajectory,
    greedy_continuation,
    repeated_token_suffix,
)


def fingerprint(config: dict[str, Any]) -> str:
    scientific = {key: value for key, value in config.items() if key != "run_mode"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def full_forward_kl(
    student_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    student_logp = student_logits.log_softmax(-1)
    target_logp = target_logits.log_softmax(-1)
    return (target_logp.exp() * (target_logp - student_logp)).sum(-1)


def combined_objective(
    student_hidden: torch.Tensor,
    student_logits: torch.Tensor,
    target_hidden: torch.Tensor,
    target_logits: torch.Tensor,
    baseline_hidden: torch.Tensor,
    vector: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    baseline_mse = (
        torch.nn.functional.mse_loss(baseline_hidden, target_hidden)
        .detach()
        .clamp_min(1e-12)
    )
    activation = (
        torch.nn.functional.mse_loss(student_hidden, target_hidden) / baseline_mse
    )
    student_delta = student_hidden - baseline_hidden
    target_delta = target_hidden - baseline_hidden
    student_projection = torch.einsum("td,d->t", student_delta, vector.float())
    target_projection = torch.einsum("td,d->t", target_delta, vector.float())
    projection_scale = target_projection.square().mean().detach().clamp_min(1e-12)
    projection = (
        torch.nn.functional.mse_loss(student_projection, target_projection)
        / projection_scale
    )
    per_position_kl = full_forward_kl(student_logits, target_logits)
    position_0_fraction = float(weights["position_0_fraction"])
    weighted_kl = (
        position_0_fraction * per_position_kl[0]
        + (1.0 - position_0_fraction) * per_position_kl[1:].mean()
    )
    total = (
        float(weights["normalized_activation_weight"]) * activation
        + float(weights["normalized_projection_weight"]) * projection
        + float(weights["position_weighted_kl_weight"]) * weighted_kl
    )
    return total, {
        "normalized_activation_loss": activation,
        "normalized_projection_loss": projection,
        "position_weighted_kl": weighted_kl,
        "position_0_kl": per_position_kl[0],
        "later_position_kl": per_position_kl[1:].mean(),
    }


def plot_history(output: Path) -> None:
    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    steps = [row["completed"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(steps, [row["activation_cosine"] for row in rows], marker="o")
    axes[0].axhline(0.8, color="gray", linestyle="--")
    axes[0].set(xlabel="step", ylabel="activation cosine")
    axes[1].plot(steps, [row["position_0_kl"] for row in rows], marker="o")
    axes[1].axhline(5.0, color="gray", linestyle="--")
    axes[1].set(xlabel="step", ylabel="position-0 KL", yscale="log")
    axes[2].plot(steps, [row["mean_forward_kl"] for row in rows], marker="o")
    axes[2].set(xlabel="step", ylabel="mean forward KL", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "combined_prefill.png", dpi=180)
    plt.close(figure)


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    config_fingerprint = fingerprint(config)
    writer = ArtifactWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    if mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError("fresh run refuses an existing checkpoint")
    checkpoint = (
        json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    )
    if mode == "resume" and checkpoint.get("config_fingerprint") != config_fingerprint:
        raise ValueError("resume config fingerprint mismatch")
    source = Path(config["source_run_dir"])
    source_resolved = json.loads((source / "resolved_config.json").read_text())
    if source_resolved["config_fingerprint"] != config["source_config_fingerprint"]:
        raise ValueError("source run fingerprint mismatch")
    source_cache = source / config["source_teacher_cache"]
    source_suffix = source / config["source_initial_suffix"]
    for path, expected in (
        (source_cache, config["source_teacher_cache_sha256"]),
        (source_suffix, config["source_initial_suffix_sha256"]),
        (Path(config["vector_path"]), config["vector_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {path}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    config.update(run_id=output.name, config_fingerprint=config_fingerprint)
    writer.json("resolved_config.json", config)
    writer.json(
        "source_artifacts.json",
        {
            "source_run_id": config["source_run_id"],
            "source_config_fingerprint": config["source_config_fingerprint"],
            "teacher_cache": config["source_teacher_cache"],
            "initial_suffix": config["source_initial_suffix"],
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vector = torch.tensor(
        np.load(config["vector_path"]), device=model.device, dtype=torch.float32
    )
    vector /= vector.norm()
    selection = json.loads((source / "selection.json").read_text())
    prompt = selection["prompt"]
    layout = build_layout(tokenizer, prompt, config["suffix_length"])
    filler, filler_ids = repeated_token_suffix(
        model, tokenizer, config["filler_suffix_text"], config["suffix_length"]
    )
    cached = torch.load(source_cache, map_location=model.device, weights_only=False)
    continuation = cached["continuation_ids"].cpu()
    target_hidden = cached["target_hidden"].to(model.device).float()
    target_logits = cached["target_logits"].to(model.device).float()
    baseline_hidden = cached["baseline_hidden"].to(model.device).float()
    suffix = torch.nn.Parameter(
        torch.load(source_suffix, map_location=model.device, weights_only=False).float()
    )
    if suffix.shape[0] != config["suffix_length"]:
        raise ValueError("source suffix length mismatch")
    optimizer = torch.optim.Adam([suffix], lr=config["learning_rate"])
    start_step = 0
    best = {"objective": float("inf"), "step": 0, "suffix": suffix.detach().cpu()}
    state_dir = output / "optimizer_state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / "combined.pt"
    if mode == "resume" and state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        suffix.data.copy_(state["suffix"].to(model.device))
        optimizer.load_state_dict(state["optimizer"])
        best = state["best"]
        start_step = int(state["next_step"])

    stopped = False

    def terminate(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    for step in tqdm(
        range(start_step, config["steps"]),
        initial=start_step,
        total=config["steps"],
        desc="combined_prefill",
        unit="step",
    ):
        optimizer.zero_grad(set_to_none=True)
        student_hidden, student_logits = forward_trajectory(
            model, layout, suffix, continuation, config["hidden_state_index"]
        )
        loss, components = combined_objective(
            student_hidden,
            student_logits,
            target_hidden,
            target_logits,
            baseline_hidden,
            vector,
            config["loss"],
        )
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % config["diagnostic_every"] == 0:
            with torch.no_grad():
                current_hidden, current_logits = forward_trajectory(
                    model, layout, suffix, continuation, config["hidden_state_index"]
                )
                objective, current_components = combined_objective(
                    current_hidden,
                    current_logits,
                    target_hidden,
                    target_logits,
                    baseline_hidden,
                    vector,
                    config["loss"],
                )
                metrics = diagnostic(
                    current_hidden,
                    current_logits,
                    target_hidden,
                    target_logits,
                    baseline_hidden,
                    vector,
                )
            if float(objective) < best["objective"]:
                best = {
                    "objective": float(objective),
                    "step": step + 1,
                    "suffix": suffix.detach().cpu().clone(),
                }
            elapsed = time.monotonic() - started
            row = {
                "phase": "combined_prefill",
                "completed": step + 1,
                "total": config["steps"],
                "elapsed_seconds": elapsed,
                "throughput_steps_per_second": (step + 1 - start_step)
                / max(elapsed, 1e-9),
                "latest_metric": float(objective),
                "best_metric": best["objective"],
                "best_step": best["step"],
                "error_count": 0,
                "run_id": config["run_id"],
                "config_fingerprint": config_fingerprint,
                **metrics,
                **{key: float(value) for key, value in current_components.items()},
            }
            writer.json("progress.json", row)
            writer.jsonl("metrics.jsonl", row)
            torch.save(
                {
                    "suffix": suffix.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "next_step": step + 1,
                },
                state_path,
            )
            writer.json(
                "checkpoint.json",
                {
                    "status": "stopped" if stopped else "running",
                    "active_phase": "combined_prefill",
                    "next_step": step + 1,
                    "latest_metric": float(objective),
                    "best_metric": best["objective"],
                    "best_step": best["step"],
                    "error_count": 0,
                    "run_id": config["run_id"],
                    "config_fingerprint": config_fingerprint,
                },
            )
            if stopped:
                raise KeyboardInterrupt("SIGTERM: checkpoint committed")

    best_suffix = best["suffix"].to(model.device)
    with torch.no_grad():
        final_hidden, final_logits = forward_trajectory(
            model, layout, best_suffix, continuation, config["hidden_state_index"]
        )
        final_metrics = diagnostic(
            final_hidden,
            final_logits,
            target_hidden,
            target_logits,
            baseline_hidden,
            vector,
        )
        baseline_ids = greedy_continuation(
            model, tokenizer, layout, filler, config["continuation_tokens"]
        )
        student_ids = greedy_continuation(
            model, tokenizer, layout, best_suffix, config["continuation_tokens"]
        )
    soft_dir = output / "soft_prompts"
    soft_dir.mkdir(exist_ok=True)
    torch.save(best["suffix"], soft_dir / "combined.pt")
    generations = {
        "prompt": prompt,
        "teacher_response": tokenizer.decode(continuation, skip_special_tokens=True),
        "baseline_response": tokenizer.decode(baseline_ids, skip_special_tokens=True),
        "student_response": tokenizer.decode(student_ids, skip_special_tokens=True),
    }
    writer.json("generations.json", generations)
    passed = (
        final_metrics["activation_cosine"] >= config["success"]["activation_cosine"]
        and final_metrics["position_0_kl"] <= config["success"]["position_0_kl"]
    )
    results = {
        "status": "complete",
        "run_id": config["run_id"],
        "best_step": best["step"],
        "best_objective": best["objective"],
        "success": passed,
        "success_thresholds": config["success"],
        "final_metrics": final_metrics,
        "generations": generations,
    }
    writer.json("results.json", results)
    plot_history(output)
    (output / "RESULTS.md").write_text(
        f"# Combined prefill probe\n\n- Success: {passed}\n- Best step: {best['step']}\n- Activation cosine: {final_metrics['activation_cosine']:.6g}\n- Position-0 KL: {final_metrics['position_0_kl']:.6g}\n- Mean forward KL: {final_metrics['mean_forward_kl']:.6g}\n"
    )
    writer.json(
        "checkpoint.json",
        {
            "status": "complete",
            "active_phase": None,
            "next_step": config["steps"],
            "latest_metric": best["objective"],
            "best_metric": best["objective"],
            "best_step": best["step"],
            "error_count": 0,
            "run_id": config["run_id"],
            "config_fingerprint": config_fingerprint,
        },
    )
    writer.commit()
    return results
