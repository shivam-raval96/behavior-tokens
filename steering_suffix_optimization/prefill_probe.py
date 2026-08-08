from __future__ import annotations

import csv
import hashlib
import json
import signal
import time
from contextlib import nullcontext
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
from .layout import PromptLayout, build_layout, splice_embeddings
from .teacher import steering_hook


def config_fingerprint(config: dict[str, Any]) -> str:
    scientific = {key: value for key, value in config.items() if key != "run_mode"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def repeated_token_suffix(
    model, tokenizer, text: str, length: int
) -> tuple[torch.Tensor, list[int]]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("suffix text tokenized to zero tokens")
    ids = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    tensor = torch.tensor(ids, device=model.device)
    return model.get_input_embeddings()(tensor).detach(), ids


def model_inputs(
    model, layout: PromptLayout, suffix: torch.Tensor, continuation: torch.Tensor
):
    inputs, continuation_slice = splice_embeddings(
        model.get_input_embeddings(), layout, suffix, continuation
    )
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
    return inputs, attention, positions, continuation_slice


def forward_trajectory(
    model,
    layout: PromptLayout,
    suffix: torch.Tensor,
    continuation: torch.Tensor,
    hidden_state_index: int,
    hook: tuple[int, torch.Tensor, float, str] | tuple[int, torch.Tensor, float, str, int] | None = None,
):
    inputs, attention, positions, continuation_slice = model_inputs(
        model, layout, suffix, continuation
    )
    context = (
        steering_hook(
            model,
            hook[0],
            hook[1],
            hook[2],
            hook[3],
            hook[4] if len(hook) == 5 else layout.steer_start,
        )
        if hook
        else nullcontext()
    )
    with context:
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            output_hidden_states=True,
            use_cache=False,
        )
    return (
        output.hidden_states[hidden_state_index][0, continuation_slice].float(),
        output.logits[0, continuation_slice].float(),
    )


@torch.no_grad()
def greedy_continuation(
    model,
    tokenizer,
    layout: PromptLayout,
    suffix: torch.Tensor,
    tokens: int,
    hook: tuple[int, torch.Tensor, float, str] | tuple[int, torch.Tensor, float, str, int] | None = None,
) -> torch.Tensor:
    empty = torch.empty(0, dtype=torch.long)
    inputs, attention, positions, _ = model_inputs(model, layout, suffix, empty)
    context = (
        steering_hook(
            model,
            hook[0],
            hook[1],
            hook[2],
            hook[3],
            hook[4] if len(hook) == 5 else layout.steer_start,
        )
        if hook
        else nullcontext()
    )
    with context:
        generated = model.generate(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            max_new_tokens=tokens,
            do_sample=False,
            use_cache=False,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return generated[0, -tokens:].cpu()


def position_metrics(kl: torch.Tensor) -> dict[str, float]:
    return {
        "position_0_kl": float(kl[:1].mean()),
        "positions_1_8_kl": float(kl[1:9].mean()),
        "positions_9_32_kl": float(kl[9:].mean()),
    }


def diagnostic(
    student_hidden: torch.Tensor,
    student_logits: torch.Tensor,
    target_hidden: torch.Tensor,
    target_logits: torch.Tensor,
    baseline_hidden: torch.Tensor,
    vector: torch.Tensor,
) -> dict[str, float]:
    activation_mse = torch.nn.functional.mse_loss(student_hidden, target_hidden)
    student_delta = student_hidden - baseline_hidden
    target_delta = target_hidden - baseline_hidden
    flat_student = student_delta.flatten().float()
    flat_target = target_delta.flatten().float()
    trajectory_cosine = torch.nn.functional.cosine_similarity(
        flat_student, flat_target, dim=0
    )
    projections = torch.einsum("td,d->t", student_delta, vector.float())
    target_projections = torch.einsum("td,d->t", target_delta, vector.float())
    student_logp = student_logits.log_softmax(-1)
    target_logp = target_logits.log_softmax(-1)
    target_p = target_logp.exp()
    kl = (target_p * (target_logp - student_logp)).sum(-1)
    result = {
        "activation_mse": float(activation_mse),
        "activation_cosine": float(trajectory_cosine),
        "on_vector_projection": float(projections.mean()),
        "target_on_vector_projection": float(target_projections.mean()),
        "delta_norm_ratio": float(
            flat_student.norm() / flat_target.norm().clamp_min(1e-12)
        ),
        "mean_forward_kl": float(kl.mean()),
    }
    result.update(position_metrics(kl))
    return result


def optimize_phase(
    phase: str,
    model,
    tokenizer,
    layout,
    continuation,
    target_hidden,
    target_logits,
    baseline_hidden,
    vector,
    config,
    writer,
    checkpoint,
    should_stop,
):
    embedding = model.get_input_embeddings()
    torch.manual_seed(config["seed"])
    initial_ids = torch.randint(
        0, embedding.num_embeddings, (config["suffix_length"],), device=model.device
    )
    suffix = torch.nn.Parameter(embedding(initial_ids).detach().float())
    optimizer = torch.optim.Adam([suffix], lr=config["learning_rate"])
    state_dir = writer.output / "optimizer_state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{phase}.pt"
    start_step = 0
    best = {"activation_mse": float("inf"), "step": 0, "suffix": suffix.detach().cpu()}
    if checkpoint.get("active_phase") == phase and state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        suffix.data.copy_(state["suffix"].to(model.device))
        optimizer.load_state_dict(state["optimizer"])
        best = state["best"]
        start_step = int(state["next_step"])
    started = time.monotonic()
    for step in tqdm(
        range(start_step, config["steps"]),
        initial=start_step,
        total=config["steps"],
        desc=phase,
        unit="step",
    ):
        optimizer.zero_grad(set_to_none=True)
        student_hidden, student_logits = forward_trajectory(
            model,
            layout,
            suffix,
            continuation,
            config["hidden_state_index"],
        )
        loss = torch.nn.functional.mse_loss(student_hidden, target_hidden)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % config["diagnostic_every"] == 0:
            with torch.no_grad():
                current_hidden, current_logits = forward_trajectory(
                    model,
                    layout,
                    suffix,
                    continuation,
                    config["hidden_state_index"],
                )
                metrics = diagnostic(
                    current_hidden,
                    current_logits,
                    target_hidden,
                    target_logits,
                    baseline_hidden,
                    vector,
                )
            if metrics["activation_mse"] < best["activation_mse"]:
                best = {
                    "activation_mse": metrics["activation_mse"],
                    "step": step + 1,
                    "suffix": suffix.detach().cpu().clone(),
                }
            row = {
                "phase": phase,
                "completed": step + 1,
                "total": config["steps"],
                "elapsed_seconds": time.monotonic() - started,
                "throughput_steps_per_second": (step + 1 - start_step)
                / max(time.monotonic() - started, 1e-9),
                "latest_metric": metrics["activation_mse"],
                "best_metric": best["activation_mse"],
                "best_step": best["step"],
                "error_count": 0,
                "run_id": config["run_id"],
                "config_fingerprint": config["config_fingerprint"],
                **metrics,
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
                    "status": "stopped" if should_stop() else "running",
                    "active_phase": phase,
                    "next_step": step + 1,
                    "completed_phases": checkpoint.get("completed_phases", []),
                    "phase_results": checkpoint.get("phase_results", []),
                    "latest_metric": metrics["activation_mse"],
                    "best_metric": best["activation_mse"],
                    "error_count": 0,
                    "run_id": config["run_id"],
                    "config_fingerprint": config["config_fingerprint"],
                },
            )
            if should_stop():
                raise KeyboardInterrupt("SIGTERM: checkpoint committed")
    return best


def plot_metrics(output: Path) -> None:
    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase in sorted({row["phase"] for row in rows}):
        selected = [row for row in rows if row["phase"] == phase]
        x = [row["completed"] for row in selected]
        axes[0].plot(
            x, [row["activation_mse"] for row in selected], marker="o", label=phase
        )
        axes[1].plot(
            x, [row["mean_forward_kl"] for row in selected], marker="o", label=phase
        )
    axes[0].set(xlabel="step", ylabel="activation MSE", yscale="log")
    axes[1].set(xlabel="step", ylabel="forward KL", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "prefill_inversion.png", dpi=180)
    plt.close(figure)


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    fingerprint = config_fingerprint(config)
    writer = ArtifactWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    if mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError("fresh run refuses an existing checkpoint")
    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.exists()
        else {"completed_phases": [], "phase_results": []}
    )
    if mode == "resume" and checkpoint.get("config_fingerprint") != fingerprint:
        raise ValueError("resume config fingerprint mismatch")
    vector_path = Path(config["vector_path"])
    if file_sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("steering vector checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    config.update(run_id=output.name, config_fingerprint=fingerprint)
    writer.json("resolved_config.json", config)
    with Path(config["dataset_csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    prompt = rows[config["prompt_index"]][config["prompt_column"]].strip()
    writer.json(
        "selection.json", {"prompt_index": config["prompt_index"], "prompt": prompt}
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
        np.load(vector_path), device=model.device, dtype=torch.float32
    )
    vector /= vector.norm()
    layout = build_layout(
        tokenizer, prompt, config["suffix_length"], config.get("system_prompt")
    )
    known_suffix, known_ids = repeated_token_suffix(
        model, tokenizer, config["known_suffix_text"], config["suffix_length"]
    )
    filler_suffix, filler_ids = repeated_token_suffix(
        model, tokenizer, config["filler_suffix_text"], config["suffix_length"]
    )
    hook = (
        config["hook_layer_index"],
        vector,
        config["alpha"],
        config["hook_scale_mode"],
    )
    writer.json(
        "suffix_tokens.json",
        {
            "known_ids": known_ids,
            "known_text": tokenizer.decode(known_ids),
            "filler_ids": filler_ids,
            "filler_text": tokenizer.decode(filler_ids),
        },
    )

    stopped = False

    def terminate(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)

    phases = (
        ("positive_control", known_suffix, None),
        ("steered_teacher", filler_suffix, hook),
    )
    completed = set(checkpoint.get("completed_phases", []))
    phase_results = list(checkpoint.get("phase_results", []))
    for phase, teacher_suffix, phase_hook in phases:
        if phase in completed:
            continue
        continuation = greedy_continuation(
            model,
            tokenizer,
            layout,
            teacher_suffix,
            config["continuation_tokens"],
            phase_hook,
        )
        with torch.no_grad():
            target_hidden, target_logits = forward_trajectory(
                model,
                layout,
                teacher_suffix,
                continuation,
                config["hidden_state_index"],
                phase_hook,
            )
            baseline_hidden, _ = forward_trajectory(
                model, layout, filler_suffix, continuation, config["hidden_state_index"]
            )
        cache_dir = output / "teacher_cache"
        cache_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "continuation_ids": continuation,
                "target_hidden": target_hidden.cpu(),
                "target_logits": target_logits.cpu(),
                "baseline_hidden": baseline_hidden.cpu(),
            },
            cache_dir / f"{phase}.pt",
        )
        writer.commit()
        best = optimize_phase(
            phase,
            model,
            tokenizer,
            layout,
            continuation,
            target_hidden,
            target_logits,
            baseline_hidden,
            vector,
            config,
            writer,
            checkpoint,
            lambda: stopped,
        )
        best_suffix = best["suffix"].to(model.device)
        baseline_ids = greedy_continuation(
            model, tokenizer, layout, filler_suffix, config["continuation_tokens"]
        )
        student_ids = greedy_continuation(
            model, tokenizer, layout, best_suffix, config["continuation_tokens"]
        )
        record = {
            "phase": phase,
            "best_step": best["step"],
            "best_activation_mse": best["activation_mse"],
            "teacher_response": tokenizer.decode(
                continuation, skip_special_tokens=True
            ),
            "baseline_response": tokenizer.decode(
                baseline_ids, skip_special_tokens=True
            ),
            "student_response": tokenizer.decode(student_ids, skip_special_tokens=True),
            "best_suffix_file": f"soft_prompts/{phase}.pt",
        }
        soft_dir = output / "soft_prompts"
        soft_dir.mkdir(exist_ok=True)
        torch.save(best["suffix"], output / record["best_suffix_file"])
        writer.jsonl("generations.jsonl", record)
        phase_results.append(record)
        completed.add(phase)
        checkpoint = {
            "status": "running",
            "active_phase": None,
            "next_step": 0,
            "completed_phases": sorted(completed),
            "phase_results": phase_results,
            "latest_metric": best["activation_mse"],
            "best_metric": min(row["best_activation_mse"] for row in phase_results),
            "error_count": 0,
            "run_id": config["run_id"],
            "config_fingerprint": fingerprint,
        }
        writer.json("checkpoint.json", checkpoint)
    results = {
        "status": "complete",
        "run_id": config["run_id"],
        "config_fingerprint": fingerprint,
        "phase_results": phase_results,
    }
    writer.json("results.json", results)
    plot_metrics(output)
    (output / "RESULTS.md").write_text(
        "# Prefill inversion probe\n\n"
        + "\n".join(
            f"- {row['phase']}: best activation MSE {row['best_activation_mse']:.6g} at step {row['best_step']}"
            for row in phase_results
        )
        + "\n"
    )
    writer.json("checkpoint.json", {**checkpoint, "status": "complete"})
    writer.commit()
    return results
