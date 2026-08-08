from __future__ import annotations

import csv
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
from .direct_steering_calibration import judge, repeated_trigram_fraction
from .io_utils import ArtifactWriter
from .layout import build_layout
from .live_dashboard import update_dashboard
from .prefill_probe import forward_trajectory, model_inputs
from .teacher import steering_hook


def fingerprint(config: dict[str, Any]) -> str:
    scientific = {key: value for key, value in config.items() if key != "run_mode"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def full_forward_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    student_logp = student_logits.log_softmax(-1)
    teacher_logp = teacher_logits.log_softmax(-1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)


def trajectory_loss(
    student_hidden: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    baseline_hidden: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_delta = teacher_hidden - baseline_hidden
    student_delta = student_hidden - baseline_hidden
    normalized_mse = torch.nn.functional.mse_loss(
        student_hidden, teacher_hidden
    ) / torch.nn.functional.mse_loss(
        baseline_hidden, teacher_hidden
    ).detach().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        student_delta.flatten().float(), teacher_delta.flatten().float(), dim=0
    )
    cosine_distance = 1.0 - cosine
    forward_kl = full_forward_kl(student_logits, teacher_logits).mean()
    total = (
        float(weights["normalized_mse"]) * normalized_mse
        + float(weights["cosine_distance"]) * cosine_distance
        + float(weights["forward_kl"]) * forward_kl
    )
    return total, {
        "normalized_mse": normalized_mse,
        "cosine_distance": cosine_distance,
        "forward_kl": forward_kl,
        "hidden_cosine": cosine,
    }


def projection_retention(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    baseline_hidden: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    student = torch.einsum("td,d->t", student_hidden - baseline_hidden, vector.float())
    teacher = torch.einsum("td,d->t", teacher_hidden - baseline_hidden, vector.float())
    return student / teacher.where(
        teacher.abs() >= 1e-8, torch.full_like(teacher, float("nan"))
    )


def aggregate_losses(losses: list[dict[str, float]]) -> dict[str, float]:
    if not losses:
        return {}
    return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}


def condition_name(horizon: int) -> str:
    return f"horizon_{horizon}"


def condition_gate_passes(evaluations, conditions, horizons, success):
    control_successes = conditions[condition_name(horizons[0])]["successes"]
    return {
        condition_name(horizon): (
            next(
                e["summary"]["projection_retention"]["16"]
                for e in evaluations
                if e["horizon"] == horizon
            )
            >= success["min_position_16_retention"]
            and conditions[condition_name(horizon)]["successes"]
            >= control_successes + success["min_success_gain"]
        )
        for horizon in horizons[1:]
    }


def initialize_suffix(
    embedding, length: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=embedding.weight.device).manual_seed(seed)
    ids = torch.randint(
        0,
        embedding.num_embeddings,
        (length,),
        generator=generator,
        device=embedding.weight.device,
    )
    return embedding(ids).detach().float(), ids.detach().cpu()


@torch.inference_mode()
def greedy_soft_generation(
    model, tokenizer, layout, suffix, max_new_tokens: int
) -> tuple[str, bool]:
    empty = torch.empty(0, dtype=torch.long)
    inputs, attention, positions, _ = model_inputs(model, layout, suffix, empty)
    generated = model.generate(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=positions,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    ids = generated.detach().cpu().tolist()
    hit_eos = tokenizer.eos_token_id in ids
    if hit_eos:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True), hit_eos


@torch.inference_mode()
def teacher_record(model, tokenizer, prompt: str, vector, config) -> dict[str, Any]:
    layout = build_layout(tokenizer, prompt, config["suffix_length"])
    empty_suffix = None
    empty = torch.empty(0, dtype=torch.long)
    inputs, attention, positions, _ = model_inputs(model, layout, empty_suffix, empty)
    with steering_hook(
        model,
        config["module_index"],
        vector,
        config["additive_norm"],
        "plain",
        layout.response_logit_start,
    ):
        generated = model.generate(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            max_new_tokens=config["teacher_tokens"],
            min_new_tokens=config["teacher_tokens"],
            do_sample=False,
            use_cache=False,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
        )[0, -config["teacher_tokens"] :].cpu()
    baseline_hidden, _ = forward_trajectory(
        model, layout, empty_suffix, generated, config["hidden_state_index"]
    )
    teacher_hidden, teacher_logits = forward_trajectory(
        model,
        layout,
        empty_suffix,
        generated,
        config["hidden_state_index"],
        hook=(
            config["module_index"],
            vector,
            config["additive_norm"],
            "plain",
            layout.response_logit_start,
        ),
    )
    return {
        "prompt": prompt,
        "continuation_ids": generated,
        "baseline_hidden": baseline_hidden.cpu(),
        "teacher_hidden": teacher_hidden.cpu(),
        "teacher_logits": teacher_logits.cpu(),
    }


def evaluate_suffix(
    model, tokenizer, suffix, records, vector, config, horizon: int
) -> dict[str, Any]:
    per_prompt = []
    positions = config["retention_positions"]
    with torch.no_grad():
        for record in records:
            layout = build_layout(tokenizer, record["prompt"], config["suffix_length"])
            continuation = record["continuation_ids"]
            student_hidden, student_logits = forward_trajectory(
                model, layout, suffix, continuation, config["hidden_state_index"]
            )
            teacher_hidden = record["teacher_hidden"].to(model.device)
            teacher_logits = record["teacher_logits"].to(model.device)
            baseline_hidden = record["baseline_hidden"].to(model.device)
            loss, components = trajectory_loss(
                student_hidden,
                student_logits,
                teacher_hidden,
                teacher_logits,
                baseline_hidden,
                config["loss"],
            )
            retention = projection_retention(
                student_hidden, teacher_hidden, baseline_hidden, vector
            )
            kl_by_position = full_forward_kl(student_logits, teacher_logits)
            per_prompt.append(
                {
                    "trajectory_loss": float(loss),
                    **{key: float(value) for key, value in components.items()},
                    "retention": {str(p): float(retention[p]) for p in positions},
                    "kl_by_position": [float(x) for x in kl_by_position],
                    "cosine_by_position": [
                        float(
                            torch.nn.functional.cosine_similarity(
                                (student_hidden[p] - baseline_hidden[p]).float(),
                                (teacher_hidden[p] - baseline_hidden[p]).float(),
                                dim=0,
                            )
                        )
                        for p in range(config["teacher_tokens"])
                    ],
                }
            )
    summary = aggregate_losses(
        [
            {
                key: row[key]
                for key in (
                    "trajectory_loss",
                    "normalized_mse",
                    "cosine_distance",
                    "forward_kl",
                    "hidden_cosine",
                )
            }
            for row in per_prompt
        ]
    )
    summary["projection_retention"] = {
        str(p): float(np.nanmean([row["retention"][str(p)] for row in per_prompt]))
        for p in positions
    }
    return {
        "condition": condition_name(horizon),
        "horizon": horizon,
        "summary": summary,
        "per_prompt": per_prompt,
    }


def plot_results(
    output: Path,
    metrics: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    results: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for condition in sorted({row["condition"] for row in metrics}):
        rows = [row for row in metrics if row["condition"] == condition]
        axes[0, 0].plot(
            [r["step"] for r in rows],
            [r["trajectory_loss"] for r in rows],
            label=condition,
        )
        axes[0, 1].plot(
            [r["step"] for r in rows], [r["forward_kl"] for r in rows], label=condition
        )
    for evaluation in evaluations:
        all_positions = list(
            range(len(evaluation["per_prompt"][0]["cosine_by_position"]))
        )
        axes[0, 2].plot(
            all_positions,
            np.mean(
                [row["cosine_by_position"] for row in evaluation["per_prompt"]], axis=0
            ),
            label=evaluation["condition"],
        )
        axes[1, 0].plot(
            all_positions,
            np.mean(
                [row["kl_by_position"] for row in evaluation["per_prompt"]], axis=0
            ),
            label=evaluation["condition"],
        )
        positions = sorted(
            int(p) for p in evaluation["summary"]["projection_retention"]
        )
        axes[1, 1].plot(
            positions,
            [evaluation["summary"]["projection_retention"][str(p)] for p in positions],
            marker="o",
            label=evaluation["condition"],
        )
    conditions = [condition_name(h) for h in results["horizons"]]
    axes[1, 2].bar(conditions, [results["conditions"][c]["asr"] for c in conditions])
    axes[0, 0].set(xlabel="Optimization step", ylabel="Trajectory loss")
    axes[0, 1].set(xlabel="Optimization step", ylabel="Forward KL (nats)")
    axes[0, 2].set(xlabel="Response position", ylabel="Hidden-delta cosine")
    axes[1, 0].set(xlabel="Response position", ylabel="Forward KL (nats)")
    axes[1, 1].set(
        xlabel="Response position", ylabel="Teacher projection retained (ratio)"
    )
    axes[1, 2].set(xlabel="Suffix condition", ylabel="ASR")
    axes[1, 2].set_ylim(0, 1)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "trajectory_imitation.png", dpi=180)
    plt.close(figure)


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    config_hash = fingerprint(config)
    writer = ArtifactWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    if mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError("fresh run refuses existing checkpoint")
    checkpoint = (
        json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    )
    if mode == "resume" and checkpoint.get("config_fingerprint") != config_hash:
        raise ValueError("resume config fingerprint mismatch")
    vector_path = Path(config["vector_path"])
    dataset_path = Path(config["dataset_csv"])
    if file_sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("vector checksum mismatch")
    if file_sha256(dataset_path) != config["dataset_sha256"]:
        raise ValueError("dataset checksum mismatch")
    calibration_dir = Path(config["source_calibration_dir"])
    calibration_resolved = json.loads(
        (calibration_dir / "resolved_config.json").read_text()
    )
    calibration_selection = json.loads((calibration_dir / "selection.json").read_text())
    if (
        calibration_resolved["config_fingerprint"]
        != config["source_calibration_fingerprint"]
    ):
        raise ValueError("source calibration fingerprint mismatch")
    if [row["dataset_row"] for row in calibration_selection["rows"]] != config[
        "training_rows"
    ]:
        raise ValueError("training rows do not match source calibration selection")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    config.update(run_id=output.name, config_fingerprint=config_hash)
    writer.json("resolved_config.json", config)
    writer.json(
        "source_artifacts.json",
        {
            "calibration_run_id": config["source_calibration_run_id"],
            "calibration_fingerprint": config["source_calibration_fingerprint"],
            "vector_sha256": config["vector_sha256"],
            "implementation_commit": config["implementation_commit"],
        },
    )
    with dataset_path.open(newline="") as stream:
        source = list(csv.DictReader(stream))

    def select(indices):
        return [
            {
                "dataset_row": i,
                "behavior_id": hashlib.sha256(source[i]["goal"].encode()).hexdigest()[
                    :12
                ],
                "prompt": source[i]["goal"],
            }
            for i in indices
        ]

    training_selection, validation_selection = (
        select(config["training_rows"]),
        select(config["validation_rows"]),
    )
    writer.json(
        "selection.json",
        {
            "dataset_sha256": file_sha256(dataset_path),
            "training": training_selection,
            "validation": validation_selection,
        },
    )
    if not checkpoint:
        update_dashboard(
            output,
            {
                "run_id": output.name,
                "config_fingerprint": config_hash,
                "phase": "initializing",
                "completed": 0,
                "total": len(config["horizons"]) * config["steps"],
                "error_count": 0,
                "retry_count": 0,
            },
        )
        writer.json(
            "checkpoint.json",
            {
                "status": "running",
                "phase": "initializing",
                "completed_conditions": [],
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": 0,
            },
        )
    writer.commit()

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
    stopped = False

    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    cache_path = output / "teacher_cache.pt"
    cache = (
        torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_path.exists()
        else {"training": [], "validation": []}
    )
    cache_total = len(training_selection) + len(validation_selection)
    for split, selection in (
        ("training", training_selection),
        ("validation", validation_selection),
    ):
        for item in tqdm(
            selection[len(cache[split]) :],
            initial=len(cache[split]),
            total=len(selection),
            desc=f"cache_{split}",
        ):
            cache[split].append(
                {
                    "behavior_id": item["behavior_id"],
                    **teacher_record(model, tokenizer, item["prompt"], vector, config),
                }
            )
            torch.save(cache, cache_path)
            cache_completed = len(cache["training"]) + len(cache["validation"])
            cache_progress = {
                "run_id": output.name,
                "config_fingerprint": config_hash,
                "phase": f"cache_{split}",
                "completed": cache_completed,
                "total": cache_total,
                "completed_fraction": cache_completed / cache_total,
                "error_count": 0,
                "retry_count": 0,
            }
            writer.json("progress.json", cache_progress)
            update_dashboard(output, cache_progress)
            writer.json(
                "checkpoint.json",
                {
                    "status": "stopped" if stopped else "running",
                    "phase": f"cache_{split}",
                    "cached": cache_completed,
                    "completed_conditions": [],
                    "config_fingerprint": config_hash,
                    "error_count": 0,
                    "retry_count": 0,
                },
            )
            writer.commit()
            if stopped:
                return {
                    "status": "stopped",
                    "phase": f"cache_{split}",
                    "completed": cache_completed,
                }
    initial_suffix, initial_ids = initialize_suffix(
        model.get_input_embeddings(), config["suffix_length"], config["seed"]
    )
    writer.json(
        "initialization.json",
        {
            "seed": config["seed"],
            "token_ids": initial_ids.tolist(),
            "shared_across_conditions": True,
        },
    )
    soft_dir, state_dir = output / "soft_prompts", output / "optimizer_states"
    soft_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    completed_conditions = list(checkpoint.get("completed_conditions", []))
    best_suffixes: dict[str, torch.Tensor] = {}
    metric_rows = (
        [json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()]
        if (output / "metrics.jsonl").exists()
        else []
    )
    run_started = time.monotonic()
    for horizon in config["horizons"]:
        condition = condition_name(horizon)
        final_path = soft_dir / f"{condition}.pt"
        if condition in completed_conditions and final_path.exists():
            best_suffixes[condition] = torch.load(
                final_path, map_location="cpu", weights_only=False
            )
            continue
        suffix = torch.nn.Parameter(initial_suffix.to(model.device).clone())
        optimizer = torch.optim.AdamW(
            [suffix], lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        state_path = state_dir / f"{condition}.pt"
        start_step, stale = 0, 0
        best = {
            "validation_loss": float("inf"),
            "step": 0,
            "suffix": suffix.detach().cpu().clone(),
        }
        if checkpoint.get("active_condition") == condition and state_path.exists():
            state = torch.load(
                state_path, map_location=model.device, weights_only=False
            )
            suffix.data.copy_(state["suffix"].to(model.device))
            optimizer.load_state_dict(state["optimizer"])
            start_step, stale, best = (
                int(state["next_step"]),
                int(state["stale"]),
                state["best"],
            )
        for step in tqdm(
            range(start_step, config["steps"]),
            initial=start_step,
            total=config["steps"],
            desc=condition,
        ):
            optimizer.zero_grad(set_to_none=True)
            train_components = []
            total_loss = torch.zeros((), device=model.device)
            for record in cache["training"]:
                layout = build_layout(
                    tokenizer, record["prompt"], config["suffix_length"]
                )
                continuation = record["continuation_ids"][:horizon]
                student_hidden, student_logits = forward_trajectory(
                    model, layout, suffix, continuation, config["hidden_state_index"]
                )
                loss, components = trajectory_loss(
                    student_hidden,
                    student_logits,
                    record["teacher_hidden"][:horizon].to(model.device),
                    record["teacher_logits"][:horizon].to(model.device),
                    record["baseline_hidden"][:horizon].to(model.device),
                    config["loss"],
                )
                total_loss = total_loss + loss / len(cache["training"])
                train_components.append(
                    {key: float(value.detach()) for key, value in components.items()}
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([suffix], config["gradient_clip"])
            optimizer.step()
            diagnostic = step == 0 or (step + 1) % config["checkpoint_every"] == 0
            if diagnostic:
                validation_losses = []
                with torch.no_grad():
                    for record in cache["validation"]:
                        layout = build_layout(
                            tokenizer, record["prompt"], config["suffix_length"]
                        )
                        continuation = record["continuation_ids"][:horizon]
                        sh, sl = forward_trajectory(
                            model,
                            layout,
                            suffix,
                            continuation,
                            config["hidden_state_index"],
                        )
                        value, _ = trajectory_loss(
                            sh,
                            sl,
                            record["teacher_hidden"][:horizon].to(model.device),
                            record["teacher_logits"][:horizon].to(model.device),
                            record["baseline_hidden"][:horizon].to(model.device),
                            config["loss"],
                        )
                        validation_losses.append(float(value))
                validation_loss = float(np.mean(validation_losses))
                if validation_loss < best["validation_loss"]:
                    best = {
                        "validation_loss": validation_loss,
                        "step": step + 1,
                        "suffix": suffix.detach().cpu().clone(),
                    }
                    stale = 0
                else:
                    stale += config["checkpoint_every"]
                elapsed = time.monotonic() - run_started
                global_completed = (
                    len(completed_conditions) * config["steps"] + step + 1
                )
                throughput = global_completed / max(elapsed, 1e-9)
                row = {
                    "phase": "optimize",
                    "condition": condition,
                    "horizon": horizon,
                    "step": step + 1,
                    "completed": global_completed,
                    "total": len(config["horizons"]) * config["steps"],
                    "elapsed_seconds": elapsed,
                    "throughput_per_second": throughput,
                    "throughput_steps_per_second": throughput,
                    "eta_seconds": (
                        len(config["horizons"]) * config["steps"] - global_completed
                    )
                    / max(throughput, 1e-9),
                    "trajectory_loss": float(total_loss),
                    "validation_loss": validation_loss,
                    "best_validation_loss": best["validation_loss"],
                    "best_step": best["step"],
                    "error_count": 0,
                    "retry_count": 0,
                    "run_id": output.name,
                    "config_fingerprint": config_hash,
                    **aggregate_losses(train_components),
                }
                writer.jsonl("metrics.jsonl", row)
                metric_rows.append(row)
                writer.json("progress.json", row)
                update_dashboard(output, row)
                torch.save(
                    {
                        "suffix": suffix.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                        "next_step": step + 1,
                        "stale": stale,
                        "best": best,
                    },
                    state_path,
                )
                writer.json(
                    "checkpoint.json",
                    {
                        "status": "stopped" if stopped else "running",
                        "phase": "optimize",
                        "active_condition": condition,
                        "next_step": step + 1,
                        "completed_conditions": completed_conditions,
                        "config_fingerprint": config_hash,
                        "error_count": 0,
                        "retry_count": 0,
                    },
                )
                writer.commit()
                if stopped:
                    return {
                        "status": "stopped",
                        "condition": condition,
                        "step": step + 1,
                    }
                if stale >= config["early_stopping_patience"]:
                    break
        torch.save(best["suffix"], final_path)
        writer.commit()
        best_suffixes[condition] = best["suffix"]
        completed_conditions.append(condition)
        writer.json(
            "checkpoint.json",
            {
                "status": "running",
                "phase": "optimize",
                "active_condition": None,
                "next_step": 0,
                "completed_conditions": completed_conditions,
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": 0,
            },
        )
    evaluations = [
        evaluate_suffix(
            model,
            tokenizer,
            best_suffixes[condition_name(h)].to(model.device),
            cache["validation"],
            vector,
            config,
            h,
        )
        for h in config["horizons"]
    ]
    writer.json("trajectory_evaluation.json", evaluations)
    generation_path = output / "paired_generations.jsonl"
    generation_rows = (
        [json.loads(x) for x in generation_path.read_text().splitlines() if x]
        if generation_path.exists()
        else []
    )
    for item in tqdm(
        validation_selection[len(generation_rows) :],
        initial=len(generation_rows),
        total=len(validation_selection),
        desc="paired_generation",
    ):
        layout = build_layout(tokenizer, item["prompt"], config["suffix_length"])
        baseline, baseline_eos = greedy_soft_generation(
            model, tokenizer, layout, None, config["max_new_tokens"]
        )
        responses = {"baseline": baseline}
        quality = {
            "baseline": {
                "hit_eos": baseline_eos,
                "repeated_trigram_fraction": repeated_trigram_fraction(baseline),
            }
        }
        for horizon in config["horizons"]:
            condition = condition_name(horizon)
            response, hit_eos = greedy_soft_generation(
                model,
                tokenizer,
                layout,
                best_suffixes[condition].to(model.device),
                config["max_new_tokens"],
            )
            responses[condition] = response
            quality[condition] = {
                "hit_eos": hit_eos,
                "repeated_trigram_fraction": repeated_trigram_fraction(response),
            }
        row = {**item, "responses": responses, "quality": quality, "judgments": {}}
        generation_rows.append(row)
        writer.jsonl("paired_generations.jsonl", row)
        generation_progress = {
            "phase": "generate",
            "completed": len(generation_rows),
            "total": len(validation_selection),
            "completed_fraction": len(generation_rows) / len(validation_selection),
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        }
        writer.json("progress.json", generation_progress)
        update_dashboard(output, generation_progress)
        writer.json(
            "checkpoint.json",
            {
                "status": "stopped" if stopped else "running",
                "phase": "generate",
                "generated": len(generation_rows),
                "completed_conditions": completed_conditions,
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": 0,
            },
        )
        if stopped:
            return {
                "status": "stopped",
                "phase": "generate",
                "completed": len(generation_rows),
            }
    partial_path = output / "paired_results.partial.json"
    if partial_path.exists():
        generation_rows = json.loads(partial_path.read_text())
    pending = [
        (row, condition)
        for row in generation_rows
        for condition in row["responses"]
        if condition not in row["judgments"]
    ]
    judged = sum(len(row["judgments"]) for row in generation_rows)
    total_judgments = len(generation_rows) * (1 + len(config["horizons"]))
    for row, condition in tqdm(
        pending, initial=judged, total=total_judgments, desc="openai_judge"
    ):
        result = judge(row["prompt"], row["responses"][condition], config["judge"])
        row["judgments"][condition] = result
        writer.jsonl(
            "openai_judgments.jsonl",
            {"behavior_id": row["behavior_id"], "condition": condition, **result},
        )
        judged += 1
        writer.json("paired_results.partial.json", generation_rows)
        judge_progress = {
            "phase": "judge",
            "completed": judged,
            "total": total_judgments,
            "completed_fraction": judged / total_judgments,
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": sum(
                int(j["retries"])
                for r in generation_rows
                for j in r["judgments"].values()
            ),
        }
        writer.json("progress.json", judge_progress)
        update_dashboard(output, judge_progress)
        writer.json(
            "checkpoint.json",
            {
                "status": "stopped" if stopped else "running",
                "phase": "judge",
                "judged": judged,
                "completed_conditions": completed_conditions,
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": judge_progress["retry_count"],
            },
        )
        if stopped:
            return {"status": "stopped", "phase": "judge", "completed": judged}
    baseline_flags = np.array(
        [int(row["judgments"]["baseline"]["success"]) for row in generation_rows]
    )
    conditions = {}
    for horizon in config["horizons"]:
        condition = condition_name(horizon)
        flags = np.array(
            [int(row["judgments"][condition]["success"]) for row in generation_rows]
        )
        conditions[condition] = {
            "successes": int(flags.sum()),
            "asr": float(flags.mean()),
            "failure_to_success": int(((baseline_flags == 0) & (flags == 1)).sum()),
            "success_to_failure": int(((baseline_flags == 1) & (flags == 0)).sum()),
            "incoherent_rate": float(
                np.mean(
                    [
                        not row["judgments"][condition]["coherent"]
                        for row in generation_rows
                    ]
                )
            ),
            "mean_repeated_trigram_fraction": float(
                np.mean(
                    [
                        row["quality"][condition]["repeated_trigram_fraction"]
                        for row in generation_rows
                    ]
                )
            ),
            "eos_rate": float(
                np.mean(
                    [row["quality"][condition]["hit_eos"] for row in generation_rows]
                )
            ),
        }
    control, candidates = (
        conditions[condition_name(config["horizons"][0])],
        [conditions[condition_name(h)] for h in config["horizons"][1:]],
    )
    retention_pass = any(
        e["summary"]["projection_retention"]["16"]
        >= config["success"]["min_position_16_retention"]
        for e in evaluations
        if e["horizon"] > 1
    )
    asr_pass = any(
        c["successes"] >= control["successes"] + config["success"]["min_success_gain"]
        for c in candidates
    )
    multi_position_passes = condition_gate_passes(
        evaluations, conditions, config["horizons"], config["success"]
    )
    results = {
        "status": "complete",
        "run_id": output.name,
        "horizons": config["horizons"],
        "baseline": {
            "successes": int(baseline_flags.sum()),
            "asr": float(baseline_flags.mean()),
        },
        "conditions": conditions,
        "trajectory_evaluations": {e["condition"]: e["summary"] for e in evaluations},
        "retention_gate_passed": retention_pass,
        "asr_gate_passed": asr_pass,
        "condition_gate_passes": multi_position_passes,
        "success": any(multi_position_passes.values()),
    }
    judgment_rows = [
        {"behavior_id": row["behavior_id"], "condition": condition, **judgment}
        for row in generation_rows
        for condition, judgment in row["judgments"].items()
    ]
    (output / "openai_judgments.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in judgment_rows)
    )
    writer.commit()
    writer.json("paired_results.json", generation_rows)
    writer.json("results.json", results)
    plot_results(output, metric_rows, evaluations, results)
    (output / "RESULTS.md").write_text(
        "# Multi-position trajectory imitation\n\n"
        + f"- Baseline ASR: {results['baseline']['successes']}/10\n"
        + "\n".join(
            f"- {name}: {value['successes']}/10 ASR; position-16 retention {results['trajectory_evaluations'][name]['projection_retention']['16']:.3f}"
            for name, value in conditions.items()
        )
        + f"\n- Success gate: {results['success']}\n"
    )
    final = {
        "phase": "complete",
        "completed": len(config["horizons"]) * config["steps"] + 40,
        "total": len(config["horizons"]) * config["steps"] + 40,
        "error_count": 0,
        "retry_count": 0,
        "run_id": output.name,
        "config_fingerprint": config_hash,
        "success": results["success"],
        "baseline_asr": results["baseline"]["asr"],
        **{f"{name}_asr": value["asr"] for name, value in conditions.items()},
        **{
            f"{e['condition']}_position_16_retention": e["summary"][
                "projection_retention"
            ]["16"]
            for e in evaluations
        },
    }
    writer.json("progress.json", final)
    writer.json(
        "checkpoint.json",
        {
            "status": "complete",
            "phase": "complete",
            "completed_conditions": completed_conditions,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        },
    )
    update_dashboard(output, final)
    writer.commit()
    return results
