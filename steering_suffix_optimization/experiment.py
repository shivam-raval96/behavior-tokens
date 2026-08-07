from __future__ import annotations

import csv
import json
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import file_sha256, fingerprint, load_config
from .io_utils import ArtifactWriter
from .judge import judge_behavior
from .layout import PromptLayout, build_layout, splice_embeddings
from .metrics import (
    activation_diagnostics,
    position_buckets,
    project_mean_embedding_norm_,
    sparse_forward_kl,
)
from .teacher import steering_hook


@dataclass(frozen=True)
class Target:
    prompt_index: int
    continuation_index: int
    continuation_ids: torch.Tensor
    top_ids: torch.Tensor
    top_logp: torch.Tensor


def _read_prompts(path: Path, column: str, indices: list[int]) -> list[tuple[int, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if max(indices) >= len(rows):
        raise IndexError("prompt index exceeds dataset")
    return [(i, rows[i][column].strip()) for i in indices]


def _model_inputs(
    model, layout: PromptLayout, suffix: torch.Tensor | None, continuation: torch.Tensor
):
    embedding = model.get_input_embeddings()
    inputs, continuation_slice = splice_embeddings(
        embedding, layout, suffix, continuation
    )
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
    return inputs, attention, positions, continuation_slice


def _score_logits(
    model,
    layout: PromptLayout,
    continuation: torch.Tensor,
    suffix: torch.Tensor | None = None,
    hook: tuple[int, torch.Tensor, float, str] | None = None,
) -> torch.Tensor:
    inputs, attention, positions, continuation_slice = _model_inputs(
        model, layout, suffix, continuation
    )
    if hook is None:
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        )
    else:
        layer, vector, alpha, mode = hook
        with steering_hook(model, layer, vector, alpha, mode, layout.steer_start):
            output = model(
                inputs_embeds=inputs,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            )
    return output.logits[0, continuation_slice].float()


def _score_logits_batch(
    model,
    layout: PromptLayout,
    continuations: list[torch.Tensor],
    suffix=None,
    hook=None,
) -> torch.Tensor:
    packed = [_model_inputs(model, layout, suffix, row) for row in continuations]
    inputs = torch.cat([row[0] for row in packed], dim=0)
    attention = torch.cat([row[1] for row in packed], dim=0)
    positions = torch.cat([row[2] for row in packed], dim=0)
    continuation_slice = packed[0][3]
    if hook is None:
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
        )
    else:
        layer, vector, alpha, mode = hook
        with steering_hook(model, layer, vector, alpha, mode, layout.steer_start):
            output = model(
                inputs_embeds=inputs,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            )
    return output.logits[:, continuation_slice].float()


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    layout: PromptLayout,
    tokens: int,
    temperature: float,
    seed: int,
    suffix: torch.Tensor | None = None,
    hook: tuple[int, torch.Tensor, float, str] | None = None,
) -> torch.Tensor:
    empty = torch.empty(0, dtype=torch.long)
    inputs, attention, positions, _ = _model_inputs(model, layout, suffix, empty)
    kwargs = dict(
        inputs_embeds=inputs.unsqueeze(0) if inputs.ndim == 2 else inputs,
        attention_mask=attention,
        position_ids=positions,
        max_new_tokens=tokens,
        do_sample=True,
        temperature=temperature,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,
    )
    torch.manual_seed(seed)
    if inputs.is_cuda:
        torch.cuda.manual_seed_all(seed)
    if hook is None:
        result = model.generate(**kwargs)
    else:
        layer, vector, alpha, mode = hook
        with steering_hook(model, layer, vector, alpha, mode, layout.steer_start):
            result = model.generate(**kwargs)
    return result[0, -tokens:].detach().cpu()


def _cache_targets(
    model,
    tokenizer,
    layouts: dict[int, PromptLayout],
    vector: torch.Tensor,
    config: dict,
    alpha: float,
    suffix_length: int,
    position_matched: bool,
    seed: int,
    reference: list[Target] | None = None,
    teacher_suffix: torch.Tensor | None = None,
) -> list[Target]:
    targets: list[Target] = []
    filler = teacher_suffix
    if filler is None and position_matched:
        filler = _fixed_suffix(model, tokenizer, config["filler_suffix"], suffix_length)
    hook = (config["hook_layer_index"], vector, alpha, config["hook_scale_mode"])
    reference_map = {
        (row.prompt_index, row.continuation_index): row for row in reference or []
    }
    for prompt_index, layout in tqdm(
        layouts.items(), desc="cache teacher", unit="prompt"
    ):
        continuations = []
        for sample in range(config["continuations_per_prompt"]):
            prior = reference_map.get((prompt_index, sample))
            continuation = (
                prior.continuation_ids
                if prior
                else _generate(
                    model,
                    tokenizer,
                    layout,
                    config["continuation_tokens"],
                    config["temperature"],
                    seed + prompt_index * 1000 + sample,
                    filler,
                    hook,
                )
            )
            continuations.append(continuation)
        with torch.no_grad():
            batch_logits = _score_logits_batch(
                model, layout, continuations, filler, hook
            )
        for sample, (continuation, logits) in enumerate(
            zip(continuations, batch_logits)
        ):
            logp = logits.log_softmax(-1)
            values, ids = logp.topk(
                min(config["teacher_top_k"], logp.shape[-1]), dim=-1
            )
            targets.append(
                Target(prompt_index, sample, continuation, ids.cpu(), values.cpu())
            )
    return targets


def _teacher_discrepancy(naive: list[Target], matched: list[Target]) -> float:
    """Symptom metric for C2, using matched scores on identical continuations."""
    by_key = {(x.prompt_index, x.continuation_index): x for x in matched}
    values = []
    for left in naive:
        right = by_key[(left.prompt_index, left.continuation_index)]
        # The union would be tighter; intersection is stable and conservative for
        # deciding whether positional matching is non-negligible.
        right_maps = [
            {int(i): float(v) for i, v in zip(ids, vals)}
            for ids, vals in zip(right.top_ids, right.top_logp)
        ]
        for position, (ids, logp) in enumerate(zip(left.top_ids, left.top_logp)):
            keep_ids, left_values, right_values = [], [], []
            mapping = right_maps[position]
            for token, value in zip(ids.tolist(), logp.tolist()):
                if token in mapping:
                    keep_ids.append(token)
                    left_values.append(value)
                    right_values.append(mapping[token])
            if keep_ids:
                lp = torch.tensor(left_values)
                rp = torch.tensor(right_values)
                values.append(float((lp.exp() * (lp - rp)).sum()))
    return float(np.mean(values)) if values else float("inf")


def _target_kl(
    model,
    layouts: dict[int, PromptLayout],
    targets: list[Target],
    suffix: torch.Tensor | None,
    need_hidden: bool = False,
    vector: torch.Tensor | None = None,
    layer_index: int | None = None,
):
    all_kl, deltas = [], []
    for prompt_index, layout in layouts.items():
        prompt_targets = [row for row in targets if row.prompt_index == prompt_index]
        if not prompt_targets:
            continue
        batch_logits = _score_logits_batch(
            model, layout, [row.continuation_ids for row in prompt_targets], suffix
        )
        for target, logits in zip(prompt_targets, batch_logits):
            all_kl.append(
                sparse_forward_kl(
                    logits.log_softmax(-1),
                    target.top_ids.to(logits.device),
                    target.top_logp.to(logits.device),
                )
            )
        if need_hidden:
            target = prompt_targets[0]
            inputs, attention, positions, continuation_slice = _model_inputs(
                model, layout, suffix, target.continuation_ids
            )
            with torch.no_grad():
                states = model(
                    inputs_embeds=inputs,
                    attention_mask=attention,
                    position_ids=positions,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states[layer_index]
                clean_inputs, clean_attention, clean_positions, clean_slice = (
                    _model_inputs(model, layout, None, target.continuation_ids)
                )
                clean = model(
                    inputs_embeds=clean_inputs,
                    attention_mask=clean_attention,
                    position_ids=clean_positions,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states[layer_index]
            deltas.append(
                states[0, continuation_slice].cpu() - clean[0, clean_slice].cpu()
            )
    per_position = torch.stack(all_kl).mean(0)
    diagnostics = position_buckets(per_position)
    if deltas:
        diagnostics["on_target_projection"], diagnostics["off_target_norm"] = (
            activation_diagnostics(torch.cat(deltas), vector.cpu())
        )
    return torch.stack(all_kl).mean(), diagnostics


def _fixed_suffix(model, tokenizer, text: str, length: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("fixed suffix tokenized to no tokens")
    ids = (ids * math.ceil(length / len(ids)))[:length]
    return model.get_input_embeddings()(torch.tensor(ids, device=model.device)).detach()


def _random_suffix(model, length: int, seed: int) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    generator = torch.Generator(device=embedding.weight.device).manual_seed(seed)
    ids = torch.randint(
        0,
        embedding.num_embeddings,
        (length,),
        generator=generator,
        device=embedding.weight.device,
    )
    return embedding(ids).detach()


def _behavior_checkpoint(
    model, tokenizer, layouts, prompt_texts, suffix, config, writer, cell_id, step
):
    """Paired raw baseline/suffix generations required for every ASR diagnostic."""
    successes = []
    for prompt_index, layout in tqdm(
        layouts.items(), desc="behavior eval", unit="prompt"
    ):
        seed = config["behavior_seed"] + prompt_index + step * 10000
        baseline_ids = _generate(
            model, tokenizer, layout, config["behavior_generation_tokens"], 0.7, seed
        )
        suffix_ids = _generate(
            model,
            tokenizer,
            layout,
            config["behavior_generation_tokens"],
            0.7,
            seed,
            suffix.detach(),
        )
        baseline = tokenizer.decode(baseline_ids, skip_special_tokens=True)
        attacked = tokenizer.decode(suffix_ids, skip_special_tokens=True)
        baseline_judgment = judge_behavior(
            prompt_texts[prompt_index], baseline, config["behavior_metric"]
        )
        attacked_judgment = judge_behavior(
            prompt_texts[prompt_index], attacked, config["behavior_metric"]
        )
        row = {
            "cell_id": cell_id,
            "step": step,
            "prompt_index": prompt_index,
            "prompt": prompt_texts[prompt_index],
            "baseline_response": baseline,
            "suffix_response": attacked,
            "baseline_success": baseline_judgment["success"],
            "suffix_success": attacked_judgment["success"],
            "baseline_judgment": baseline_judgment,
            "suffix_judgment": attacked_judgment,
        }
        writer.jsonl("behavior_generations.jsonl", row)
        successes.append(bool(attacked_judgment["success"]))
    return sum(successes) / len(successes)


def _optimize(
    model,
    tokenizer,
    train_layouts,
    heldout_layouts,
    train_targets,
    heldout_targets,
    baseline_train: float,
    baseline_heldout: float,
    config,
    length: int,
    constraint: str,
    seed: int,
    writer: ArtifactWriter,
    cell_id: str,
    prompt_texts,
    checkpoint: dict,
    completed: set[str],
    results: list[dict],
    should_stop,
):
    torch.manual_seed(seed)
    random.seed(seed)
    embedding = model.get_input_embeddings()
    candidates = torch.randint(
        0, embedding.num_embeddings, (length,), device=model.device
    )
    suffix = torch.nn.Parameter(embedding(candidates).detach().float())
    optimizer = torch.optim.Adam([suffix], lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["steps"], eta_min=config["min_learning_rate"]
    )
    best = {"heldout_normalized_kl": float("inf"), "step": -1, "suffix": None}
    patience = 0
    state_dir = writer.output / "optimizer_state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{cell_id}.pt"
    start_step = 0
    if checkpoint.get("active_cell") == cell_id and state_path.exists():
        state = torch.load(state_path, map_location=model.device, weights_only=False)
        suffix.data.copy_(state["suffix"].to(suffix.device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["next_step"])
        best = state["best"]
        patience = int(state["patience"])
    target_norm = float(embedding.weight.float().norm(dim=-1).mean())
    for step in tqdm(
        range(start_step, config["steps"]),
        initial=start_step,
        total=config["steps"],
        desc=cell_id,
        unit="step",
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _target_kl(model, train_layouts, train_targets, suffix)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if constraint == "mean_embedding_norm":
            project_mean_embedding_norm_(suffix, target_norm)
        diagnostic = step == 0 or (step + 1) % config["diagnostic_every"] == 0
        if diagnostic:
            current = suffix.detach()
            train_loss, train_diag = _target_kl(
                model,
                train_layouts,
                train_targets,
                current,
                True,
                config["vector"],
                config["diagnostic_hidden_state_index"],
            )
            held_loss, held_diag = _target_kl(
                model, heldout_layouts, heldout_targets, current
            )
            normalized = float(held_loss) / max(baseline_heldout, 1e-12)
            row = {
                "phase": "optimize",
                "cell_id": cell_id,
                "step": step + 1,
                "total": config["steps"],
                "elapsed_seconds": time.monotonic() - config["started_monotonic"],
                "latest_metric": normalized,
                "best_metric": min(normalized, best["heldout_normalized_kl"]),
                "learning_rate": scheduler.get_last_lr()[0],
                "train_normalized_kl": float(train_loss) / max(baseline_train, 1e-12),
                "heldout_normalized_kl": normalized,
                "train_position_kl": train_diag,
                "heldout_position_kl": held_diag,
                "error_count": 0,
                "run_id": config["run_id"],
                "config_fingerprint": config["config_fingerprint"],
            }
            writer.json("progress.json", row)
            writer.jsonl("metrics.jsonl", row)
            if (step + 1) % config["generation_every"] == 0:
                row["behavior_success_rate"] = _behavior_checkpoint(
                    model,
                    tokenizer,
                    heldout_layouts,
                    prompt_texts,
                    current,
                    config,
                    writer,
                    cell_id,
                    step + 1,
                )
                writer.json("progress.json", row)
            if normalized < best["heldout_normalized_kl"]:
                best = {
                    "heldout_normalized_kl": normalized,
                    "step": step + 1,
                    "suffix": suffix.detach().cpu().clone(),
                }
                patience = 0
            else:
                patience += config["diagnostic_every"]
            torch.save(
                {
                    "suffix": suffix.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "next_step": step + 1,
                    "best": best,
                    "patience": patience,
                },
                state_path,
            )
            writer.json(
                "checkpoint.json",
                {
                    "status": "stopped" if should_stop() else "running",
                    "active_cell": cell_id,
                    "next_step": step + 1,
                    "completed_cells": sorted(completed),
                    "results": results,
                    "latest_metric": normalized,
                    "best_metric": min(
                        [normalized] + [r["heldout_normalized_kl"] for r in results]
                    ),
                    "config_fingerprint": config["config_fingerprint"],
                    "run_id": config["run_id"],
                    "error_count": 0,
                },
            )
            if should_stop():
                raise KeyboardInterrupt("SIGTERM: optimizer checkpoint committed")
            if patience >= config["early_stop_patience"]:
                break
    return {
        "heldout_normalized_kl": best["heldout_normalized_kl"],
        "best_step": best["step"],
        "suffix": best["suffix"],
    }


def run(
    config_path: Path, output: Path, run_mode: str | None = None, commit=None
) -> dict[str, Any]:
    config = load_config(config_path)
    config["run_mode"] = run_mode or config["run_mode"]
    run_id = output.name
    fp = fingerprint(config)
    writer = ArtifactWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    if config["run_mode"] == "fresh" and checkpoint_path.exists():
        raise FileExistsError("fresh run refuses an existing checkpoint")
    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.exists()
        else {"completed_cells": [], "results": []}
    )
    if config["run_mode"] == "resume" and checkpoint.get("config_fingerprint") != fp:
        raise ValueError("resume config fingerprint mismatch")
    vector_path = Path(config["vector_path"])
    if file_sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("steering vector checksum mismatch")
    config_path_text = config_path.read_text()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path_text)
    writer.json(
        "resolved_config.json", {**config, "run_id": run_id, "config_fingerprint": fp}
    )
    config.update(
        run_id=run_id, config_fingerprint=fp, started_monotonic=time.monotonic()
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
    config["vector"] = vector
    all_rows = _read_prompts(
        Path(config["dataset_csv"]),
        config["prompt_column"],
        config["train_indices"] + config["heldout_indices"],
    )
    prompt_map = dict(all_rows)
    completed = set(checkpoint.get("completed_cells", []))
    results = list(checkpoint.get("results", []))

    stopped = False

    def terminate(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)

    # C1 always runs before any steered target. The one-prompt ceiling is its own cell.
    cells = [("c1_one_prompt", 1)] + [("c1_five_prompt", 5)]
    cells += [
        (f"main_a{a:g}_k{k}_{c}_s{s}", 5)
        for a in config["alpha_multipliers"]
        for k in config["suffix_lengths"]
        for c in config["constraint_modes"]
        for s in config["seeds"]
    ]
    for cell_id, train_count in cells:
        if cell_id in completed:
            continue
        is_control = cell_id.startswith("c1_")
        length = (
            len(
                tokenizer(
                    config["positive_instruction"], add_special_tokens=False
                ).input_ids
            )
            if is_control
            else int(cell_id.split("_k")[1].split("_")[0])
        )
        constraint = (
            "free"
            if is_control
            else next(c for c in config["constraint_modes"] if f"_{c}_" in cell_id)
        )
        seed = config["seeds"][0] if is_control else int(cell_id.rsplit("_s", 1)[1])
        alpha_multiplier = (
            0.0
            if is_control
            else float(cell_id.split("main_a", 1)[1].split("_k", 1)[0])
        )
        layouts = {
            i: build_layout(tokenizer, p, length, config.get("system_prompt"))
            for i, p in all_rows
        }
        train_ids = config["train_indices"][:train_count]
        heldout_ids = config["heldout_indices"]
        train_layouts = {i: layouts[i] for i in train_ids}
        heldout_layouts = {i: layouts[i] for i in heldout_ids}
        if is_control:
            reachable_suffix = _fixed_suffix(
                model, tokenizer, config["positive_instruction"], length
            )
            targets = _cache_targets(
                model,
                tokenizer,
                layouts,
                vector,
                config,
                0.0,
                length,
                False,
                seed,
                teacher_suffix=reachable_suffix,
            )
        else:
            alpha = config["alpha0"] * alpha_multiplier
            naive = _cache_targets(
                model, tokenizer, layouts, vector, config, alpha, length, False, seed
            )
            matched = _cache_targets(
                model,
                tokenizer,
                layouts,
                vector,
                config,
                alpha,
                length,
                True,
                seed,
                reference=naive,
            )
            discrepancy = _teacher_discrepancy(naive, matched)
            targets = (
                matched if discrepancy > config["position_match_threshold"] else naive
            )
        train_targets = [x for x in targets if x.prompt_index in train_ids]
        heldout_targets = [x for x in targets if x.prompt_index in heldout_ids]
        cache_dir = output / "teacher_cache"
        cache_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "target_variant": "positive_control"
                if is_control
                else ("position_matched" if targets is matched else "naive"),
                "targets": [asdict(row) for row in targets],
            },
            cache_dir / f"{cell_id}.pt",
        )
        writer.commit()
        baseline_train = float(_target_kl(model, train_layouts, train_targets, None)[0])
        baseline_heldout = float(
            _target_kl(model, heldout_layouts, heldout_targets, None)[0]
        )
        result = _optimize(
            model,
            tokenizer,
            train_layouts,
            heldout_layouts,
            train_targets,
            heldout_targets,
            baseline_train,
            baseline_heldout,
            config,
            length,
            constraint,
            seed,
            writer,
            cell_id,
            prompt_map,
            checkpoint,
            completed,
            results,
            lambda: stopped,
        )
        suffix = result.pop("suffix")
        suffix_file = f"soft_prompts/{cell_id}.pt"
        (output / "soft_prompts").mkdir(exist_ok=True)
        torch.save(suffix, output / suffix_file)
        random_suffix = _random_suffix(model, length, config["behavior_seed"] + seed)
        natural_suffix = _fixed_suffix(
            model, tokenizer, config["natural_language_suffix"], length
        )
        controls = {
            "random_normalized_kl": float(
                _target_kl(model, heldout_layouts, heldout_targets, random_suffix)[0]
            )
            / max(baseline_heldout, 1e-12),
            "natural_language_normalized_kl": float(
                _target_kl(model, heldout_layouts, heldout_targets, natural_suffix)[0]
            )
            / max(baseline_heldout, 1e-12),
        }
        record = {
            "cell_id": cell_id,
            "positive_control": is_control,
            "train_prompts": train_count,
            "alpha_multiplier": alpha_multiplier,
            "suffix_length": length,
            "constraint": constraint,
            "seed": seed,
            "baseline_train_kl": baseline_train,
            "baseline_heldout_kl": baseline_heldout,
            "soft_prompt_file": suffix_file,
            **result,
            **controls,
        }
        if not is_control:
            record.update(
                c2_teacher_kl=discrepancy,
                target_variant="position_matched" if targets is matched else "naive",
            )
        results.append(record)
        completed.add(cell_id)
        checkpoint = {
            "status": "stopped" if stopped else "running",
            "next_cell": len(completed),
            "completed_cells": sorted(completed),
            "results": results,
            "latest_metric": record["heldout_normalized_kl"],
            "best_metric": min(r["heldout_normalized_kl"] for r in results),
            "config_fingerprint": fp,
            "run_id": run_id,
            "error_count": 0,
        }
        writer.json("checkpoint.json", checkpoint)
        if stopped:
            raise KeyboardInterrupt("SIGTERM: checkpoint committed")
    final = {
        "status": "complete",
        "run_id": run_id,
        "config_fingerprint": fp,
        "results": results,
    }
    writer.json("results.json", final)
    writer.json("checkpoint.json", {**checkpoint, "status": "complete"})
    return final
