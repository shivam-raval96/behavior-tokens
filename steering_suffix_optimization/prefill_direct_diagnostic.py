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
from .io_utils import ArtifactWriter
from .layout import build_layout
from .prefill_asr_eval import judge_response
from .prefill_probe import forward_trajectory, greedy_continuation


def fingerprint(config: dict[str, Any]) -> str:
    scientific = {k: v for k, v in config.items() if k != "run_mode"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def trajectory_metrics(
    baseline_hidden: torch.Tensor,
    soft_hidden: torch.Tensor,
    direct_hidden: torch.Tensor,
    soft_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    unit_vector: torch.Tensor,
) -> dict[str, Any]:
    soft_delta = (soft_hidden - baseline_hidden).float()
    direct_delta = (direct_hidden - baseline_hidden).float()
    position_cosine = torch.nn.functional.cosine_similarity(soft_delta, direct_delta, dim=-1)
    position_mse = (soft_hidden.float() - direct_hidden.float()).square().mean(-1)
    soft_projection = torch.einsum("td,d->t", soft_delta, unit_vector.float())
    direct_projection = torch.einsum("td,d->t", direct_delta, unit_vector.float())
    direct_logp = direct_logits.float().log_softmax(-1)
    soft_logp = soft_logits.float().log_softmax(-1)
    kl = (direct_logp.exp() * (direct_logp - soft_logp)).sum(-1)
    def mean(values, start, end=None):
        return float(values[start:end].mean())
    return {
        "mean_activation_cosine": float(position_cosine.mean()),
        "trajectory_activation_cosine": float(torch.nn.functional.cosine_similarity(soft_delta.flatten(), direct_delta.flatten(), dim=0)),
        "mean_activation_mse": float(position_mse.mean()),
        "mean_soft_projection": float(soft_projection.mean()),
        "mean_direct_projection": float(direct_projection.mean()),
        "mean_forward_kl": float(kl.mean()),
        "position_0_kl": float(kl[0]),
        "positions_1_8_kl": mean(kl, 1, 9),
        "positions_9_32_kl": mean(kl, 9, 33),
        "positions_33_plus_kl": mean(kl, 33),
        "position_cosine": position_cosine.cpu().tolist(),
        "position_mse": position_mse.cpu().tolist(),
        "soft_projection": soft_projection.cpu().tolist(),
        "direct_projection": direct_projection.cpu().tolist(),
        "forward_kl": kl.cpu().tolist(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if "metrics" in r]
    output = {"n": len(complete)}
    for key in (
        "mean_activation_cosine", "trajectory_activation_cosine", "mean_activation_mse",
        "mean_soft_projection", "mean_direct_projection", "mean_forward_kl", "position_0_kl",
        "positions_1_8_kl", "positions_9_32_kl", "positions_33_plus_kl",
    ):
        output[key] = float(np.mean([r["metrics"][key] for r in complete])) if complete else None
    judged = [r for r in complete if len(r.get("judgments", {})) == 3]
    output["judged_prompts"] = len(judged)
    for condition in ("baseline", "direct", "soft_prefill"):
        output[f"{condition}_successes"] = sum(int(r["judgments"][condition]["success"]) for r in judged)
        output[f"{condition}_asr"] = output[f"{condition}_successes"] / len(judged) if judged else None
    return output


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for row in rows:
        x = np.arange(len(row["metrics"]["position_cosine"]))
        axes[0].plot(x, row["metrics"]["position_cosine"], alpha=.65)
        axes[1].plot(x, row["metrics"]["forward_kl"], alpha=.65)
        axes[2].plot(x, row["metrics"]["soft_projection"], alpha=.65)
        axes[2].plot(x, row["metrics"]["direct_projection"], alpha=.35, linestyle="--")
    axes[0].axhline(.8, color="black", linestyle=":")
    axes[0].set(xlabel="response position", ylabel="soft/direct delta cosine")
    axes[1].set(xlabel="response position", ylabel="forward KL", yscale="symlog")
    axes[2].set(xlabel="response position", ylabel="steering projection")
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "prefill_vs_direct.png", dpi=180)
    plt.close(figure)


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    config_hash = fingerprint(config)
    writer = ArtifactWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    if mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError("fresh run refuses an existing checkpoint")
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    if mode == "resume" and checkpoint.get("config_fingerprint") != config_hash:
        raise ValueError("resume config fingerprint mismatch")
    dataset = Path(config["dataset_csv"])
    vector_path = Path(config["vector_path"])
    soft_path = Path(config["source_soft_run_dir"]) / config["source_soft_prompt"]
    for path, expected in ((vector_path, config["vector_sha256"]), (soft_path, config["source_soft_prompt_sha256"])):
        if file_sha256(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {path}")
    with dataset.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    selection = [{"dataset_row": i, "behavior_id": hashlib.sha256(source_rows[i]["goal"].encode()).hexdigest()[:12], "prompt": source_rows[i]["goal"]} for i in config["evaluation_rows"]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    config.update(run_id=output.name, config_fingerprint=config_hash)
    writer.json("resolved_config.json", config)
    writer.json("selection.json", {"dataset_sha256": file_sha256(dataset), "rows": selection})
    writer.json("source_artifacts.json", {
        "direct_run_id": config["source_direct_run_id"], "vector_sha256": config["vector_sha256"],
        "soft_run_id": config["source_soft_run_id"], "soft_prompt_sha256": config["source_soft_prompt_sha256"],
    })
    torch.manual_seed(config["generation_seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vector = torch.tensor(np.load(vector_path), device=model.device, dtype=torch.float32)
    unit = vector / vector.norm()
    soft = torch.load(soft_path, map_location=model.device, weights_only=False).float()
    if tuple(soft.shape) != (config["suffix_length"], model.config.hidden_size):
        raise ValueError("soft-prompt shape mismatch")
    hook = (config["module_index"], unit, config["direct_additive_norm"], "plain")
    rows = list(checkpoint.get("rows", []))
    phase = checkpoint.get("phase", "measure")
    errors = int(checkpoint.get("error_count", 0))
    stopped = False
    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    def save(active_phase: str):
        summary = aggregate(rows)
        completed = len(rows) if active_phase == "measure" else sum(len(r.get("judgments", {})) for r in rows)
        total = len(selection) if active_phase == "measure" else 3 * len(selection)
        progress = {"run_id": output.name, "config_fingerprint": config_hash, "phase": active_phase,
            "completed": completed, "total": total, "elapsed_seconds": time.monotonic()-started,
            "throughput_per_second": completed/max(time.monotonic()-started, 1e-9), "error_count": errors, **summary}
        writer.json("progress.json", progress)
        writer.json("checkpoint.json", {"config_fingerprint": config_hash, "phase": active_phase, "rows": rows, "error_count": errors})
        print(json.dumps(progress), flush=True)
    if phase == "measure":
        for item in tqdm(selection[len(rows):], initial=len(rows), total=len(selection), desc="measure_prompts"):
            layout = build_layout(tokenizer, item["prompt"], config["suffix_length"])
            empty = None
            direct_tokens = greedy_continuation(model, tokenizer, layout, empty, config["max_new_tokens"], hook)
            baseline_tokens = greedy_continuation(model, tokenizer, layout, empty, config["max_new_tokens"])
            soft_tokens = greedy_continuation(model, tokenizer, layout, soft, config["max_new_tokens"])
            baseline_hidden, _ = forward_trajectory(model, layout, empty, direct_tokens, config["hidden_state_index"])
            soft_hidden, soft_logits = forward_trajectory(model, layout, soft, direct_tokens, config["hidden_state_index"])
            direct_hidden, direct_logits = forward_trajectory(model, layout, empty, direct_tokens, config["hidden_state_index"], hook)
            row = {**item, "direct_continuation_token_ids": direct_tokens.tolist(), "responses": {
                "baseline": tokenizer.decode(baseline_tokens, skip_special_tokens=True),
                "direct": tokenizer.decode(direct_tokens, skip_special_tokens=True),
                "soft_prefill": tokenizer.decode(soft_tokens, skip_special_tokens=True),
            }, "metrics": trajectory_metrics(baseline_hidden, soft_hidden, direct_hidden, soft_logits, direct_logits, unit), "judgments": {}}
            rows.append(row)
            writer.jsonl("paired_generations.jsonl", row)
            save("measure")
            if stopped:
                return {"status": "stopped", "phase": "measure"}
        phase = "judge"
        save(phase)
    pending = [(r, c) for r in rows for c in ("baseline", "direct", "soft_prefill") if c not in r["judgments"]]
    for index, (row, condition) in enumerate(tqdm(pending, desc="openai_judge"), 1):
        try:
            row["judgments"][condition] = judge_response(row["prompt"], row["responses"][condition], config["judge"])
        except Exception:
            errors += 1
            save("judge")
            raise
        writer.jsonl("openai_judgments.jsonl", {"behavior_id": row["behavior_id"], "condition": condition, **row["judgments"][condition]})
        if index % config["judge"]["checkpoint_every"] == 0 or index == len(pending):
            save("judge")
        if stopped:
            save("judge")
            return {"status": "stopped", "phase": "judge"}
    summary = aggregate(rows)
    result = {"status": "complete", "run_id": output.name, "activation_generalization": summary["trajectory_activation_cosine"] >= config["activation_cosine_gate"], **summary}
    writer.json("paired_results.json", rows)
    writer.json("results.json", result)
    plot(rows, output)
    (output / "RESULTS.md").write_text(
        "# Held-out soft-prefill vs direct-steering diagnostic\n\n"
        f"- Mean trajectory cosine: {summary['trajectory_activation_cosine']:.6f}\n"
        f"- Mean position cosine: {summary['mean_activation_cosine']:.6f}\n"
        f"- Position-0 / mean KL: {summary['position_0_kl']:.6f} / {summary['mean_forward_kl']:.6f}\n"
        f"- Baseline ASR: {summary['baseline_successes']}/5 ({summary['baseline_asr']:.0%})\n"
        f"- Direct-steering ASR: {summary['direct_successes']}/5 ({summary['direct_asr']:.0%})\n"
        f"- Soft-prefill ASR: {summary['soft_prefill_successes']}/5 ({summary['soft_prefill_asr']:.0%})\n"
        f"- Activation-generalization gate: {result['activation_generalization']}\n"
    )
    writer.json("checkpoint.json", {"config_fingerprint": config_hash, "phase": "complete", "rows": rows, "error_count": errors})
    writer.json("progress.json", {"run_id": output.name, "phase": "complete", "completed": 15, "total": 15, "error_count": errors, **summary})
    if commit:
        commit()
    return result
