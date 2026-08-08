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

from .fixed_rollout_ce import RunWriter, append_jsonl, fingerprint, jsonl_rows, sha256, steering


def continuation_full_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    prefix_length: int,
    full_length: int,
) -> torch.Tensor:
    if prefix_length < 1 or prefix_length >= full_length:
        raise ValueError("input must contain a nonempty prefix and continuation")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logit shapes differ")
    if teacher_logits.shape[1] != full_length:
        raise ValueError("logit sequence length does not match token sequence")
    teacher_lp = teacher_logits[0, prefix_length - 1 : -1].float().log_softmax(-1)
    student_lp = student_logits[0, prefix_length - 1 : -1].float().log_softmax(-1)
    return (teacher_lp.exp() * (teacher_lp - student_lp)).sum(-1)


def summarize_positions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maximum = max(len(row["kl_by_position"]) for row in records)
    result = []
    for position in range(maximum):
        values = np.asarray(
            [row["kl_by_position"][position] for row in records if position < len(row["kl_by_position"])],
            dtype=np.float64,
        )
        result.append(
            {
                "position": position,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p99": float(np.quantile(values, 0.99)),
            }
        )
    return result


def distribution_diagnostics(values: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    order = np.sort(values)[::-1]
    total = float(order.sum())
    mass = {}
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(fraction * len(order))))
        mass[f"top_{int(fraction * 100)}pct"] = float(order[:count].sum() / total)
    slope, intercept = np.polyfit(positions, values, 1)
    correlation = float(np.corrcoef(positions, values)[0, 1])
    maximum = int(positions.max()) + 1
    quarter_edges = np.linspace(0, maximum, 5, dtype=int)
    quarters = []
    for index, (left, right) in enumerate(zip(quarter_edges[:-1], quarter_edges[1:])):
        selected = values[(positions >= left) & (positions < right)]
        quarters.append({"quarter": index + 1, "start": int(left), "end_exclusive": int(right), "count": int(selected.size), "mean": float(selected.mean())})
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
        "mass_concentration": mass,
        "position_linear_slope": float(slope),
        "position_linear_intercept": float(intercept),
        "position_correlation": correlation,
        "quarters": quarters,
    }


def plot_diagnostics(output: Path, values: np.ndarray, positions: list[dict[str, Any]], bins: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    cap = float(np.quantile(values, 0.995))
    axes[0, 0].hist(values, bins=np.linspace(0, cap, bins + 1), color="#4c78a8")
    axes[0, 0].set(xlabel="Per-token full KL (nats; clipped at p99.5)", ylabel="Token count", title="Full-KL histogram")
    ordered = np.sort(values)
    axes[0, 1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered))
    axes[0, 1].set(xlabel="Per-token full KL (nats)", ylabel="Empirical CDF", xscale="symlog", title="Full-KL CDF")
    x = [row["position"] for row in positions]
    axes[1, 0].plot(x, [row["mean"] for row in positions], label="mean")
    axes[1, 0].plot(x, [row["median"] for row in positions], label="median")
    axes[1, 0].plot(x, [row["p90"] for row in positions], label="p90")
    axes[1, 0].set(xlabel="Continuation position (0 = first token)", ylabel="KL (nats)", title="KL by response position")
    axes[1, 0].legend()
    mass_order = np.sort(values)[::-1]
    axes[1, 1].plot(np.arange(1, len(values) + 1) / len(values), np.cumsum(mass_order) / mass_order.sum())
    axes[1, 1].set(xlabel="Fraction of tokens, highest KL first", ylabel="Fraction of total KL mass", title="KL mass concentration")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "full_kl_distribution.png", dpi=180)
    plt.close(figure)


@torch.inference_mode()
def score_record(model, record, vector, config) -> list[float]:
    full_ids = record["prefix_ids"] + record["continuation_ids"]
    ids = torch.tensor([full_ids], device=model.device)
    attention = torch.ones_like(ids)
    with steering(model, config["module_index"], vector, config["coefficient"]):
        teacher_logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    student_logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    values = continuation_full_kl(teacher_logits, student_logits, len(record["prefix_ids"]), len(full_ids))
    return values.cpu().tolist()


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    config_hash = fingerprint(config)
    writer = RunWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    if mode == "fresh" and checkpoint:
        raise FileExistsError("fresh run refuses an existing checkpoint")
    if mode == "resume" and checkpoint.get("config_fingerprint") != config_hash:
        raise ValueError("resume config fingerprint mismatch")

    cache_path, vector_path = Path(config["teacher_cache_path"]), Path(config["vector_path"])
    if sha256(cache_path) != config["teacher_cache_sha256"]:
        raise ValueError("teacher cache SHA-256 mismatch")
    if sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("vector SHA-256 mismatch")
    cache = jsonl_rows(cache_path)
    if len(cache) != config["expected_records"] or sum(len(row["continuation_ids"]) for row in cache) != config["expected_tokens"]:
        raise ValueError("source cache count mismatch")

    writer.json("resolved_config.json", config)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    writer.json("source_artifacts.json", {"source_run_id": config["source_run_id"], "teacher_cache_sha256": sha256(cache_path), "records": len(cache), "tokens": config["expected_tokens"], "vector_sha256": sha256(vector_path), "config_fingerprint": config_hash})
    writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "initializing_model", "completed": 0, "total": len(cache), "completed_fraction": 0.0, "elapsed_seconds": 0.0, "latest_ce": None, "best_gap": None, "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    vector = torch.tensor(np.load(vector_path), device=model.device, dtype=torch.float32)
    stopped = False

    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    records_path = output / "full_kl_records.jsonl"
    records = jsonl_rows(records_path)
    started = time.monotonic()
    for index, source in enumerate(tqdm(cache[len(records):], initial=len(records), total=len(cache), desc="full_kl"), start=len(records)):
        values = score_record(model, source, vector, config)
        if len(values) != len(source["continuation_ids"]):
            raise RuntimeError("continuation KL count mismatch")
        row = {"record_index": index, "behavior_id": source["behavior_id"], "rollout_index": source["rollout_index"], "token_count": len(values), "kl_sum": float(sum(values)), "kl_by_position": values}
        append_jsonl(records_path, row)
        records.append(row)
        if len(records) % config["checkpoint_every_records"] == 0 or len(records) == len(cache):
            token_count = sum(row["token_count"] for row in records)
            current = sum(row["kl_sum"] for row in records) / token_count
            discrepancy = current - config["source_sample_ce_gap"]
            progress = {"run_id": output.name, "config_fingerprint": config_hash, "phase": "full_kl", "completed": len(records), "total": len(cache), "completed_fraction": len(records)/len(cache), "elapsed_seconds": time.monotonic()-started, "latest_ce": current, "full_kl": current, "sample_ce_gap": config["source_sample_ce_gap"], "signed_discrepancy": discrepancy, "best_gap": abs(discrepancy), "tokens_scored": token_count, "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))}
            writer.progress(progress)
            writer.json("checkpoint.json", {"status": "stopped" if stopped else "running", "phase": "full_kl", "completed_records": len(records), "tokens_scored": token_count, "current_full_kl": current, "config_fingerprint": config_hash})
        if stopped:
            return {"status": "stopped", "completed": len(records)}

    values = np.asarray([value for row in records for value in row["kl_by_position"]], dtype=np.float64)
    response_positions = np.asarray([position for row in records for position in range(len(row["kl_by_position"]))], dtype=np.float64)
    position_summary = summarize_positions(records)
    diagnostics = distribution_diagnostics(values, response_positions)
    full_kl = float(values.mean())
    discrepancy = full_kl - config["source_sample_ce_gap"]
    passed = abs(discrepancy) <= config["pass_tolerance"]
    first = position_summary[0]["mean"]
    rest = float(np.mean([value for row in records for value in row["kl_by_position"][1:]]))
    results = {"run_id": output.name, "config_fingerprint": config_hash, "source_run_id": config["source_run_id"], "teacher_cache_sha256": config["teacher_cache_sha256"], "records": len(records), "tokens": int(values.size), "full_kl": full_kl, "source_sample_ce_gap": config["source_sample_ce_gap"], "signed_discrepancy": discrepancy, "absolute_discrepancy": abs(discrepancy), "pass_tolerance": config["pass_tolerance"], "passed": passed, "first_position_mean_kl": first, "remaining_positions_mean_kl": rest, "distribution": diagnostics}
    writer.json("position_summary.json", position_summary)
    writer.json("results.json", results)
    plot_diagnostics(output, values, position_summary, config["histogram_bins"])
    decision = "PASS: masking/alignment gate cleared; gradients may be planned next." if passed else "FAIL: audit boundary alignment before any gradient work."
    (output / "RESULTS.md").write_text(f"# Full-distribution KL cross-check\n\n- Full KL: **{full_kl:.6f} nats/token**\n- Source sampled CE gap: **{config['source_sample_ce_gap']:.6f} nats/token**\n- Signed discrepancy: **{discrepancy:+.6f} nats/token**\n- Absolute discrepancy: **{abs(discrepancy):.6f}** (tolerance {config['pass_tolerance']:.3f})\n- Result: **{'PASS' if passed else 'FAIL'}**\n- First-position mean KL: {first:.6f}\n- Remaining-position mean KL: {rest:.6f}\n- Position slope: {diagnostics['position_linear_slope']:+.6g} nats/token/position\n- Top 1% token KL-mass share: {diagnostics['mass_concentration']['top_1pct']:.3%}\n\n{decision}\n")
    writer.json("checkpoint.json", {"status": "complete", "phase": "complete", "config_fingerprint": config_hash, "full_kl": full_kl, "passed": passed})
    writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "complete", "completed": len(records), "total": len(records), "completed_fraction": 1.0, "elapsed_seconds": time.monotonic()-started, "latest_ce": full_kl, "full_kl": full_kl, "sample_ce_gap": config["source_sample_ce_gap"], "signed_discrepancy": discrepancy, "best_gap": abs(discrepancy), "tokens_scored": int(values.size), "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
    return results

