from __future__ import annotations

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
from transformers import AutoModelForCausalLM

from .fixed_rollout_ce import RunWriter, append_jsonl, fingerprint, jsonl_rows, sha256, steering


def build_q(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature and top_p are out of range")
    probabilities = (logits.float() / temperature).softmax(-1)
    sorted_probabilities, sorted_ids = probabilities.sort(dim=-1, descending=True)
    cumulative_before = sorted_probabilities.cumsum(-1) - sorted_probabilities
    keep_sorted = cumulative_before < top_p
    kept = sorted_probabilities * keep_sorted
    retained_mass = kept.sum(-1)
    normalized = kept / retained_mass.unsqueeze(-1)
    q = torch.zeros_like(probabilities).scatter(-1, sorted_ids, normalized)
    support = torch.zeros_like(keep_sorted).scatter(-1, sorted_ids, keep_sorted)
    return q, support, retained_mass


def q_weighted_terms(teacher_logits, student_logits, temperature: float, top_p: float):
    teacher_logp = teacher_logits.float().log_softmax(-1)
    student_logp = student_logits.float().log_softmax(-1)
    q, support, retained_mass = build_q(teacher_logits, temperature, top_p)
    teacher_ce = -(q * teacher_logp).sum(-1)
    student_ce = -(q * student_logp).sum(-1)
    gap = student_ce - teacher_ce
    return {
        "q": q,
        "support": support,
        "retained_mass": retained_mass,
        "teacher_logp": teacher_logp,
        "teacher_ce": teacher_ce,
        "student_ce": student_ce,
        "gap": gap,
    }


def continuation_slice(logits: torch.Tensor, prefix_length: int, full_length: int):
    if prefix_length < 1 or prefix_length >= full_length or logits.shape[1] != full_length:
        raise ValueError("invalid continuation boundary")
    return logits[0, prefix_length - 1 : -1]


@torch.inference_mode()
def score_record(model, record, vector, config):
    full_ids = record["prefix_ids"] + record["continuation_ids"]
    ids = torch.tensor([full_ids], device=model.device)
    attention = torch.ones_like(ids)
    with steering(model, config["module_index"], vector, config["coefficient"]):
        teacher = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    student = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    teacher = continuation_slice(teacher, len(record["prefix_ids"]), len(full_ids))
    student = continuation_slice(student, len(record["prefix_ids"]), len(full_ids))
    terms = q_weighted_terms(teacher, student, config["temperature"], config["top_p"])
    q, support, teacher_logp = terms["q"], terms["support"], terms["teacher_logp"]
    sparse = []
    for position in range(q.shape[0]):
        token_ids = torch.nonzero(support[position], as_tuple=False).flatten()
        sparse.append(
            {
                "token_ids": token_ids.cpu().tolist(),
                "probabilities": q[position, token_ids].cpu().tolist(),
                "teacher_raw_logprobs": teacher_logp[position, token_ids].cpu().tolist(),
            }
        )
    outside_max = float(q.masked_select(~support).abs().max()) if (~support).any() else 0.0
    return {
        "teacher_ce_by_position": terms["teacher_ce"].cpu().tolist(),
        "student_ce_by_position": terms["student_ce"].cpu().tolist(),
        "gap_by_position": terms["gap"].cpu().tolist(),
        "support_counts": support.sum(-1).cpu().tolist(),
        "retained_mass": terms["retained_mass"].cpu().tolist(),
        "q_sum_error_max": float((q.sum(-1) - 1).abs().max()),
        "outside_support_max": outside_max,
        "sparse_q_targets": sparse,
    }


def flatten(records, key):
    return np.asarray([value for row in records for value in row[key]], dtype=np.float64)


def summarize_support(counts: np.ndarray, vocabulary_size: int):
    return {
        "minimum": int(counts.min()),
        "median": float(np.median(counts)),
        "mean": float(counts.mean()),
        "p90": float(np.quantile(counts, 0.90)),
        "p99": float(np.quantile(counts, 0.99)),
        "maximum": int(counts.max()),
        "fraction_single_token": float(np.mean(counts == 1)),
        "fraction_full_vocabulary": float(np.mean(counts == vocabulary_size)),
    }


def position_summary(records):
    maximum = max(len(row["gap_by_position"]) for row in records)
    result = []
    for position in range(maximum):
        selected = [row for row in records if position < len(row["gap_by_position"])]
        gaps = np.asarray([row["gap_by_position"][position] for row in selected])
        result.append(
            {
                "position": position,
                "count": len(selected),
                "teacher_floor_mean": float(np.mean([row["teacher_ce_by_position"][position] for row in selected])),
                "student_ce_mean": float(np.mean([row["student_ce_by_position"][position] for row in selected])),
                "gap_mean": float(gaps.mean()),
                "gap_median": float(np.median(gaps)),
                "gap_p90": float(np.quantile(gaps, 0.90)),
                "support_mean": float(np.mean([row["support_counts"][position] for row in selected])),
            }
        )
    return result


def plot(output: Path, gaps: np.ndarray, positions, supports: np.ndarray, bins: int):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    cap = float(np.quantile(gaps, 0.995))
    axes[0, 0].hist(gaps, bins=np.linspace(float(gaps.min()), cap, bins + 1))
    axes[0, 0].set(xlabel="q-weighted raw logprob gap (nats; clipped p99.5)", ylabel="Positions", title="Gap distribution")
    ordered = np.sort(gaps)
    axes[0, 1].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered))
    axes[0, 1].set(xlabel="q-weighted gap (nats)", ylabel="Empirical CDF", title="Gap CDF")
    x = [row["position"] for row in positions]
    axes[1, 0].plot(x, [row["teacher_floor_mean"] for row in positions], label="teacher floor")
    axes[1, 0].plot(x, [row["student_ce_mean"] for row in positions], label="student CE")
    axes[1, 0].plot(x, [row["gap_mean"] for row in positions], label="gap")
    axes[1, 0].set(xlabel="Continuation position", ylabel="Nats", title="q-weighted terms by position")
    axes[1, 0].legend()
    axes[1, 1].plot(x, [row["support_mean"] for row in positions])
    axes[1, 1].set(xlabel="Continuation position", ylabel="Mean surviving tokens", yscale="log", title="Top-p support by position")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "q_weighted_diagnostics.png", dpi=180)
    plt.close(figure)


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
    if sha256(cache_path) != config["teacher_cache_sha256"] or sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("source artifact SHA-256 mismatch")
    cache = jsonl_rows(cache_path)
    if len(cache) != config["expected_records"] or sum(len(row["continuation_ids"]) for row in cache) != config["expected_tokens"]:
        raise ValueError("source cache count mismatch")
    writer.json("resolved_config.json", config)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    writer.json("source_artifacts.json", {"source_run_id": config["source_run_id"], "teacher_cache_sha256": sha256(cache_path), "vector_sha256": sha256(vector_path), "records": len(cache), "tokens": config["expected_tokens"], "config_fingerprint": config_hash})
    writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "initializing_model", "completed": 0, "total": len(cache), "completed_fraction": 0.0, "elapsed_seconds": 0.0, "latest_ce": None, "best_gap": None, "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    vector = torch.tensor(np.load(vector_path), device=model.device, dtype=torch.float32)
    stopped = False
    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, terminate)
    records_path = output / "q_weighted_records.jsonl"
    records = jsonl_rows(records_path)
    started = time.monotonic()
    for index, source in enumerate(tqdm(cache[len(records):], initial=len(records), total=len(cache), desc="q_weighted"), start=len(records)):
        detail = score_record(model, source, vector, config)
        count = len(source["continuation_ids"])
        if any(len(detail[key]) != count for key in ("teacher_ce_by_position", "student_ce_by_position", "gap_by_position", "support_counts", "retained_mass", "sparse_q_targets")):
            raise RuntimeError("q-weighted continuation count mismatch")
        row = {"record_index": index, "behavior_id": source["behavior_id"], "rollout_index": source["rollout_index"], "token_count": count, **detail}
        append_jsonl(records_path, row); records.append(row)
        if len(records) % config["checkpoint_every_records"] == 0 or len(records) == len(cache):
            gaps = flatten(records, "gap_by_position")
            current = float(gaps.mean()); residual = current - config["source_sample_ce_gap"]
            progress = {"run_id": output.name, "config_fingerprint": config_hash, "phase": "q_weighted", "completed": len(records), "total": len(cache), "completed_fraction": len(records)/len(cache), "elapsed_seconds": time.monotonic()-started, "latest_ce": current, "q_weighted_gap": current, "sample_ce_gap": config["source_sample_ce_gap"], "signed_residual": residual, "best_gap": abs(residual), "tokens_scored": int(gaps.size), "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))}
            writer.progress(progress)
            writer.json("checkpoint.json", {"status": "stopped" if stopped else "running", "phase": "q_weighted", "completed_records": len(records), "tokens_scored": int(gaps.size), "current_gap": current, "config_fingerprint": config_hash})
        if stopped: return {"status": "stopped", "completed": len(records)}
    teacher = flatten(records, "teacher_ce_by_position"); student = flatten(records, "student_ce_by_position"); gaps = flatten(records, "gap_by_position"); supports = flatten(records, "support_counts"); retained = flatten(records, "retained_mass")
    exact_floor, exact_student, exact_gap = float(teacher.mean()), float(student.mean()), float(gaps.mean())
    residual = exact_gap - config["source_sample_ce_gap"]
    q_sum_error = max(row["q_sum_error_max"] for row in records); outside_max = max(row["outside_support_max"] for row in records)
    support_stats = summarize_support(supports, model.config.vocab_size)
    self_checks = {"q_sum_error_max": q_sum_error, "q_sum_passed": q_sum_error <= config["q_sum_tolerance"], "outside_support_max": outside_max, "outside_support_zero": outside_max == 0.0, "retained_mass_min": float(retained.min()), "retained_mass_at_least_top_p": bool(retained.min() >= config["top_p"] - 1e-6), "no_full_vocabulary_support": support_stats["maximum"] < model.config.vocab_size}
    passed = abs(residual) <= config["pass_tolerance"] and all((self_checks["q_sum_passed"], self_checks["outside_support_zero"], self_checks["retained_mass_at_least_top_p"], self_checks["no_full_vocabulary_support"]))
    positions = position_summary(records)
    results = {"run_id": output.name, "config_fingerprint": config_hash, "source_run_id": config["source_run_id"], "teacher_cache_sha256": config["teacher_cache_sha256"], "records": len(records), "tokens": int(gaps.size), "exact_q_teacher_floor": exact_floor, "exact_q_student_ce": exact_student, "exact_q_gap": exact_gap, "source_sample_teacher_floor": config["source_sample_teacher_floor"], "source_sample_student_ce": config["source_sample_student_ce"], "source_sample_ce_gap": config["source_sample_ce_gap"], "signed_gap_residual": residual, "absolute_gap_residual": abs(residual), "pass_tolerance": config["pass_tolerance"], "passed": passed, "self_checks": self_checks, "support": support_stats, "first_position_gap": positions[0]["gap_mean"], "remaining_positions_gap": float(np.mean([row["gap_by_position"][position] for row in records for position in range(1, len(row["gap_by_position"]))]))}
    writer.json("position_summary.json", positions); writer.json("results.json", results)
    plot(output, gaps, positions, supports, config["histogram_bins"])
    (output / "RESULTS.md").write_text(f"# q-weighted cross-check\n\n- Exact q-weighted teacher floor: **{exact_floor:.6f} nats/token**\n- Exact q-weighted student CE: **{exact_student:.6f} nats/token**\n- Exact q-weighted gap: **{exact_gap:.6f} nats/token**\n- Source sampled gap: **{config['source_sample_ce_gap']:.6f} nats/token**\n- Signed residual: **{residual:+.6f}**\n- Gate: **{'PASS' if passed else 'FAIL'}** (tolerance {config['pass_tolerance']:.3f})\n- Max q normalization error: {q_sum_error:.3g}\n- Max probability outside support: {outside_max:.3g}\n- Support min/median/mean/p90/p99/max: {support_stats['minimum']} / {support_stats['median']:.1f} / {support_stats['mean']:.1f} / {support_stats['p90']:.1f} / {support_stats['p99']:.1f} / {support_stats['maximum']}\n\nSparse q targets are in `q_weighted_records.jsonl`. GCG was not run.\n")
    writer.json("checkpoint.json", {"status": "complete", "phase": "complete", "config_fingerprint": config_hash, "exact_q_gap": exact_gap, "passed": passed})
    writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "complete", "completed": len(records), "total": len(records), "completed_fraction": 1.0, "elapsed_seconds": time.monotonic()-started, "latest_ce": exact_gap, "q_weighted_gap": exact_gap, "sample_ce_gap": config["source_sample_ce_gap"], "signed_residual": residual, "best_gap": abs(residual), "tokens_scored": int(gaps.size), "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
    return results
