from __future__ import annotations

import csv
import hashlib
import json
import signal
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
from .trajectory_imitation import greedy_soft_generation


def fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in config.items() if k != "run_mode"}, sort_keys=True
        ).encode()
    ).hexdigest()


def baseline_equivalence(
    generated: list[dict[str, Any]], references: dict[int, str]
) -> dict[str, Any]:
    comparisons = [
        {
            "dataset_row": row["dataset_row"],
            "exact_match": row["response"] == references[row["dataset_row"]],
            "generated_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
            "reference_sha256": hashlib.sha256(
                references[row["dataset_row"]].encode()
            ).hexdigest(),
        }
        for row in generated
    ]
    return {
        "passed": len(comparisons) == len(references)
        and all(row["exact_match"] for row in comparisons),
        "matched": sum(row["exact_match"] for row in comparisons),
        "total": len(references),
        "comparisons": comparisons,
    }


def summarize(rows: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    baseline = np.array([int(row["judgments"]["baseline"]["success"]) for row in rows])
    result = {
        "baseline": {"successes": int(baseline.sum()), "asr": float(baseline.mean())}
    }
    for condition in conditions:
        flags = np.array([int(row["judgments"][condition]["success"]) for row in rows])
        result[condition] = {
            "successes": int(flags.sum()),
            "asr": float(flags.mean()),
            "failure_to_success": int(((baseline == 0) & (flags == 1)).sum()),
            "success_to_failure": int(((baseline == 1) & (flags == 0)).sum()),
            "incoherent_rate": float(
                np.mean([not row["judgments"][condition]["coherent"] for row in rows])
            ),
            "mean_repeated_trigram_fraction": float(
                np.mean(
                    [
                        row["quality"][condition]["repeated_trigram_fraction"]
                        for row in rows
                    ]
                )
            ),
            "eos_rate": float(
                np.mean([row["quality"][condition]["hit_eos"] for row in rows])
            ),
        }
    return result


def success_gate(summary, retention, config):
    control = summary["horizon_1"]["successes"]
    return {
        condition: (
            retention[condition] >= config["success"]["min_position_16_retention"]
            and summary[condition]["successes"]
            >= control + config["success"]["min_success_gain"]
            and summary[condition]["incoherent_rate"]
            <= config["success"]["max_incoherent_rate"]
            and summary[condition]["mean_repeated_trigram_fraction"]
            < config["success"]["max_mean_repeated_trigram_fraction"]
        )
        for condition in ("horizon_8", "horizon_32")
    }


def plot_results(output: Path, summary: dict[str, Any]) -> None:
    conditions = ["baseline", "horizon_1", "horizon_8", "horizon_32"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(conditions, [summary[c]["asr"] for c in conditions])
    axes[1].bar(conditions[1:], [summary[c]["incoherent_rate"] for c in conditions[1:]])
    axes[2].bar(
        conditions[1:],
        [summary[c]["mean_repeated_trigram_fraction"] for c in conditions[1:]],
    )
    axes[0].set(
        xlabel="Generation condition", ylabel="Attack success rate", ylim=(0, 1)
    )
    axes[1].set(
        xlabel="Soft-prompt condition", ylabel="Incoherent response rate", ylim=(0, 1)
    )
    axes[2].set(
        xlabel="Soft-prompt condition",
        ylabel="Mean repeated-trigram fraction",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "corrected_asr.png", dpi=180)
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
    source_dir = Path(config["source_run_dir"])
    source_resolved = json.loads((source_dir / "resolved_config.json").read_text())
    if source_resolved["config_fingerprint"] != config["source_config_fingerprint"]:
        raise ValueError("source fingerprint mismatch")
    for condition, expected in config["soft_prompt_sha256"].items():
        if file_sha256(source_dir / "soft_prompts" / f"{condition}.pt") != expected:
            raise ValueError(f"soft prompt checksum mismatch: {condition}")
    dataset_path = Path(config["dataset_csv"])
    if file_sha256(dataset_path) != config["dataset_sha256"]:
        raise ValueError("dataset checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    config.update(run_id=output.name, config_fingerprint=config_hash)
    writer.json("resolved_config.json", config)
    writer.json(
        "source_artifacts.json",
        {
            "source_run_id": config["source_run_id"],
            "source_config_fingerprint": config["source_config_fingerprint"],
            "soft_prompt_sha256": config["soft_prompt_sha256"],
            "reference_run_id": config["reference_baseline_run_id"],
            "implementation_commit": config["implementation_commit"],
        },
    )
    with dataset_path.open(newline="") as stream:
        source = list(csv.DictReader(stream))
    selection = [
        {
            "dataset_row": i,
            "behavior_id": hashlib.sha256(source[i]["goal"].encode()).hexdigest()[:12],
            "prompt": source[i]["goal"],
        }
        for i in config["evaluation_rows"]
    ]
    writer.json(
        "selection.json",
        {"dataset_sha256": file_sha256(dataset_path), "rows": selection},
    )
    if not checkpoint:
        initial = {
            "phase": "initializing",
            "completed": 0,
            "total": 50,
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        }
        writer.json("progress.json", initial)
        update_dashboard(output, initial)
        writer.json(
            "checkpoint.json",
            {
                "status": "running",
                "phase": "initializing",
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": 0,
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
    prompts = {
        condition: torch.load(
            source_dir / "soft_prompts" / f"{condition}.pt",
            map_location=model.device,
            weights_only=False,
        )
        for condition in config["conditions"]
    }
    reference_rows = json.loads(
        (Path(config["reference_baseline_run_dir"]) / "paired_results.json").read_text()
    )
    references = {
        row["dataset_row"]: row["responses"]["baseline"]
        for row in reference_rows
        if row["dataset_row"] in config["evaluation_rows"]
    }
    stopped = False

    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    gate_path = output / "baseline_generations.jsonl"
    baseline_rows = (
        [json.loads(x) for x in gate_path.read_text().splitlines() if x]
        if gate_path.exists()
        else []
    )
    for item in tqdm(
        selection[len(baseline_rows) :],
        initial=len(baseline_rows),
        total=len(selection),
        desc="baseline_gate",
    ):
        response, hit_eos = greedy_soft_generation(
            model,
            tokenizer,
            build_layout(tokenizer, item["prompt"], config["suffix_length"]),
            None,
            config["max_new_tokens"],
        )
        row = {**item, "response": response, "hit_eos": hit_eos}
        baseline_rows.append(row)
        writer.jsonl("baseline_generations.jsonl", row)
        progress = {
            "phase": "baseline_gate",
            "completed": len(baseline_rows),
            "total": len(selection),
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        }
        writer.json("progress.json", progress)
        update_dashboard(output, progress)
        writer.json(
            "checkpoint.json",
            {
                "status": "stopped" if stopped else "running",
                "phase": "baseline_gate",
                "completed": len(baseline_rows),
                "config_fingerprint": config_hash,
                "error_count": 0,
                "retry_count": 0,
            },
        )
        if stopped:
            return {"status": "stopped", **progress}
    equivalence = baseline_equivalence(baseline_rows, references)
    writer.json("baseline_equivalence.json", equivalence)
    if not equivalence["passed"]:
        result = {
            "status": "invalid",
            "reason": "baseline equivalence failed",
            "baseline_equivalence": equivalence,
        }
        writer.json("results.json", result)
        (output / "RESULTS.md").write_text(
            f"# Corrected trajectory evaluation\n\nBaseline equivalence failed: {equivalence['matched']}/{equivalence['total']}. No judging performed.\n"
        )
        final = {
            "phase": "invalid",
            "completed": equivalence["matched"],
            "total": equivalence["total"],
            "measurement_valid": False,
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        }
        writer.json("progress.json", final)
        writer.json("checkpoint.json", {"status": "complete", **final})
        update_dashboard(output, final)
        writer.commit()
        return result
    generation_path = output / "paired_generations.jsonl"
    rows = (
        [json.loads(x) for x in generation_path.read_text().splitlines() if x]
        if generation_path.exists()
        else []
    )
    for item in tqdm(
        selection[len(rows) :],
        initial=len(rows),
        total=len(selection),
        desc="suffix_generation",
    ):
        baseline = baseline_rows[len(rows)]
        responses = {"baseline": baseline["response"]}
        quality = {
            "baseline": {
                "hit_eos": baseline["hit_eos"],
                "repeated_trigram_fraction": repeated_trigram_fraction(
                    baseline["response"]
                ),
            }
        }
        layout = build_layout(tokenizer, item["prompt"], config["suffix_length"])
        for condition, suffix in prompts.items():
            response, hit_eos = greedy_soft_generation(
                model, tokenizer, layout, suffix, config["max_new_tokens"]
            )
            responses[condition] = response
            quality[condition] = {
                "hit_eos": hit_eos,
                "repeated_trigram_fraction": repeated_trigram_fraction(response),
            }
        row = {**item, "responses": responses, "quality": quality, "judgments": {}}
        rows.append(row)
        writer.jsonl("paired_generations.jsonl", row)
        progress = {
            "phase": "generate",
            "completed": len(rows),
            "total": len(selection),
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": 0,
        }
        writer.json("progress.json", progress)
        update_dashboard(output, progress)
        writer.json(
            "checkpoint.json",
            {"status": "stopped" if stopped else "running", **progress},
        )
        if stopped:
            return {"status": "stopped", **progress}
    partial = output / "paired_results.partial.json"
    if partial.exists():
        rows = json.loads(partial.read_text())
    pending = [
        (row, c)
        for row in rows
        for c in ["baseline", *config["conditions"]]
        if c not in row["judgments"]
    ]
    judged = sum(len(row["judgments"]) for row in rows)
    for row, condition in tqdm(pending, initial=judged, total=40, desc="judge"):
        result = judge(row["prompt"], row["responses"][condition], config["judge"])
        row["judgments"][condition] = result
        judged += 1
        writer.json("paired_results.partial.json", rows)
        judgments = [
            {"behavior_id": r["behavior_id"], "condition": c, **j}
            for r in rows
            for c, j in r["judgments"].items()
        ]
        (output / "openai_judgments.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in judgments)
        )
        writer.commit()
        progress = {
            "phase": "judge",
            "completed": judged,
            "total": 40,
            "run_id": output.name,
            "config_fingerprint": config_hash,
            "error_count": 0,
            "retry_count": sum(x["retries"] for x in judgments),
        }
        writer.json("progress.json", progress)
        update_dashboard(output, progress)
        writer.json(
            "checkpoint.json",
            {"status": "stopped" if stopped else "running", **progress},
        )
        if stopped:
            return {"status": "stopped", **progress}
    summary = summarize(rows, config["conditions"])
    retention = json.loads((source_dir / "results.json").read_text())[
        "trajectory_evaluations"
    ]
    retention = {
        c: retention[c]["projection_retention"]["16"] for c in config["conditions"]
    }
    gates = success_gate(summary, retention, config)
    results = {
        "status": "complete",
        "measurement_valid": True,
        "baseline_equivalence": equivalence,
        "summary": summary,
        "position_16_retention": retention,
        "condition_gate_passes": gates,
        "success": any(gates.values()),
    }
    writer.json("paired_results.json", rows)
    writer.json("results.json", results)
    plot_results(output, summary)
    (output / "RESULTS.md").write_text(
        "# Corrected trajectory evaluation\n\n"
        + f"- Baseline equivalence: {equivalence['matched']}/{equivalence['total']}\n"
        + "\n".join(
            f"- {c}: {summary[c]['successes']}/10 ASR, incoherence {summary[c]['incoherent_rate']:.0%}"
            for c in config["conditions"]
        )
        + f"\n- Gate passed: {results['success']}\n"
    )
    final = {
        "phase": "complete",
        "completed": 40,
        "total": 40,
        "measurement_valid": True,
        "baseline_asr": summary["baseline"]["asr"],
        **{f"{c}_asr": summary[c]["asr"] for c in config["conditions"]},
        "success": results["success"],
        "run_id": output.name,
        "config_fingerprint": config_hash,
        "error_count": 0,
        "retry_count": 0,
    }
    writer.json("progress.json", final)
    writer.json("checkpoint.json", {"status": "complete", **final})
    update_dashboard(output, final)
    writer.commit()
    return results
