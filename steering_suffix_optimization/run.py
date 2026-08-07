"""Checkpointed CLI for steering-vector suffix optimization.

The CLI intentionally accepts a JSONL prompt file so every selected row is an
explicit artifact. Dataset selection belongs in experiment preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .optimizer import ActivationTarget, SteeringSuffixOptimizer, direction_metrics


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_prompts(path: Path, tokenizer, system_prompt: str) -> tuple[list[str], list[torch.Tensor]]:
    texts = [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]
    ids = []
    for text in texts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids.append(rendered[0].cpu())
    return texts, ids


def run(config_path: Path, prompt_path: Path, output: Path, mode: str) -> dict:
    config = yaml.safe_load(config_path.read_text())
    fp = fingerprint(config)
    output.mkdir(parents=True, exist_ok=mode == "resume")
    checkpoint_path, progress_path = output / "checkpoint.json", output / "progress.json"
    if mode == "fresh" and checkpoint_path.exists():
        raise FileExistsError(f"fresh run already exists: {output}")
    checkpoint = json.loads(checkpoint_path.read_text()) if mode == "resume" else None
    if checkpoint and checkpoint["config_fingerprint"] != fp:
        raise ValueError("resume config fingerprint mismatch")
    (output / "config.yaml").write_text(config_path.read_text())
    atomic_json(output / "resolved_config.json", config)
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config.get("model_revision"),
                                                  torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config.get("model_revision"))
    texts, prompts = load_prompts(prompt_path, tokenizer, config.get("system_prompt", "You are a helpful assistant."))
    vector = torch.from_numpy(np.load(config["vector"]).astype("float32"))
    optimizer = SteeringSuffixOptimizer(model, tokenizer,
        ActivationTarget(config["layer"], vector, config.get("vector_scale", 1.0)),
        top_k=config["top_k"], candidate_batch_size=config["candidate_batch_size"],
        eval_chunk_size=config["evaluation_chunk_size"], norm_weight=config["norm_weight"])
    baselines = optimizer.baseline(prompts)
    generator = torch.Generator(device=optimizer.device).manual_seed(config["seed"])
    if checkpoint:
        suffix = torch.tensor(checkpoint["suffix_ids"], device=optimizer.device)
        start, best_loss, history = checkpoint["next_step"], checkpoint["best_loss"], checkpoint["history"]
        generator.set_state(torch.tensor(checkpoint["generator_state"], dtype=torch.uint8))
    else:
        allowed = [i for i in range(len(tokenizer)) if i not in optimizer.special_ids]
        rng = random.Random(config["seed"])
        suffix = torch.tensor([rng.choice(allowed) for _ in range(config["suffix_length"])], device=optimizer.device)
        start, best_loss, history = 0, float("inf"), []
    started = time.monotonic(); errors = 0
    for step in tqdm(range(start, config["steps"]), initial=start, total=config["steps"], desc="steering suffix"):
        suffix, loss, accepted = optimizer.step(suffix, prompts, baselines, generator)
        best_loss = min(best_loss, loss)
        with torch.no_grad():
            deltas = []
            for ids, base in zip(prompts, baselines):
                joined = torch.cat((optimizer.embedding(ids[None].to(optimizer.device)).detach(),
                                    optimizer.embedding(suffix)[None]), dim=1)
                mask = torch.ones(joined.shape[:2], device=optimizer.device, dtype=torch.long)
                deltas.append(optimizer._hidden(joined, mask)[0] - base)
            metrics = direction_metrics(torch.stack(deltas), optimizer.target.normalized(deltas[0].device, deltas[0].dtype))
        history.append({"step": step + 1, "loss": loss, "accepted": accepted, **metrics})
        payload = {"status": "running", "phase": "optimization", "completed": step + 1,
            "total": config["steps"], "elapsed_seconds": time.monotonic() - started,
            "throughput_steps_per_second": (step + 1 - start) / max(time.monotonic() - started, 1e-6),
            "config_fingerprint": fp, "run_id": output.name, "latest_metric": metrics,
            "best_loss": best_loss, "suffix_ids": suffix.tolist(), "suffix_text": tokenizer.decode(suffix),
            "next_step": step + 1, "history": history, "error_count": errors,
            "generator_state": generator.get_state().cpu().tolist()}
        atomic_json(checkpoint_path, payload); atomic_json(progress_path, payload)
        if (step + 1) % config["checkpoint_every"] == 0:
            print("METRIC " + json.dumps({k: payload[k] for k in ("phase", "completed", "total", "best_loss", "latest_metric", "run_id")}), flush=True)
    payload["status"] = "optimized_pending_behavioral_evaluation"
    atomic_json(checkpoint_path, payload); atomic_json(output / "results.json", payload)
    (output / "RESULTS.md").write_text(f"# Steering suffix optimization\n\n- Status: {payload['status']}\n- Suffix: `{payload['suffix_text']}`\n- Cosine: {metrics['cosine']:.4f}\n- Projection: {metrics['projection']:.4f}\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    args = parser.parse_args(); run(args.config, args.prompts, args.output, args.mode)


if __name__ == "__main__":
    main()
