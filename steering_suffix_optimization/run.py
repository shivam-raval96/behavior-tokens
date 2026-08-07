"""Checkpointed CLI for steering-vector suffix optimization.

The CLI intentionally accepts a JSONL prompt file so every selected row is an
explicit artifact. Dataset selection belongs in experiment preparation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import time
from collections.abc import Callable

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluation import jailbreak_success, paired_asr
from .optimizer import ActivationTarget, SteeringSuffixOptimizer, direction_metrics


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_prompt_texts(path: Path, config: dict) -> tuple[list[str], list[str]]:
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        column = config.get("prompt_column", "goal")
        train = [rows[index][column].strip() for index in config["train_indices"]]
        evaluation = [rows[index][column].strip() for index in config["evaluation_indices"]]
        return train, evaluation
    texts = [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]
    return texts[:config["optimization_rows"]], texts[config["optimization_rows"]:]


def token_layouts(texts: list[str], tokenizer, system_prompt: str,
                  suffix_length: int) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    marker = " !" * suffix_length
    marker_ids = torch.tensor(tokenizer(marker, add_special_tokens=False).input_ids)
    if len(marker_ids) != suffix_length:
        raise AssertionError("suffix marker must tokenize to exactly suffix_length tokens")
    clean, layouts = [], []
    for text in texts:
        plain = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        marked = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text + marker}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")[0].cpu()
        matches = [i for i in range(len(marked) - len(marker_ids) + 1)
                   if torch.equal(marked[i:i + len(marker_ids)], marker_ids)]
        if len(matches) != 1:
            raise AssertionError(f"expected one exact suffix marker span, found {len(matches)}")
        start = matches[0]
        clean.append(plain[0].cpu()); layouts.append((marked[:start], marked[start + len(marker_ids):]))
    return clean, layouts, marker_ids


def run(config_path: Path, prompt_path: Path, output: Path, mode: str,
        commit: Callable[[], None] | None = None) -> dict:
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
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config.get("model_revision"))
    train_texts, evaluation_texts = load_prompt_texts(prompt_path, config)
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    prompts, layouts, initial_suffix = token_layouts(train_texts, tokenizer, system_prompt, config["suffix_length"])
    atomic_json(output / "selection.json", {"train_indices": config.get("train_indices"),
        "evaluation_indices": config.get("evaluation_indices"), "train_prompts": train_texts,
        "evaluation_prompts": evaluation_texts})
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
        suffix = initial_suffix.to(optimizer.device)
        start, best_loss, history = 0, float("inf"), []
    started = time.monotonic(); errors = 0
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for step in tqdm(range(start, config["steps"]), initial=start, total=config["steps"], desc="steering suffix"):
            suffix, loss, accepted = optimizer.step(suffix, layouts, baselines, generator)
            best_loss = min(best_loss, loss)
            with torch.no_grad():
                deltas = []
                for (before, after), base in zip(layouts, baselines):
                    joined = torch.cat((optimizer.embedding(before[None].to(optimizer.device)).detach(),
                                        optimizer.embedding(suffix)[None],
                                        optimizer.embedding(after[None].to(optimizer.device)).detach()), dim=1)
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
            if commit: commit()
            if (step + 1) % config["checkpoint_every"] == 0:
                print("METRIC " + json.dumps({k: payload[k] for k in ("phase", "completed", "total", "best_loss", "latest_metric", "run_id")}), flush=True)
            if mode == "fresh" and config.get("stop_after_step") == step + 1:
                raise KeyboardInterrupt("intentional interruption test")
    except KeyboardInterrupt:
        payload.update(status="stopped", next_step=len(history))
        atomic_json(checkpoint_path, payload); atomic_json(progress_path, payload)
        if commit: commit()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    generations = []
    evaluation_clean, evaluation_layouts, _ = token_layouts(
        evaluation_texts, tokenizer, system_prompt, config["suffix_length"])
    for index, text in enumerate(tqdm(evaluation_texts, desc="paired ASR", unit="prompt")):
        def generate(with_suffix: bool) -> str:
            if with_suffix:
                before, after = evaluation_layouts[index]
                ids = torch.cat((before, suffix.cpu(), after))[None].to(optimizer.device)
            else:
                ids = evaluation_clean[index][None].to(optimizer.device)
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=config["generation"]["max_new_tokens"],
                                 do_sample=config["generation"]["do_sample"], pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        baseline_response, suffix_response = generate(False), generate(True)
        row = {"index": index, "prompt": text, "baseline_response": baseline_response,
               "suffix_response": suffix_response, "baseline_success": jailbreak_success(baseline_response),
               "suffix_success": jailbreak_success(suffix_response)}
        generations.append(row)
        with (output / "generations.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        payload.update(status="evaluating", phase="paired_asr", completed=index + 1,
                       total=len(evaluation_texts), asr=paired_asr(generations))
        atomic_json(checkpoint_path, payload); atomic_json(progress_path, payload)
        if commit: commit()
    payload["status"] = "complete"
    payload["asr"] = paired_asr(generations)
    atomic_json(checkpoint_path, payload); atomic_json(output / "results.json", payload)
    (output / "RESULTS.md").write_text(f"# Steering suffix optimization\n\n- Status: {payload['status']}\n- Suffix: `{payload['suffix_text']}`\n- Cosine: {metrics['cosine']:.4f}\n- Projection: {metrics['projection']:.4f}\n- Baseline ASR: {payload['asr']['baseline_asr']:.4f}\n- Suffix ASR: {payload['asr']['suffix_asr']:.4f}\n- ASR delta: {payload['asr']['asr_delta']:+.4f}\n")
    if commit: commit()
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
