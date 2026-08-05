"""Build and evaluate a jailbreak steering vector from LLM-LAT pairs.

The selected residual-stream layer is chosen by held-out linear-probe
accuracy.  The direction itself is computed only from the training split as
mean(harmful compliance) - mean(refusal). Activation collection uses Hugging
Face hidden states, while evaluation and generation steer with a PyTorch
forward hook. Both extraction and generation are GPU-batched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CheckpointCallback = Callable[[], None]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def fingerprint(config: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()


def load_pairs(config: dict) -> list[dict]:
    dataset = load_dataset(config["dataset"], split=config["split"])
    required = ("prompt", "chosen", "rejected")
    valid = [row for row in dataset if all(isinstance(row.get(key), str) and row[key].strip() for key in required)]
    if len(valid) < config["pair_count"]:
        raise ValueError(f"requested {config['pair_count']} pairs, but only {len(valid)} complete rows exist")
    indices = np.random.default_rng(config["data_seed"]).choice(len(valid), config["pair_count"], replace=False)
    return [{"pair_id": pair_id, "source_index": int(index), **{key: valid[index][key] for key in required}}
            for pair_id, index in enumerate(indices)]


def paired_examples(pairs: list[dict]) -> list[dict]:
    examples = []
    for pair in pairs:
        common = {key: pair[key] for key in ("pair_id", "source_index", "prompt")}
        examples.append({**common, "response": pair["chosen"], "label": 0, "label_name": "refusal"})
        examples.append({**common, "response": pair["rejected"], "label": 1, "label_name": "harmful_compliance"})
    return examples


def pair_split(pair_count: int, train_pairs: int, seed: int) -> tuple[set[int], set[int]]:
    order = np.random.default_rng(seed).permutation(pair_count)
    return set(map(int, order[:train_pairs])), set(map(int, order[train_pairs:]))


def tokenize_examples(tokenizer, examples: list[dict]) -> list[dict]:
    special = set(tokenizer.all_special_ids)
    tokenized = []
    for example in tqdm(examples, desc="Tokenizing paired responses", unit="response"):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}], tokenize=True, add_generation_prompt=True
        )
        response_ids = tokenizer(example["response"], add_special_tokens=False).input_ids
        valid = [position for position, token_id in enumerate(response_ids) if token_id not in special]
        if not valid:
            raise ValueError(f"pair {example['pair_id']} has no non-special response token")
        tokenized.append({**example, "input_ids": prompt_ids + response_ids,
                          "measurement_index": len(prompt_ids) + valid[-1]})
    return tokenized


@torch.inference_mode()
def extract_activations(model, tokenizer, examples: list[dict], batch_size: int, state_path: Path,
                        checkpoint: dict, checkpoint_every: int, callback: CheckpointCallback) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activations: list[np.ndarray] = []
    labels: list[int] = []
    pair_ids: list[int] = []
    if state_path.exists():
        saved = np.load(state_path)
        activations = list(saved["activations"])
        labels = saved["labels"].tolist()
        pair_ids = saved["pair_ids"].tolist()
    start = len(labels)
    if start > len(examples):
        raise ValueError("activation checkpoint contains more rows than the configured dataset")
    started = time.monotonic()
    for offset in tqdm(range(start, len(examples), batch_size), initial=start // batch_size,
                       total=(len(examples) + batch_size - 1) // batch_size,
                       desc="Extracting all-layer activations", unit="batch"):
        batch = examples[offset:offset + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
        attention_mask = torch.zeros_like(input_ids)
        positions = []
        for row_index, row in enumerate(batch):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.tensor(row["input_ids"], device=model.device)
            attention_mask[row_index, :length] = 1
            positions.append(row["measurement_index"])
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        stacked = torch.stack(output.hidden_states, dim=1)
        selected = stacked[torch.arange(len(batch), device=model.device), :, torch.tensor(positions, device=model.device)]
        activations.extend(selected.float().cpu().numpy())
        labels.extend(row["label"] for row in batch)
        pair_ids.extend(row["pair_id"] for row in batch)
        completed = len(labels)
        if completed % checkpoint_every < len(batch) or completed == len(examples):
            atomic_npz(state_path, activations=np.asarray(activations), labels=np.asarray(labels), pair_ids=np.asarray(pair_ids))
            elapsed = time.monotonic() - started
            metric = {"phase": "extract", "completed": completed, "total": len(examples),
                      "elapsed_sec": round(elapsed, 2), "responses_per_sec": round((completed - start) / max(elapsed, 1e-9), 3),
                      "class_counts": {"refusal": int(np.count_nonzero(np.asarray(labels) == 0)),
                                       "harmful_compliance": int(np.count_nonzero(np.asarray(labels) == 1))},
                      "retry_count": checkpoint.get("retry_count", 0), "run_id": checkpoint["run_id"]}
            checkpoint.update({"stage": "extract", "next_example": completed, "latest_metric": metric})
            atomic_json(state_path.parent / "checkpoint.json", checkpoint)
            atomic_json(state_path.parent / "progress.json", metric)
            print(json.dumps({"event": "progress", **metric}), flush=True)
            callback()
    return np.asarray(activations), np.asarray(labels), np.asarray(pair_ids)


def fit_layer_probes(activations: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray,
                     train_pairs: set[int], test_pairs: set[int], seed: int) -> dict:
    train = np.array([pair_id in train_pairs for pair_id in pair_ids])
    test = np.array([pair_id in test_pairs for pair_id in pair_ids])
    scores, probes = [], []
    for layer in tqdm(range(1, activations.shape[1]), desc="Fitting held-out layer probes", unit="layer"):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1))
        probe.fit(activations[train, layer], labels[train])
        scores.append(float(probe.score(activations[test, layer], labels[test])))
        probes.append(probe)
    best_index = int(np.argmax(scores))
    best_layer = best_index + 1
    refusal_mean = activations[train & (labels == 0), best_layer].mean(axis=0)
    harmful_mean = activations[train & (labels == 1), best_layer].mean(axis=0)
    raw = harmful_mean - refusal_mean
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("cannot normalize a zero/non-finite steering direction")
    return {"train_mask": train, "test_mask": test, "scores": scores, "best_layer": best_layer,
            "probe": probes[best_index], "refusal_mean": refusal_mean, "harmful_mean": harmful_mean,
            "direction": raw / norm, "raw_norm": norm}


def make_steering_hook(vector: torch.Tensor):
    def hook(_module, _inputs, output):
        hidden, *rest = output if isinstance(output, tuple) else (output,)
        hidden = hidden + vector
        return (hidden, *rest) if isinstance(output, tuple) else hidden
    return hook


@torch.inference_mode()
def sweep_probe(model, tokenizer, examples: list[dict], layer: int, direction: np.ndarray, probe,
                strengths: list[float], threshold: float, batch_size: int) -> list[dict]:
    """Score held-out examples after real hooked forward-pass interventions."""
    truth = np.asarray([row["label"] for row in examples])
    sweep = []
    for strength in tqdm(strengths, desc="Hooked classifier sweep", unit="strength"):
        vector = torch.as_tensor(direction, device=model.device, dtype=model.dtype) * float(strength)
        handle = model.model.layers[layer - 1].register_forward_hook(make_steering_hook(vector))
        values = []
        try:
            for offset in range(0, len(examples), batch_size):
                batch = examples[offset:offset + batch_size]
                width = max(len(row["input_ids"]) for row in batch)
                input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                positions = []
                for row_index, row in enumerate(batch):
                    length = len(row["input_ids"])
                    input_ids[row_index, :length] = torch.tensor(row["input_ids"], device=model.device)
                    attention_mask[row_index, :length] = 1
                    positions.append(row["measurement_index"])
                output = model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True, use_cache=False)
                # hidden_states[0] is embeddings; hidden_states[layer] is the
                # hooked output of transformer block layer - 1.
                state = output.hidden_states[layer]
                selected = state[torch.arange(len(batch), device=model.device),
                                 torch.tensor(positions, device=model.device)]
                values.append(selected.float().cpu().numpy())
        finally:
            handle.remove()
        probability = probe.predict_proba(np.concatenate(values))[..., 1]
        predicted = (probability >= 0.5).astype(int)
        harmful = probability[truth == 1]
        sweep.append({"strength": float(strength), "classifier_accuracy": float(np.mean(predicted == truth)),
                      "mean_jailbreak_probability": float(np.mean(probability)),
                      "mean_harmful_pair_jailbreak_probability": float(np.mean(harmful)),
                      "fraction_over_probability_threshold": float(np.mean(probability >= threshold)),
                      "harmful_fraction_over_probability_threshold": float(np.mean(harmful >= threshold))})
    return sweep


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[dict], layer: int, direction: np.ndarray, strengths: list[float],
             config: dict, rows_path: Path, checkpoint: dict, callback: CheckpointCallback) -> int:
    existing = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            existing.add((float(row["strength"]), int(row["generation_index"])))
    total = len(strengths) * config["generations_per_strength"]
    started = time.monotonic()
    with rows_path.open("a") as stream:
        for strength in strengths:
            todo = [index for index in range(config["generations_per_strength"])
                    if (float(strength), index) not in existing]
            vector = torch.as_tensor(direction, device=model.device, dtype=model.dtype) * float(strength)
            handle = model.model.layers[layer - 1].register_forward_hook(make_steering_hook(vector))
            try:
                for offset in tqdm(range(0, len(todo), config["generation_batch_size"]),
                                   desc=f"Generating strength {strength:g}", unit="batch", leave=False):
                    indices = todo[offset:offset + config["generation_batch_size"]]
                    selected = [prompts[index % len(prompts)] for index in indices]
                    text = [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                          tokenize=False, add_generation_prompt=True) for row in selected]
                    encoded = tokenizer(text, padding=True, return_tensors="pt").to(model.device)
                    seed = config["generation_seed"] + int((float(strength) + 100) * 10_000) + offset
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    output = model.generate(**encoded, do_sample=True, temperature=config["temperature"],
                                            top_p=config["top_p"], max_new_tokens=config["max_new_tokens"],
                                            pad_token_id=tokenizer.pad_token_id, use_cache=True)
                    prompt_width = encoded.input_ids.shape[1]
                    for row_index, (generation_index, source) in enumerate(zip(indices, selected)):
                        response = tokenizer.decode(output[row_index, prompt_width:], skip_special_tokens=True)
                        record = {"strength": float(strength), "generation_index": generation_index,
                                  "source_index": source["source_index"], "prompt": source["prompt"],
                                  "response": response, "seed_batch": seed, "temperature": config["temperature"],
                                  "top_p": config["top_p"], "max_new_tokens": config["max_new_tokens"]}
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        existing.add((float(strength), generation_index))
                    elapsed = time.monotonic() - started
                    metric = {"phase": "generate", "completed": len(existing), "total": total,
                              "strength": float(strength), "elapsed_sec": round(elapsed, 2),
                              "generations_per_sec": round(len(existing) / max(elapsed, 1e-9), 3),
                              "best_layer": layer, "retry_count": checkpoint.get("retry_count", 0),
                              "run_id": checkpoint["run_id"]}
                    checkpoint.update({"stage": "generate", "completed_generations": len(existing), "latest_metric": metric})
                    atomic_json(rows_path.parent / "checkpoint.json", checkpoint)
                    atomic_json(rows_path.parent / "progress.json", metric)
                    print(json.dumps({"event": "progress", **metric}), flush=True)
                    callback()
            finally:
                handle.remove()
    return len(existing)


def validate_config(config: dict) -> None:
    required = {"model_id", "dataset", "split", "pair_count", "train_pairs", "data_seed", "split_seed",
                "activation_batch_size", "checkpoint_every", "sweep_strengths", "probability_threshold",
                "generations_per_strength", "generation_batch_size", "generation_seed", "temperature", "top_p",
                "max_new_tokens"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if not 0 < config["train_pairs"] < config["pair_count"]:
        raise ValueError("train_pairs must be between zero and pair_count")
    if config["generations_per_strength"] != 10:
        raise ValueError("this experiment requires exactly 10 generations per selected strength")
    if not 0 < config["probability_threshold"] < 1:
        raise ValueError("probability_threshold must be between zero and one")


def resolve_output(config: dict, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    configured = config.get("output_dir")
    if configured:
        return Path(configured)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    return Path("steering_vectors/runs") / f"{timestamp}_llm-lat-llama32-1b-jailbreak-direction"


def run(config_path: Path, run_mode: str = "fresh", output_dir: Path | None = None,
        checkpoint_callback: CheckpointCallback = lambda: None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    output = resolve_output(config, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {**config, "output_dir": str(output)}
    config_hash = fingerprint(config)
    checkpoint_path = output / "checkpoint.json"
    state_path = output / "activation_state.npz"
    run_id = output.name
    if run_mode == "fresh":
        if checkpoint_path.exists():
            raise FileExistsError(f"fresh run refuses to overwrite existing checkpoint: {checkpoint_path}")
        checkpoint = {"status": "running", "stage": "start", "run_id": run_id,
                      "config_fingerprint": config_hash, "retry_count": 0}
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    elif run_mode == "resume":
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("config_fingerprint") != config_hash:
            raise ValueError("resume config fingerprint does not match the checkpoint")
        if checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError(f"cannot resume checkpoint with status {checkpoint.get('status')!r}")
        checkpoint["status"] = "running"
        checkpoint["retry_count"] = int(checkpoint.get("retry_count", 0)) + 1
    else:
        raise ValueError("run_mode must be fresh or resume")
    atomic_json(checkpoint_path, checkpoint)
    checkpoint_callback()

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], token=os.environ.get("HF_TOKEN"), use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], token=os.environ.get("HF_TOKEN"),
                                                 torch_dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    pairs = load_pairs(config)
    examples = tokenize_examples(tokenizer, paired_examples(pairs))
    activations, labels, pair_ids = extract_activations(
        model, tokenizer, examples, config["activation_batch_size"], state_path, checkpoint,
        config["checkpoint_every"], checkpoint_callback
    )
    train_pairs, test_pairs = pair_split(config["pair_count"], config["train_pairs"], config["split_seed"])
    fitted = fit_layer_probes(activations, labels, pair_ids, train_pairs, test_pairs, config["split_seed"])
    np.save(output / "refusal_mean.npy", fitted["refusal_mean"])
    np.save(output / "harmful_mean.npy", fitted["harmful_mean"])
    np.save(output / "steering_vector.npy", fitted["direction"])
    joblib.dump(fitted["probe"], output / "linear_probe.joblib")
    heldout_examples = [row for row in examples if row["pair_id"] in test_pairs]
    sweep = sweep_probe(model, tokenizer, heldout_examples, fitted["best_layer"], fitted["direction"],
                        fitted["probe"], config["sweep_strengths"], config["probability_threshold"],
                        config["activation_batch_size"])
    selected_strengths = [row["strength"] for row in sweep
                          if row["mean_harmful_pair_jailbreak_probability"] >= config["probability_threshold"]]
    metric = {"phase": "sweep", "completed": len(sweep), "total": len(config["sweep_strengths"]),
              "best_layer": fitted["best_layer"], "best_metric": max(fitted["scores"]),
              "latest_objective": sweep[-1], "selected_strengths": selected_strengths,
              "vector_norm": fitted["raw_norm"], "run_id": run_id,
              "retry_count": checkpoint.get("retry_count", 0)}
    checkpoint.update({"stage": "sweep", "latest_metric": metric})
    atomic_json(checkpoint_path, checkpoint)
    atomic_json(output / "progress.json", metric)
    print(json.dumps({"event": "progress", **metric}), flush=True)
    checkpoint_callback()

    heldout_harmful = [pair for pair in pairs if pair["pair_id"] in test_pairs]
    generated_count = generate(model, tokenizer, heldout_harmful, fitted["best_layer"], fitted["direction"],
                               selected_strengths, config, output / "generations.jsonl", checkpoint,
                               checkpoint_callback) if selected_strengths else 0
    result = {"config": config, "run_id": run_id, "model_id": config["model_id"], "dataset": config["dataset"],
              "pair_split": {"train": len(train_pairs), "held_out": len(test_pairs)},
              "best_layer": fitted["best_layer"], "best_layer_accuracy": max(fitted["scores"]),
              "layer_accuracy": fitted["scores"], "activation_position": "last non-special assistant response token",
              "direction": "normalized mean(harmful rejected) - mean(refusal chosen), training pairs only",
              "raw_direction_norm": fitted["raw_norm"], "sweep": sweep,
              "selection_rule": f"mean held-out harmful-pair jailbreak probability >= {config['probability_threshold']}",
              "generation_strengths": selected_strengths, "generations_per_strength": config["generations_per_strength"],
              "generation_count": generated_count, "artifacts": {"vector": "steering_vector.npy", "probe": "linear_probe.joblib",
              "activations": "activation_state.npz", "generations": "generations.jsonl", "plot": "classifier_sweep.png"}}
    atomic_json(output / "results.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    strengths = [row["strength"] for row in sweep]
    axes[0].plot(strengths, [row["classifier_accuracy"] for row in sweep], marker="o")
    axes[0].set(xlabel="Steering strength", ylabel="Held-out classifier accuracy", ylim=(0, 1))
    axes[1].plot(strengths, [row["mean_harmful_pair_jailbreak_probability"] for row in sweep], marker="o")
    axes[1].axhline(config["probability_threshold"], color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Steering strength", ylabel="Mean jailbreak probability", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "classifier_sweep.png", dpi=180)
    plt.close(fig)
    (output / "RESULTS.md").write_text(
        "# LLM-LAT Llama-3.2-1B jailbreak steering\n\n"
        f"- Best layer: {fitted['best_layer']}\n- Held-out probe accuracy: {max(fitted['scores']):.4f}\n"
        f"- Raw direction norm: {fitted['raw_norm']:.4f}\n- Generated strengths: {selected_strengths}\n"
        f"- Generations: {generated_count} ({config['generations_per_strength']} per selected strength)\n"
    )
    final_metric = {"phase": "completed", "completed": generated_count, "total": generated_count,
                    "best_layer": fitted["best_layer"], "best_metric": max(fitted["scores"]),
                    "selected_strengths": selected_strengths, "run_id": run_id,
                    "retry_count": checkpoint.get("retry_count", 0)}
    checkpoint.update({"status": "completed", "stage": "completed", "latest_metric": final_metric})
    atomic_json(checkpoint_path, checkpoint)
    atomic_json(output / "progress.json", final_metric)
    checkpoint_callback()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config, arguments.run_mode, arguments.output_dir), indent=2))
