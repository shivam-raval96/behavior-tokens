"""Extract a harmful-compliance minus refusal direction from LLM-LAT data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_state(path: Path, values: list[torch.Tensor], labels: list[int], pair_ids: list[int]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, activations=torch.stack(values).numpy(), labels=np.array(labels), pair_ids=np.array(pair_ids))
    temporary.replace(path)


def load_pairs(cfg: dict) -> list[dict]:
    dataset = load_dataset(cfg["dataset"], split=cfg["split"])
    valid = [row for row in dataset if all(isinstance(row.get(key), str) and row[key].strip() for key in ("prompt", "chosen", "rejected"))]
    if len(valid) < cfg["pair_count"]:
        raise RuntimeError(f"dataset has only {len(valid)} complete triples; need {cfg['pair_count']}")
    selected = np.random.default_rng(cfg["seed"]).choice(len(valid), size=cfg["pair_count"], replace=False)
    return [{"source_index": int(index), **{key: valid[index][key] for key in ("prompt", "chosen", "rejected")}} for index in selected]


def paired_examples(pairs: list[dict]) -> list[dict]:
    examples = []
    for pair_id, pair in enumerate(pairs):
        examples.extend((
            {"pair_id": pair_id, "source_index": pair["source_index"], "prompt": pair["prompt"], "response": pair["chosen"], "label": 0, "label_name": "refusal_chosen"},
            {"pair_id": pair_id, "source_index": pair["source_index"], "prompt": pair["prompt"], "response": pair["rejected"], "label": 1, "label_name": "harmful_rejected"},
        ))
    return examples


@torch.no_grad()
def extract(model, tokenizer, examples: list[dict], state_path: Path, checkpoint: dict, checkpoint_callback, every: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[torch.Tensor] = []
    labels: list[int] = []
    pair_ids: list[int] = []
    if state_path.exists():
        state = np.load(state_path)
        values = [torch.from_numpy(value) for value in state["activations"]]
        labels, pair_ids = state["labels"].tolist(), state["pair_ids"].tolist()
    start, special, started = len(labels), set(tokenizer.all_special_ids), time.monotonic()
    for number, example in enumerate(tqdm(examples[start:], initial=start, total=len(examples), desc="Extracting paired activations", unit="response"), start=start):
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=True, add_generation_prompt=True)
        response_ids = tokenizer(example["response"], add_special_tokens=False).input_ids
        valid = [position for position, token in enumerate(response_ids) if token not in special]
        if not valid:
            raise ValueError(f"response tokenized to special tokens only (pair {example['pair_id']})")
        output = model(input_ids=torch.tensor([prompt_ids + response_ids], device=model.device), output_hidden_states=True, use_cache=False)
        final_position = len(prompt_ids) + valid[-1]
        values.append(torch.stack([state[0, final_position].float().cpu() for state in output.hidden_states]))
        labels.append(example["label"]); pair_ids.append(example["pair_id"])
        complete = number + 1
        if complete % every == 0 or complete == len(examples):
            save_state(state_path, values, labels, pair_ids)
            elapsed = time.monotonic() - started
            metric = {"responses_extracted": complete, "total_responses": len(examples), "elapsed_sec": round(elapsed, 2), "responses_per_sec": round((complete - start) / max(elapsed, 1e-9), 3), "refusal_count": int(np.count_nonzero(np.array(labels) == 0)), "harmful_count": int(np.count_nonzero(np.array(labels) == 1))}
            checkpoint.update({"stage": "extracting", "next_response": complete, "latest_metric": metric})
            write_json(state_path.parent / "checkpoint.json", checkpoint); write_json(state_path.parent / "progress.json", {"stage": "extracting", "latest_metric": metric})
            print(json.dumps({"event": "progress", **metric}), flush=True); checkpoint_callback()
    return torch.stack(values).numpy(), np.array(labels), np.array(pair_ids)


@torch.no_grad()
def run(config_path: Path, run_mode: str = "fresh", checkpoint_callback=lambda: None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    checkpoint_path, state_path = output / "checkpoint.json", output / "activation_state.npz"
    if run_mode == "fresh":
        for path in (checkpoint_path, state_path, output / "results.json", output / "RESULTS.md"):
            path.unlink(missing_ok=True)
        checkpoint = {"status": "running", "stage": "starting", "next_response": 0, "config_fingerprint": fingerprint}
        (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    elif run_mode == "resume":
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("config_fingerprint") != fingerprint or checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError("resume requires matching config plus a running/stopped checkpoint")
        checkpoint["status"] = "running"
    else:
        raise ValueError("run_mode must be fresh or resume")
    write_json(checkpoint_path, checkpoint); checkpoint_callback()

    pairs = load_pairs(cfg); examples = paired_examples(pairs)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=os.environ.get("HF_TOKEN"), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], token=os.environ.get("HF_TOKEN"), torch_dtype=torch.float16).to("cuda").eval()
    activations, labels, pair_ids = extract(model, tokenizer, examples, state_path, checkpoint, checkpoint_callback, cfg["checkpoint_every"])
    train_pairs = set(range(cfg["train_pairs"]))
    train_idx = np.array([index for index, pair_id in enumerate(pair_ids) if pair_id in train_pairs])
    test_idx = np.array([index for index, pair_id in enumerate(pair_ids) if pair_id not in train_pairs])
    expected_train, expected_test = 2 * cfg["train_pairs"], 2 * (cfg["pair_count"] - cfg["train_pairs"])
    if len(train_idx) != expected_train or len(test_idx) != expected_test:
        raise RuntimeError("pair split was not preserved in activation state")
    scores = []
    for layer in tqdm(range(1, activations.shape[1]), desc="Training held-out layer probes", unit="layer"):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg["seed"]))
        probe.fit(activations[train_idx, layer], labels[train_idx])
        scores.append(float(probe.score(activations[test_idx, layer], labels[test_idx])))
    best_index, best_layer = int(np.argmax(scores)), int(np.argmax(scores)) + 1
    refusal_mean = activations[train_idx][labels[train_idx] == 0, best_layer].mean(0)
    harmful_mean = activations[train_idx][labels[train_idx] == 1, best_layer].mean(0)
    raw_direction = harmful_mean - refusal_mean
    raw_norm = float(np.linalg.norm(raw_direction)); direction = raw_direction / raw_norm
    np.save(output / "refusal_mean.npy", refusal_mean); np.save(output / "harmful_mean.npy", harmful_mean); np.save(output / "steering_vector.npy", direction)
    metric = {"best_layer": best_layer, "best_layer_accuracy": scores[best_index], "raw_direction_norm": raw_norm}
    print(json.dumps({"event": "probe_complete", **metric}), flush=True)
    result = {"config": cfg, "pairs": len(pairs), "responses": len(labels), "class_counts": {"refusal_chosen": int((labels == 0).sum()), "harmful_rejected": int((labels == 1).sum())}, "best_layer": best_layer, "best_layer_accuracy": scores[best_index], "layer_accuracy": scores, "direction": "normalized mean(rejected harmful) - mean(chosen refusal), training pairs only", "raw_direction_norm": raw_norm, "activation_position": "last non-special assistant response token", "vector_path": "steering_vector.npy", "refusal_mean_path": "refusal_mean.npy", "harmful_mean_path": "harmful_mean.npy"}
    write_json(output / "results.json", result)
    (output / "RESULTS.md").write_text("# LLM-LAT harmful/refusal steering vector\n\n" + f"- Pairs: {len(pairs)} (train {cfg['train_pairs']}, held out {len(pairs) - cfg['train_pairs']})\n- Best layer: {best_layer}\n- Held-out probe accuracy: {scores[best_index]:.4f}\n- Direction: rejected harmful minus chosen refusal\n")
    checkpoint.update({"status": "completed", "stage": "completed", "next_response": len(labels), "latest_metric": metric})
    write_json(checkpoint_path, checkpoint); write_json(output / "progress.json", {"stage": "completed", **metric}); checkpoint_callback()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    args = parser.parse_args(); print(json.dumps(run(args.config, args.run_mode), indent=2))
