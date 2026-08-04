"""Extract and validate a benign-conversation → jailbreak-conversation direction.

JailbreakBench's JBB-Behaviors benchmark supplies 100 benign and 100 harmful
behaviors, each with a user goal and a matching assistant target prefix. Each
conversation is encoded in Llama-2's chat format and measured at the final
non-special token of that target prefix (never at EOS/padding).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_yaml(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=True))
    tmp.replace(path)


def save_activation_state(path: Path, activations: list[torch.Tensor], labels: list[int], indices: list[int]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, activations=torch.stack(activations).numpy(), labels=np.array(labels), indices=np.array(indices))
    tmp.replace(path)

def load_examples(cfg: dict) -> list[tuple[str, str, int, int]]:
    """Return (goal, target_prefix, label, dataset_index) examples from JBB.

    Label 0 is a benign conversation and label 1 is a harmful/jailbreak
    conversation. Targets are response prefixes published by JailbreakBench,
    not model generations produced by this experiment.
    """
    examples: list[tuple[str, str, int, int]] = []
    for label, split in ((0, "benign"), (1, "harmful")):
        ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=split)
        for row in ds:
            goal, target, index = row.get("Goal"), row.get("Target"), row.get("Index")
            if not isinstance(goal, str) or not isinstance(target, str) or not isinstance(index, int):
                continue
            examples.append((goal, target, label, index))
    expected = cfg["examples_per_class"]
    counts = np.bincount([row[2] for row in examples], minlength=2)
    if counts.tolist() != [expected, expected]:
        raise RuntimeError(f"expected {expected} benign and {expected} harmful JBB examples; found {counts.tolist()}")
    return examples


@torch.no_grad()
def activations(model, tokenizer, examples, batch_name: str, state_path: Path | None = None,
                checkpoint: dict | None = None, checkpoint_callback=lambda: None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_layers, labels, indices = [], [], []
    start = 0
    if state_path and state_path.exists():
        saved = np.load(state_path)
        all_layers = [torch.from_numpy(a) for a in saved["activations"]]
        labels, indices = saved["labels"].tolist(), saved["indices"].tolist()
        start = len(labels)
    special = set(tokenizer.all_special_ids)
    for example_number, (goal, target, label, index) in enumerate(tqdm(examples[start:], desc=batch_name, unit="conversation"), start=start):
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": goal}], tokenize=True, add_generation_prompt=True)
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        valid = [i for i, token in enumerate(target_ids) if token not in special]
        if not valid:
            continue
        ids = torch.tensor([prefix + target_ids], device=model.device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        position = len(prefix) + valid[-1]
        all_layers.append(torch.stack([state[0, position].float().cpu() for state in out.hidden_states]))
        labels.append(label); indices.append(index)
        if state_path and (example_number + 1) % 10 == 0:
            save_activation_state(state_path, all_layers, labels, indices)
            if checkpoint is not None:
                checkpoint.update({"stage": "extracting", "next_example": example_number + 1,
                                   "latest_metric": {"examples_extracted": example_number + 1}})
                write_json(state_path.parent / "checkpoint.json", checkpoint)
                write_json(state_path.parent / "progress.json", {"stage": "extracting", "examples_extracted": example_number + 1})
                print(json.dumps({"event": "progress", "stage": "extracting", "examples_extracted": example_number + 1}), flush=True)
            checkpoint_callback()
    if state_path:
        save_activation_state(state_path, all_layers, labels, indices)
    return torch.stack(all_layers).numpy(), np.array(labels), np.array(indices)


@torch.no_grad()
def steered_layer_activations(model, tokenizer, examples, layer: int, direction: np.ndarray, scale: float) -> np.ndarray:
    """Re-run held-out examples with an actual residual-stream intervention."""
    vector = torch.tensor(direction, device=model.device, dtype=model.dtype) * scale

    def add_vector(_module, _inputs, output):
        hidden, *rest = output if isinstance(output, tuple) else (output,)
        # Apply at every position: this is the standard constant activation-addition
        # intervention. We read the final response position below.
        hidden = hidden + vector
        return (hidden, *rest) if isinstance(output, tuple) else hidden

    # output_hidden_states[0] is the embeddings; layer N is transformer block N-1.
    handle = model.model.layers[layer - 1].register_forward_hook(add_vector)
    try:
        values = []
        special = set(tokenizer.all_special_ids)
        for goal, target, _label, _index in tqdm(examples, desc=f"Sweep scale {scale:g}", unit="conversation", leave=False):
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": goal}], tokenize=True, add_generation_prompt=True)
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            valid = [i for i, token in enumerate(target_ids) if token not in special]
            if not valid:
                continue
            ids = torch.tensor([prefix + target_ids], device=model.device)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            values.append(out.hidden_states[layer][0, len(prefix) + valid[-1]].float().cpu())
        return torch.stack(values).numpy()
    finally:
        handle.remove()


def run(config_path: Path, run_mode: str = "fresh", checkpoint_callback=lambda: None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    checkpoint_path, state_path = out / "checkpoint.json", out / "activation_state.npz"
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    if run_mode == "fresh":
        for path in (checkpoint_path, state_path, out / "results.json", out / "RESULTS.md"):
            path.unlink(missing_ok=True)
        checkpoint = {"status": "running", "stage": "starting", "next_example": 0,
                      "completed_scales": [], "config_fingerprint": fingerprint}
        write_yaml(out / "config.yaml", cfg)
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume: {checkpoint_path} is missing")
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError("cannot resume: config fingerprint differs")
        if checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError(f"cannot resume checkpoint with status {checkpoint.get('status')!r}")
        checkpoint["status"] = "running"
    write_json(checkpoint_path, checkpoint); checkpoint_callback()
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=token, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], token=token, torch_dtype=torch.float16).to("cuda").eval()
    examples = load_examples(cfg)
    x, y, _indices = activations(model, tokenizer, examples, "Extracting benign and jailbreak activations", state_path, checkpoint, checkpoint_callback)
    checkpoint.update({"stage": "probing", "next_example": len(examples),
                       "latest_metric": {"examples_extracted": len(examples)}})
    write_json(checkpoint_path, checkpoint); write_json(out / "progress.json", {"stage": "probing", "examples_extracted": len(examples)}); checkpoint_callback()
    train_idx, test_idx = next(StratifiedShuffleSplit(test_size=cfg["test_fraction"], random_state=cfg["seed"]).split(x, y))
    scores, probes = [], []
    for layer in tqdm(range(1, x.shape[1]), desc="Training layer probes", unit="layer"):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg["seed"]))
        probe.fit(x[train_idx, layer], y[train_idx])
        scores.append(float(probe.score(x[test_idx, layer], y[test_idx])))
        probes.append(probe)
    best_index = int(np.argmax(scores)); best_layer = best_index + 1; probe = probes[best_index]
    benign_mean = x[train_idx][y[train_idx] == 0, best_layer].mean(0)
    jailbreak_mean = x[train_idx][y[train_idx] == 1, best_layer].mean(0)
    direction = jailbreak_mean - benign_mean
    direction /= np.linalg.norm(direction)
    sweep = checkpoint.get("sweep", [])
    completed_scales = {row["scale"] for row in sweep}
    for scale in cfg["sweep_scales"]:
        if float(scale) in completed_scales:
            continue
        heldout_examples = [examples[i] for i in test_idx]
        shifted = steered_layer_activations(model, tokenizer, heldout_examples, best_layer, direction, float(scale))
        predicted = probe.predict(shifted)
        sweep.append({"scale": float(scale), "classifier_accuracy": float((predicted == y[test_idx]).mean()),
                      "jailbreak_prediction_rate": float(predicted.mean())})
        checkpoint.update({"stage": "sweeping", "sweep": sweep, "completed_scales": [row["scale"] for row in sweep],
                           "latest_metric": sweep[-1]})
        write_json(checkpoint_path, checkpoint); write_json(out / "progress.json", {"stage": "sweeping", "latest_metric": sweep[-1]})
        print(json.dumps({"event": "progress", "stage": "sweeping", **sweep[-1]}), flush=True)
        checkpoint_callback()
    np.save(out / "benign_mean.npy", benign_mean)
    np.save(out / "jailbreak_mean.npy", jailbreak_mean)
    np.save(out / "steering_vector.npy", direction)
    result = {"config": cfg, "examples": len(examples), "class_counts": {"benign": int((y == 0).sum()), "jailbreak": int((y == 1).sum())}, "best_layer": best_layer,
              "layer_accuracy": scores, "best_layer_accuracy": scores[best_index], "sweep": sweep,
              "vector_path": "steering_vector.npy", "benign_mean_path": "benign_mean.npy", "jailbreak_mean_path": "jailbreak_mean.npy", "activation_position": "last non-special JBB target token",
              "direction": "mean(jailbreak) - mean(benign)"}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("# Jailbreak steering-vector results\n\n" + f"- Benign conversations: {int((y == 0).sum())}\n- Jailbreak conversations: {int((y == 1).sum())}\n- Best layer: {best_layer}\n- Probe accuracy: {scores[best_index]:.4f}\n")
    checkpoint.update({"status": "completed", "stage": "completed", "latest_metric": {"best_layer_accuracy": scores[best_index]}})
    write_json(checkpoint_path, checkpoint); write_json(out / "progress.json", {"stage": "completed", "best_layer": best_layer, "best_layer_accuracy": scores[best_index]}); checkpoint_callback()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_mode), indent=2))
