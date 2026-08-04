"""Extract and validate a benign-conversation → jailbreak-conversation direction.

JailbreakBench's JBB-Behaviors benchmark supplies 100 benign and 100 harmful
behaviors, each with a user goal and a matching assistant target prefix. Each
conversation is encoded in Llama-2's chat format and measured at the final
non-special token of that target prefix (never at EOS/padding).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
def activations(model, tokenizer, examples, batch_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_layers, labels, indices = [], [], []
    special = set(tokenizer.all_special_ids)
    for goal, target, label, index in tqdm(examples, desc=batch_name, unit="conversation"):
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


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    token = __import__("os").environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=token, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], token=token, torch_dtype=torch.float16).to("cuda").eval()
    examples = load_examples(cfg)
    x, y, _indices = activations(model, tokenizer, examples, "Extracting benign and jailbreak activations")
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
    sweep = []
    for scale in cfg["sweep_scales"]:
        heldout_examples = [examples[i] for i in test_idx]
        shifted = steered_layer_activations(model, tokenizer, heldout_examples, best_layer, direction, float(scale))
        predicted = probe.predict(shifted)
        sweep.append({"scale": float(scale), "classifier_accuracy": float((predicted == y[test_idx]).mean()),
                      "jailbreak_prediction_rate": float(predicted.mean())})
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "benign_mean.npy", benign_mean)
    np.save(out / "jailbreak_mean.npy", jailbreak_mean)
    np.save(out / "steering_vector.npy", direction)
    result = {"config": cfg, "examples": len(examples), "class_counts": {"benign": int((y == 0).sum()), "jailbreak": int((y == 1).sum())}, "best_layer": best_layer,
              "layer_accuracy": scores, "best_layer_accuracy": scores[best_index], "sweep": sweep,
              "vector_path": "steering_vector.npy", "benign_mean_path": "benign_mean.npy", "jailbreak_mean_path": "jailbreak_mean.npy", "activation_position": "last non-special JBB target token",
              "direction": "mean(jailbreak) - mean(benign)"}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("# Jailbreak steering-vector results\n\n" + f"- Benign conversations: {int((y == 0).sum())}\n- Jailbreak conversations: {int((y == 1).sum())}\n- Best layer: {best_layer}\n- Probe accuracy: {scores[best_index]:.4f}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), indent=2))
