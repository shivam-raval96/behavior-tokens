"""Extract and validate a refusal → non-refusal steering direction for Llama-2.

The default data source is WildGuardMix, a pre-existing jailbreak/refusal
dataset of open-ended prompt--response examples with human-audited refusal
labels. Activations are read at the last non-special token of each response
(never at EOS/padding).
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_examples(cfg: dict) -> list[tuple[str, str, int, int]]:
    """Return open-ended (prompt, response, refusal_label, group) rows.

    WildGuardMix labels responses directly as ``refusal`` or ``compliance``.
    It includes both direct and adversarial jailbreak prompts; filtering is done
    from labels rather than a brittle string-prefix heuristic.
    """
    ds = load_dataset(cfg["dataset"], cfg.get("dataset_config"), split=cfg.get("split", "train"), token=__import__("os").environ.get("HF_TOKEN"))
    examples: list[tuple[str, str, int, int]] = []
    for row in ds:
        prompt, response = row.get("prompt"), row.get("response")
        label = row.get("response_refusal_label")
        if not isinstance(prompt, str) or not isinstance(response, str) or label not in {"refusal", "compliance"}:
            continue
        # Avoid empty / classifier-style responses: this is an open-ended
        # response representation experiment, not a label-token probe.
        if len(response.strip()) < cfg.get("min_response_characters", 32):
            continue
        examples.append((prompt, response, int(label == "compliance"), len(examples)))
        if len(examples) >= 2 * cfg["pairs"]:
            break
    counts = np.bincount([row[2] for row in examples], minlength=2)
    if min(counts) < cfg["pairs"]:
        raise RuntimeError(f"only found refusal/compliance counts {counts.tolist()}; need {cfg['pairs']} of each")
    # Balance labels so the probe cannot exploit class imbalance.
    selected = []
    for label in (0, 1):
        label_examples = [row for row in examples if row[2] == label]
        selected.extend(label_examples[:cfg["pairs"]])
    return selected


@torch.no_grad()
def activations(model, tokenizer, examples, batch_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_layers, labels, groups = [], [], []
    special = set(tokenizer.all_special_ids)
    for prompt, response_text, label, group in tqdm(examples, desc=batch_name, unit="response"):
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        response = tokenizer(response_text, add_special_tokens=False).input_ids
        valid = [i for i, token in enumerate(response) if token not in special]
        if not valid:
            continue
        ids = torch.tensor([prefix + response], device=model.device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        position = len(prefix) + valid[-1]
        all_layers.append(torch.stack([state[0, position].float().cpu() for state in out.hidden_states]))
        labels.append(label); groups.append(group)
    return torch.stack(all_layers).numpy(), np.array(labels), np.array(groups)


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
        for prompt, response_text, _label, _group in tqdm(examples, desc=f"Sweep scale {scale:g}", unit="response", leave=False):
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            response = tokenizer(response_text, add_special_tokens=False).input_ids
            valid = [i for i, token in enumerate(response) if token not in special]
            if not valid:
                continue
            ids = torch.tensor([prefix + response], device=model.device)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            values.append(out.hidden_states[layer][0, len(prefix) + valid[-1]].float().cpu())
        return torch.stack(values).numpy()
    finally:
        handle.remove()


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg["pairs"] < 300:
        raise ValueError("pairs must be at least 300")
    token = __import__("os").environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=token, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], token=token, torch_dtype=torch.float16).to("cuda").eval()
    examples = load_examples(cfg)
    x, y, groups = activations(model, tokenizer, examples, "Extracting contrastive activations")
    train_idx, test_idx = next(GroupShuffleSplit(test_size=cfg["test_fraction"], random_state=cfg["seed"]).split(x, y, groups))
    scores, probes = [], []
    for layer in tqdm(range(1, x.shape[1]), desc="Training layer probes", unit="layer"):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg["seed"]))
        probe.fit(x[train_idx, layer], y[train_idx])
        scores.append(float(probe.score(x[test_idx, layer], y[test_idx])))
        probes.append(probe)
    best_index = int(np.argmax(scores)); best_layer = best_index + 1; probe = probes[best_index]
    direction = x[train_idx][y[train_idx] == 1, best_layer].mean(0) - x[train_idx][y[train_idx] == 0, best_layer].mean(0)
    direction /= np.linalg.norm(direction)
    sweep = []
    for scale in cfg["sweep_scales"]:
        heldout_examples = [examples[i] for i in test_idx]
        shifted = steered_layer_activations(model, tokenizer, heldout_examples, best_layer, direction, float(scale))
        predicted = probe.predict(shifted)
        sweep.append({"scale": float(scale), "classifier_accuracy": float((predicted == y[test_idx]).mean()),
                      "non_refusal_prediction_rate": float(predicted.mean())})
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "steering_vector.npy", direction)
    result = {"config": cfg, "examples": len(examples), "class_counts": {"refusal": int((y == 0).sum()), "compliance": int((y == 1).sum())}, "best_layer": best_layer,
              "layer_accuracy": scores, "best_layer_accuracy": scores[best_index], "sweep": sweep,
              "vector_path": "steering_vector.npy", "activation_position": "last non-special response token"}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("# Refusal steering-vector results\n\n" + f"- Open-ended responses: {len(examples)}\n- Best layer: {best_layer}\n- Probe accuracy: {scores[best_index]:.4f}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), indent=2))
