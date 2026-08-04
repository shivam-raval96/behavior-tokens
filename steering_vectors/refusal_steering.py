"""Extract and validate a safe-response → harmful-response direction for Llama-2.

The default data source is PKU-SafeRLHF, a pre-existing safety dataset with a
single prompt, two open-ended assistant responses, and a safety label for each.
Only pairs with one safe and one harmful response are used. Activations are read
at the last non-special token of each response (never at EOS/padding).
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
    """Return matched (prompt, response, safety_label, pair_id) rows.

    PKU-SafeRLHF supplies two responses per prompt and boolean safety labels.
    Label 0 is the safe response and label 1 is the harmful/non-refusal response.
    """
    ds = load_dataset(cfg["dataset"], cfg.get("dataset_config"), split=cfg.get("split", "train"), token=__import__("os").environ.get("HF_TOKEN"))
    examples: list[tuple[str, str, int, int]] = []
    for pair_id, row in enumerate(ds):
        prompt = row.get("prompt")
        response_0, response_1 = row.get("response_0"), row.get("response_1")
        safe_0, safe_1 = row.get("is_response_0_safe"), row.get("is_response_1_safe")
        if not isinstance(prompt, str) or not isinstance(response_0, str) or not isinstance(response_1, str):
            continue
        if not isinstance(safe_0, bool) or not isinstance(safe_1, bool) or safe_0 == safe_1:
            continue
        if min(len(response_0.strip()), len(response_1.strip())) < cfg.get("min_response_characters", 32):
            continue
        safe_response, harmful_response = (response_0, response_1) if safe_0 else (response_1, response_0)
        examples.extend(((prompt, safe_response, 0, pair_id), (prompt, harmful_response, 1, pair_id)))
        if len(examples) >= 2 * cfg["pairs"]:
            break
    if len(examples) < 2 * cfg["pairs"]:
        raise RuntimeError(f"only found {len(examples) // 2} safe/harmful response pairs; need {cfg['pairs']}")
    return examples


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
    result = {"config": cfg, "examples": len(examples), "class_counts": {"safe": int((y == 0).sum()), "harmful": int((y == 1).sum())}, "best_layer": best_layer,
              "layer_accuracy": scores, "best_layer_accuracy": scores[best_index], "sweep": sweep,
              "vector_path": "steering_vector.npy", "activation_position": "last non-special response token"}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("# Harmful-response steering-vector results\n\n" + f"- Matched safe/harmful response pairs: {len(examples) // 2}\n- Best layer: {best_layer}\n- Probe accuracy: {scores[best_index]:.4f}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), indent=2))
