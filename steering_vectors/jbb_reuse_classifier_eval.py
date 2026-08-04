"""Plot saved-classifier probability after steered text generation.

This evaluation reuses the prior JBB steering vector, selected layer, and
source activation state. It does not collect a new JBB activation dataset or
construct a new direction. Per generated response it retains only the selected
classifier probability, never a hidden-state array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def harmful_examples(cfg: dict) -> list[dict]:
    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split="harmful")
    rows = [{"dataset_index": row["Index"], "goal": row["Goal"]} for row in ds]
    rng = np.random.default_rng(cfg["seed"])
    return [rows[i] for i in sorted(rng.choice(len(rows), size=cfg["evaluation_examples"], replace=False))]


def source_probe(source: Path) -> tuple[int, np.ndarray, object]:
    result = json.loads((source / "results.json").read_text())
    state = np.load(source / "activation_state.npz")
    x, y = state["activations"], state["labels"]
    layer = int(result["best_layer"])
    _, test_idx = next(StratifiedShuffleSplit(test_size=result["config"]["test_fraction"], random_state=result["config"]["seed"]).split(x, y))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=result["config"]["seed"]))
    probe.fit(x[train_idx, layer], y[train_idx])
    return layer, np.load(source / "steering_vector.npy"), probe


def run(config_path: Path, run_mode: str = "fresh", checkpoint_callback=lambda: None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    checkpoint_path, rows_path = output / "checkpoint.json", output / "classifier_probabilities.jsonl"
    if run_mode == "fresh":
        for path in (checkpoint_path, rows_path, output / "results.json", output / "classifier_probability_vs_strength.png"):
            path.unlink(missing_ok=True)
        checkpoint = {"status": "running", "stage": "starting", "config_fingerprint": fingerprint, "completed": 0}
        (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    elif run_mode == "resume":
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("config_fingerprint") != fingerprint or checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError("resume requires a matching stopped/running checkpoint")
        checkpoint["status"] = "running"
    else:
        raise ValueError("run_mode must be fresh or resume")
    write_json(checkpoint_path, checkpoint); checkpoint_callback()

    source = Path(cfg["source_run_dir"])
    layer, direction, probe = source_probe(source)
    examples = harmful_examples(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=os.environ.get("HF_TOKEN"), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], token=os.environ.get("HF_TOKEN"), torch_dtype=torch.float16).to("cuda").eval()
    existing = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line); existing.add((float(row["scale"]), int(row["dataset_index"])))
    total = len(cfg["scales"]) * len(examples)
    started = time.monotonic()
    with rows_path.open("a") as stream:
        for scale in cfg["scales"]:
            vector = torch.tensor(direction, device=model.device, dtype=model.dtype) * float(scale)

            def add_vector(_module, _inputs, output_value):
                hidden, *rest = output_value if isinstance(output_value, tuple) else (output_value,)
                hidden = hidden + vector
                return (hidden, *rest) if isinstance(output_value, tuple) else hidden

            hook = model.model.layers[layer - 1].register_forward_hook(add_vector)
            try:
                for example in tqdm(examples, desc=f"Evaluating scale {scale:g}", unit="prompt", leave=False):
                    key = (float(scale), example["dataset_index"])
                    if key in existing:
                        continue
                    input_ids = tokenizer.apply_chat_template([{"role": "user", "content": example["goal"]}], tokenize=True,
                                                               add_generation_prompt=True, return_tensors="pt").to(model.device)
                    prompt_seed = cfg["generation_seed"] + int((float(scale) + 20) * 100_000) + example["dataset_index"]
                    torch.manual_seed(prompt_seed); torch.cuda.manual_seed_all(prompt_seed)
                    generated = model.generate(input_ids, do_sample=True, temperature=cfg["temperature"],
                                               max_new_tokens=cfg["max_new_tokens"], pad_token_id=tokenizer.eos_token_id, use_cache=True)
                    measured = model(input_ids=generated, output_hidden_states=True, use_cache=False)
                    probability = float(probe.predict_proba(measured.hidden_states[layer][0, -1].float().cpu().numpy().reshape(1, -1))[0, 1])
                    row = {"scale": float(scale), **example, "layer": layer, "temperature": cfg["temperature"],
                           "max_new_tokens": cfg["max_new_tokens"], "classifier_probability": probability,
                           "response": tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)}
                    stream.write(json.dumps(row) + "\n"); stream.flush(); os.fsync(stream.fileno())
                    existing.add(key)
                    if len(existing) % cfg["checkpoint_every"] == 0:
                        metric = {"completed": len(existing), "total": total, "scale": float(scale),
                                  "elapsed_sec": round(time.monotonic() - started, 2), "last_probability": probability}
                        checkpoint.update({"stage": "generating_and_scoring", "completed": len(existing), "latest_metric": metric})
                        write_json(checkpoint_path, checkpoint); write_json(output / "progress.json", {"stage": "generating_and_scoring", "latest_metric": metric})
                        print(json.dumps({"event": "progress", "stage": "generating_and_scoring", **metric}), flush=True)
                        checkpoint_callback()
            finally:
                hook.remove()

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    curve = []
    for scale in cfg["scales"]:
        values = [row["classifier_probability"] for row in rows if float(row["scale"]) == float(scale)]
        curve.append({"scale": float(scale), "n": len(values), "mean_probability": float(np.mean(values)),
                      "std_probability": float(np.std(values, ddof=1))})
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar([row["scale"] for row in curve], [row["mean_probability"] for row in curve],
                yerr=[row["std_probability"] for row in curve], marker="o", capsize=3)
    ax.set(xlabel="Steering strength", ylabel="Mean jailbreak-class probability", ylim=(0, 1),
           title="Classifier probability after 64-token generation")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(output / "classifier_probability_vs_strength.png", dpi=160); plt.close(fig)
    result = {"source_run_dir": str(source), "source_best_layer": layer, "direction": "reused source steering_vector.npy",
              "activation_dataset_extraction": False, "vector_recomputed": False,
              "post_generation_measurement": "selected-layer final-token state scored then discarded", "curve": curve,
              "probabilities_path": "classifier_probabilities.jsonl", "plot_path": "classifier_probability_vs_strength.png", "config": cfg}
    write_json(output / "results.json", result)
    checkpoint.update({"status": "completed", "stage": "completed", "completed": len(rows), "latest_metric": curve[-1]})
    write_json(checkpoint_path, checkpoint); write_json(output / "progress.json", {"stage": "completed", "latest_metric": curve[-1]}); checkpoint_callback()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    args = parser.parse_args(); print(json.dumps(run(args.config, args.run_mode), indent=2))
