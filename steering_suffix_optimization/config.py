from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED = {
    "model_id",
    "model_revision",
    "dataset_csv",
    "prompt_column",
    "vector_path",
    "vector_sha256",
    "hook_layer_index",
    "diagnostic_hidden_state_index",
    "alpha0",
    "hook_scale_mode",
    "train_indices",
    "heldout_indices",
    "continuations_per_prompt",
    "continuation_tokens",
    "temperature",
    "teacher_top_k",
    "alpha_multipliers",
    "suffix_lengths",
    "constraint_modes",
    "seeds",
    "steps",
    "learning_rate",
    "min_learning_rate",
    "diagnostic_every",
    "generation_every",
    "positive_instruction",
    "natural_language_suffix",
    "behavior_generation_tokens",
    "behavior_seed",
}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    missing = sorted(REQUIRED - config.keys())
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if len(config["train_indices"]) != 5 or len(config["heldout_indices"]) != 25:
        raise ValueError("protocol requires exactly 5 train and 25 held-out prompts")
    if set(config["train_indices"]) & set(config["heldout_indices"]):
        raise ValueError("train and held-out indices overlap")
    if config["continuations_per_prompt"] != 8 or config["continuation_tokens"] != 64:
        raise ValueError("protocol requires 8 continuations of 64 tokens")
    if config["hook_scale_mode"] not in {"plain", "activation_norm"}:
        raise ValueError("hook_scale_mode must be plain or activation_norm")
    if set(config["constraint_modes"]) != {"free", "mean_embedding_norm"}:
        raise ValueError("both free and mean_embedding_norm constraints are required")
    if config["run_mode"] not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    return config


def fingerprint(config: dict[str, Any]) -> str:
    # Execution intent may change from fresh to resume without changing science.
    canonical = {key: value for key, value in config.items() if key != "run_mode"}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
