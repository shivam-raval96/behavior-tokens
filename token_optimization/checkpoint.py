"""Checkpointing for token-optimization (GCG) runs.

Each run gets its own labelled folder under
`output_dir/token_optimization/<label>/` holding its artifact, transcripts,
plots, resume state, and a concise `results.md`. Steering vectors + classifiers
are loaded (never written) from the steering-vector outputs via
`steering_vectors.checkpoint`.
"""
from __future__ import annotations

import json
import logging
import os

from steering_vectors.config import Config
# re-export the (de)serializers + loaders so callers use one checkpoint module
from steering_vectors.checkpoint import (  # noqa: F401
    sv_to_dict, sv_from_dict, clf_to_dict, clf_from_dict,
    load_steering_vector, load_classifier,
)


def run_label(cfg: Config, kind: str = "gcg") -> str:
    """Human-readable, unique-per-run folder name."""
    seed = f"_seed{cfg.gcg_seed}" if cfg.gcg_seed else ""
    obj = "" if cfg.gcg_objective == "project" else f"_{cfg.gcg_objective}"
    if kind == "gcg":
        return (f"{cfg.concept}_L{cfg.layer}_a{cfg.gcg_target_scale}"
                f"_len{cfg.gcg_suffix_len}{obj}{seed}")
    return f"{cfg.concept}_L{cfg.layer}_{kind}"          # sweeps: one folder per experiment


def run_dir(cfg: Config, label: str | None = None) -> str:
    d = os.path.join(cfg.output_dir, "token_optimization", label or run_label(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def get_logger(cfg: Config, label: str | None = None) -> logging.Logger:
    d = run_dir(cfg, label)
    logger = logging.getLogger(f"gcg.{os.path.basename(d)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(os.path.join(d, "run.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---- resume state ----------------------------------------------------------
def state_path(cfg: Config, label: str | None = None) -> str:
    return os.path.join(run_dir(cfg, label), "gcg_state.json")


def save_state(cfg: Config, state: dict, label: str | None = None):
    with open(state_path(cfg, label), "w") as f:
        json.dump(state, f)


def load_state(cfg: Config, label: str | None = None):
    path = state_path(cfg, label)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---- artifacts -------------------------------------------------------------
def save_artifact(cfg: Config, art: dict, label: str | None = None, name="artifact.json") -> str:
    path = os.path.join(run_dir(cfg, label), name)
    with open(path, "w") as f:
        json.dump(art, f, indent=2)
    return path


def save_transcripts(cfg: Config, rows: list[dict], label: str | None = None,
                     name="transcripts.jsonl") -> str:
    path = os.path.join(run_dir(cfg, label), name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def write_results_md(cfg: Config, art: dict, label: str | None = None,
                     plots: list[str] | None = None) -> str:
    """Concise per-run results.md: config + numbers + plot links."""
    d = run_dir(cfg, label)
    ev = art.get("eval", {})
    lines = [
        f"# GCG run — {art.get('concept', cfg.concept)}, layer {art.get('layer', cfg.layer)}",
        "",
        f"- **suffix length:** {art.get('suffix_len', cfg.gcg_suffix_len)} tokens",
        f"- **target scale (α):** {art.get('target_scale', cfg.gcg_target_scale)}",
        f"- **objective:** {cfg.gcg_objective} | steps {cfg.gcg_steps} | "
        f"seed {cfg.gcg_seed} | opt prompts {cfg.gcg_n_prompts}",
        f"- **best loss:** {art.get('loss', float('nan')):.3f} | "
        f"**proj:** {art.get('proj', float('nan')):.3f} / {art.get('target_scale', cfg.gcg_target_scale)} "
        f"| cos_to_v {art.get('cos_to_v', float('nan')):.3f}",
        "",
        "## Suffix",
        "",
        f"```\n{art.get('suffix_text', '')}\n```",
        "",
    ]
    if ev:
        lines += [
            "## Behavioral effect (concept rate on held-out prompts)",
            "",
            "| condition | rate | mean prob |",
            "|-----------|-----:|----------:|",
            f"| clean | {ev['clean']['concept_rate']:.2f} | {ev['clean'].get('mean_prob', 0):.2f} |",
            f"| activation steering α={ev['target_scale']} | {ev['steering']['concept_rate']:.2f} | "
            f"{ev['steering'].get('mean_prob', 0):.2f} |",
            f"| **GCG suffix (input only)** | **{ev['suffix']['concept_rate']:.2f}** | "
            f"{ev['suffix'].get('mean_prob', 0):.2f} |",
            "",
            f"Interpretation: the input suffix reproduces "
            f"{ev['suffix']['concept_rate'] / max(ev['steering']['concept_rate'], 1e-9):.0%} of the "
            f"activation-steering effect through the input channel alone "
            f"({ev['n_prompts']} prompts).",
            "",
        ]
    if plots:
        lines += ["## Plots", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in plots] + [""]
    path = os.path.join(d, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
