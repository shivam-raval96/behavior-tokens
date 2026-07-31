"""Checkpointing: save each quantity as it is produced, resume where it stopped.

A run gets a directory `output_dir/<concept>_L<layer>_<pooling>/` holding:
  - run.log            append-only log (progress + errors)
  - activations.pt     A_pos, A_neg tensors (the expensive collection stage)
  - steering_vector.json
  - classifier.json
  - curve.jsonl        one line per steering-curve point (append as computed)
  - artifact.json      final combined artifact (written at the end)

Any stage whose file exists is loaded instead of recomputed; the curve resumes
from the scales already present in curve.jsonl.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from steering.classifier import ConceptClassifier
from steering.config import Config
from steering.evaluate import CurvePoint
from steering.steering import SteeringVector


def run_dir(cfg: Config) -> str:
    d = os.path.join(cfg.output_dir, f"{cfg.concept}_L{cfg.layer}_{cfg.pooling}")
    os.makedirs(d, exist_ok=True)
    return d


def _p(cfg: Config, name: str) -> str:
    return os.path.join(run_dir(cfg), name)


def get_logger(cfg: Config) -> logging.Logger:
    """Logger writing to both run.log (append) and stdout."""
    logger = logging.getLogger(f"steering.{cfg.concept}.L{cfg.layer}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(_p(cfg, "run.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---- activations -----------------------------------------------------------
def save_activations(cfg: Config, A_pos: torch.Tensor, A_neg: torch.Tensor):
    torch.save({"A_pos": A_pos, "A_neg": A_neg}, _p(cfg, "activations.pt"))


def load_activations(cfg: Config):
    path = _p(cfg, "activations.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=True)
    return d["A_pos"], d["A_neg"]


# ---- steering vector -------------------------------------------------------
def sv_to_dict(sv: SteeringVector, normalized: bool) -> dict:
    return {
        "concept": sv.concept, "layer": sv.layer, "pooling": sv.pooling,
        "raw_norm": sv.raw_norm, "normalized": normalized,
        "n_pos": sv.n_pos, "n_neg": sv.n_neg, "vector": sv.vector.tolist(),
    }


def sv_from_dict(d: dict) -> SteeringVector:
    return SteeringVector(
        vector=torch.tensor(d["vector"]), layer=d["layer"], pooling=d["pooling"],
        concept=d["concept"], raw_norm=d["raw_norm"], n_pos=d["n_pos"], n_neg=d["n_neg"],
    )


def save_steering_vector(cfg: Config, sv: SteeringVector):
    with open(_p(cfg, "steering_vector.json"), "w") as f:
        json.dump(sv_to_dict(sv, cfg.normalize), f, indent=2)


def load_steering_vector(cfg: Config):
    path = _p(cfg, "steering_vector.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sv_from_dict(json.load(f))


# ---- classifier ------------------------------------------------------------
def clf_to_dict(clf: ConceptClassifier) -> dict:
    return {
        "train_acc": clf.train_acc, "test_acc": clf.test_acc,
        "coef": clf.clf.coef_.ravel().tolist(),
        "intercept": clf.clf.intercept_.tolist(),
        "classes": clf.clf.classes_.tolist(),
    }


def clf_from_dict(d: dict, layer: int, pooling: str) -> ConceptClassifier:
    c = d["classifier"] if "classifier" in d else d
    lr = LogisticRegression()
    lr.coef_ = np.array([c["coef"]])
    lr.intercept_ = np.array(c["intercept"])
    lr.classes_ = np.array(c["classes"])
    lr.n_features_in_ = len(c["coef"])
    return ConceptClassifier(clf=lr, layer=layer, pooling=pooling,
                             train_acc=c["train_acc"], test_acc=c["test_acc"])


def save_classifier(cfg: Config, clf: ConceptClassifier):
    with open(_p(cfg, "classifier.json"), "w") as f:
        json.dump(clf_to_dict(clf), f, indent=2)


def load_classifier(cfg: Config):
    path = _p(cfg, "classifier.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return clf_from_dict(json.load(f), cfg.layer, cfg.pooling)


# ---- curve (append per point) ---------------------------------------------
def append_curve_point(cfg: Config, pt: CurvePoint):
    with open(_p(cfg, "curve.jsonl"), "a") as f:
        f.write(json.dumps({"scale": pt.scale, "concept_rate": pt.concept_rate,
                            "mean_prob": pt.mean_prob}) + "\n")


def load_curve_points(cfg: Config) -> list[dict]:
    path = _p(cfg, "curve.jsonl")
    if not os.path.exists(path):
        return []
    pts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                pts[round(p["scale"], 4)] = p          # last write wins
    return sorted(pts.values(), key=lambda p: p["scale"])


# ---- GCG suffix optimization ----------------------------------------------
def _gcg_tag(cfg: Config) -> str:
    # seed 0 keeps the original bare name; other seeds get a suffix so repeated
    # runs at different seeds are saved side-by-side, never overwritten.
    seed = f"_seed{cfg.gcg_seed}" if cfg.gcg_seed else ""
    return f"gcg_s{cfg.gcg_target_scale}_L{cfg.gcg_suffix_len}{seed}"


def gcg_state_path(cfg: Config) -> str:
    return _p(cfg, _gcg_tag(cfg) + "_state.json")


def save_gcg_state(cfg: Config, state: dict):
    with open(gcg_state_path(cfg), "w") as f:
        json.dump(state, f)


def load_gcg_state(cfg: Config):
    path = gcg_state_path(cfg)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_gcg_artifact(cfg: Config, art: dict) -> str:
    path = _p(cfg, _gcg_tag(cfg) + ".json")
    with open(path, "w") as f:
        json.dump(art, f, indent=2)
    return path


# ---- final artifact --------------------------------------------------------
def save_artifact(cfg: Config, sv: SteeringVector, clf: ConceptClassifier,
                  curve: list[dict]) -> str:
    path = _p(cfg, "artifact.json")
    with open(path, "w") as f:
        json.dump({
            "config": cfg.to_dict(),
            "steering_vector": sv_to_dict(sv, cfg.normalize),
            "classifier": clf_to_dict(clf),
            "steering_curve": curve,
        }, f, indent=2)
    return path
