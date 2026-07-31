"""Seed-variance study: run GCG across many seeds for a few suffix lengths.

For each (suffix_len, seed) it records only the decoded suffix and the probe
signal (suffix concept-rate + mean P(concept)) — everything lands in ONE json,
written incrementally so a crash keeps completed runs. No per-run artifacts or
transcripts.

Usage:
    python -m steering.seed_sweep steering/configs/sadness.yaml \
        --lengths 1 8 --seeds 10 --out seed_sweep_L1_L8.json
"""
from __future__ import annotations

import json
import os
import sys

from steering import checkpoint as ck
from steering.config import Config
from steering.data import load_conversations
from steering.evaluate import _free_memory, _score_generations, eval_prompts_from
from steering.gcg import GCG
from steering.model import SteeringModel


def _parse(argv):
    cfg_path = argv[0]
    lengths, seeds, out = [1, 8], 10, "seed_sweep.json"
    i = 1
    while i < len(argv):
        if argv[i] == "--lengths":
            i += 1
            lengths = []
            while i < len(argv) and not argv[i].startswith("--"):
                lengths.append(int(argv[i])); i += 1
        elif argv[i] == "--seeds":
            seeds = int(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]; i += 2
        else:
            i += 1
    return cfg_path, lengths, seeds, out


def run(cfg_path, lengths, n_seeds, out_name):
    cfg = Config.from_yaml(cfg_path)
    log = ck.get_logger(cfg)
    sv, clf = ck.load_steering_vector(cfg), ck.load_classifier(cfg)
    if sv is None or clf is None:
        raise SystemExit("run steering.run_experiment first")

    convs = load_conversations(cfg)
    allp = eval_prompts_from(convs, cfg.gcg_n_prompts + cfg.eval_n_prompts)
    opt_prompts = allp[: cfg.gcg_n_prompts]
    eval_prompts = allp[cfg.gcg_n_prompts:][: cfg.eval_n_prompts]

    model = SteeringModel(cfg)

    # suffix-independent baselines, once
    clean_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    clean_rate, _ = _score_generations(model, clf, eval_prompts, clean_gen, cfg.batch_size)
    _free_memory()
    model.add_steering(sv.vector, sv.layer, cfg.gcg_target_scale)
    steer_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    model.clear_steering()
    steer_rate, _ = _score_generations(model, clf, eval_prompts, steer_gen, cfg.batch_size)
    _free_memory()

    out_path = os.path.join(ck.run_dir(cfg), out_name)
    result = {
        "concept": cfg.concept, "layer": cfg.layer,
        "target_scale": cfg.gcg_target_scale, "gcg_steps": cfg.gcg_steps,
        "eval_n_prompts": len(eval_prompts),
        "baseline": {"clean": clean_rate, "steering": steer_rate},
        "runs": [],
    }
    log.info(f"[seedsweep] lengths={lengths} seeds=0..{n_seeds - 1} "
             f"baselines clean={clean_rate:.2f} steering={steer_rate:.2f}")

    for L in lengths:
        for seed in range(n_seeds):
            cfg.gcg_suffix_len, cfg.gcg_seed = L, seed
            gcg = GCG(model, cfg, sv)
            res = gcg.optimize(opt_prompts, log_every=0)
            suf_gen = [gcg.generate_with_suffix(p, res.suffix_ids) for p in eval_prompts]
            rate, prob = _score_generations(model, clf, eval_prompts, suf_gen, cfg.batch_size)
            _free_memory()
            result["runs"].append({
                "suffix_len": L, "seed": seed, "suffix": res.suffix_text,
                "suffix_rate": rate, "mean_prob": prob, "proj": res.proj,
            })
            with open(out_path, "w") as f:                       # incremental save
                json.dump(result, f, indent=2)
            log.info(f"[seedsweep] L={L} seed={seed} rate={rate:.2f} "
                     f"prob={prob:.2f} suffix={res.suffix_text!r}")

    log.info(f"[seedsweep] saved -> {out_path} ({len(result['runs'])} runs)")
    return out_path


if __name__ == "__main__":
    cfg_path, lengths, seeds, out = _parse(sys.argv[1:])
    run(cfg_path, lengths, seeds, out)
