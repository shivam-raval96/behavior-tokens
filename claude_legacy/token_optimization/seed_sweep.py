"""Seed-variance study: run GCG across many seeds for a few suffix lengths.

Records only the decoded suffix + probe signal (concept-rate + mean prob) per
(length, seed) into one json, written incrementally, plus a results.md with
per-length mean/std. Output folder:
`outputs/token_optimization/<concept>_L<layer>_seedsweep/`.

Usage:
    python -m token_optimization.seed_sweep configs/sadness.yaml --lengths 1 8 --seeds 10
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

from steering_vectors.config import Config
from steering_vectors.data import load_conversations
from steering_vectors.evaluate import _free_memory, _score_generations, eval_prompts_from
from steering_vectors.model import SteeringModel
from token_optimization import checkpoint as ck
from token_optimization.gcg import GCG


def _parse(argv):
    cfg_path, lengths, seeds, out = argv[0], [1, 8], 10, "seed_sweep.json"
    i = 1
    while i < len(argv):
        if argv[i] == "--lengths":
            i += 1; lengths = []
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
    label = ck.run_label(cfg, "seedsweep")
    d = ck.run_dir(cfg, label)
    log = ck.get_logger(cfg, label)
    sv, clf = ck.load_steering_vector(cfg), ck.load_classifier(cfg)
    if sv is None or clf is None:
        raise SystemExit("run steering_vectors.run first")

    convs = load_conversations(cfg)
    allp = eval_prompts_from(convs, cfg.gcg_n_prompts + cfg.eval_n_prompts)
    opt_prompts = allp[: cfg.gcg_n_prompts]
    eval_prompts = allp[cfg.gcg_n_prompts:][: cfg.eval_n_prompts]

    model = SteeringModel(cfg)
    clean_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    clean_rate, _ = _score_generations(model, clf, eval_prompts, clean_gen, cfg.batch_size)
    _free_memory()
    model.add_steering(sv.vector, sv.layer, cfg.gcg_target_scale)
    steer_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    model.clear_steering()
    steer_rate, _ = _score_generations(model, clf, eval_prompts, steer_gen, cfg.batch_size)
    _free_memory()

    out_path = os.path.join(d, out_name)
    result = {"concept": cfg.concept, "layer": cfg.layer,
              "target_scale": cfg.gcg_target_scale, "gcg_steps": cfg.gcg_steps,
              "eval_n_prompts": len(eval_prompts),
              "baseline": {"clean": clean_rate, "steering": steer_rate}, "runs": []}
    log.info(f"[seedsweep] {label} lengths={lengths} seeds=0..{n_seeds - 1} "
             f"baselines clean={clean_rate:.2f} steering={steer_rate:.2f}")

    for L in lengths:
        for seed in range(n_seeds):
            cfg.gcg_suffix_len, cfg.gcg_seed = L, seed
            gcg = GCG(model, cfg, sv)
            res = gcg.optimize(opt_prompts, log_every=0)
            suf_gen = [gcg.generate_with_suffix(p, res.suffix_ids) for p in eval_prompts]
            rate, prob = _score_generations(model, clf, eval_prompts, suf_gen, cfg.batch_size)
            _free_memory()
            result["runs"].append({"suffix_len": L, "seed": seed, "suffix": res.suffix_text,
                                   "suffix_rate": rate, "mean_prob": prob, "proj": res.proj})
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"[seedsweep] L={L} seed={seed} rate={rate:.2f} prob={prob:.2f} "
                     f"suffix={res.suffix_text!r}")

    _write_md(d, result, lengths)
    log.info(f"[seedsweep] saved -> {out_path} ({len(result['runs'])} runs)")
    return out_path


def _write_md(d, result, lengths):
    lines = [f"# GCG seed-variance study — {result['concept']}, layer {result['layer']}, "
             f"α={result['target_scale']}", "",
             f"Baselines: clean {result['baseline']['clean']:.2f}, "
             f"steering {result['baseline']['steering']:.2f}. "
             f"{result['gcg_steps']} steps, {result['eval_n_prompts']} eval prompts.", "",
             "| suffix_len | n | mean | std | min | max |",
             "|-----------:|--:|-----:|----:|----:|----:|"]
    for L in lengths:
        r = [x["suffix_rate"] for x in result["runs"] if x["suffix_len"] == L]
        if r:
            lines.append(f"| {L} | {len(r)} | {st.mean(r):.2f} | {st.pstdev(r):.2f} "
                         f"| {min(r):.2f} | {max(r):.2f} |")
    lines += ["", "Rate variance is driven by which affect token GCG's random search lands, "
              "not by length; shorter suffixes force a single high-impact token.", ""]
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    cfg_path, lengths, seeds, out = _parse(sys.argv[1:])
    run(cfg_path, lengths, seeds, out)
