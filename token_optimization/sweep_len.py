"""Sweep GCG suffix length; plot length vs. behavioral concept-rate.

Loads model + steering vector + classifier ONCE, runs GCG per length, and writes
one experiment folder `outputs/token_optimization/<concept>_L<layer>_lensweep/`
with per-length artifacts (gcg_len<L>.json), the length_sweep.png, and results.md.

Usage:
    python -m token_optimization.sweep_len configs/sadness.yaml 1 8 16 32
"""
from __future__ import annotations

import json
import os
import sys

from steering_vectors.config import Config
from steering_vectors.data import load_conversations
from steering_vectors.evaluate import _free_memory, _score_generations, eval_prompts_from
from steering_vectors.model import SteeringModel
from token_optimization import checkpoint as ck
from token_optimization.gcg import GCG
from token_optimization.plot import plot_length_sweep


def run(config_path: str, lengths: list[int]):
    cfg = Config.from_yaml(config_path)
    label = ck.run_label(cfg, "lensweep")
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
    log.info(f"[lensweep] {label} lengths={lengths}")

    clean_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    clean_rate, _ = _score_generations(model, clf, eval_prompts, clean_gen, cfg.batch_size)
    _free_memory()
    model.add_steering(sv.vector, sv.layer, cfg.gcg_target_scale)
    steer_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    model.clear_steering()
    steer_rate, _ = _score_generations(model, clf, eval_prompts, steer_gen, cfg.batch_size)
    _free_memory()
    log.info(f"[lensweep] baselines clean={clean_rate:.2f} steering={steer_rate:.2f}")

    rows = []
    for L in lengths:
        cfg.gcg_suffix_len = L
        gcg = GCG(model, cfg, sv)
        res = gcg.optimize(opt_prompts, log_every=0)
        suf_gen = [gcg.generate_with_suffix(p, res.suffix_ids) for p in eval_prompts]
        rate, prob = _score_generations(model, clf, eval_prompts, suf_gen, cfg.batch_size)
        _free_memory()
        rows.append((L, rate, res.proj))
        log.info(f"[lensweep] L={L:>2} proj={res.proj:.2f} suffix_rate={rate:.2f} "
                 f"suffix={res.suffix_text!r}")
        art = {
            "concept": cfg.concept, "layer": cfg.layer, "suffix_len": L,
            "target_scale": cfg.gcg_target_scale, "proj": res.proj, "loss": res.loss,
            "suffix_text": res.suffix_text, "suffix_ids": res.suffix_ids,
            "eval": {"n_prompts": len(eval_prompts), "target_scale": cfg.gcg_target_scale,
                     "clean": {"concept_rate": clean_rate}, "steering": {"concept_rate": steer_rate},
                     "suffix": {"concept_rate": rate, "mean_prob": prob}},
        }
        with open(os.path.join(d, f"gcg_len{L}.json"), "w") as f:
            json.dump(art, f, indent=2)

    png = plot_length_sweep(d)
    _write_md(d, cfg, rows, clean_rate, steer_rate, png)
    log.info(f"[lensweep] saved -> {d}")
    return d


def _write_md(d, cfg, rows, clean, steer, png):
    rows = sorted(rows)
    tbl = "\n".join(f"| {L} | {proj:.2f} | {rate:.2f} |" for L, rate, proj in rows)
    peak = max(rows, key=lambda r: r[1])
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write(
            f"# GCG suffix-length sweep — {cfg.concept}, layer {cfg.layer}, α={cfg.gcg_target_scale}\n\n"
            f"Baselines: clean **{clean:.2f}**, activation steering **{steer:.2f}**.\n\n"
            f"| suffix_len | proj/{cfg.gcg_target_scale} | {cfg.concept} rate |\n"
            f"|-----------:|------:|----:|\n{tbl}\n\n"
            f"Peak: **{peak[0]} tokens → {peak[1]:.2f}**.\n\n"
            f"![length_sweep]({os.path.basename(png)})\n"
        )


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    lengths = [int(x) for x in sys.argv[2:]] or [1, 8, 16, 32]
    run(cfg_path, lengths)
