"""Sweep GCG suffix length and plot length vs. behavioral concept-rate.

Loads the model + steering vector + classifier ONCE, then runs GCG for each
requested suffix length (reusing everything). clean/steering baselines are
computed once (they don't depend on the suffix). Each length's result is saved
as a normal GCG artifact (gcg_s<scale>_L<len>.json) so plot_length_sweep can
combine it with any lengths already on disk.

Usage:
    python -m steering.sweep_len steering/configs/sadness.yaml 1 3 10 14 20 25
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
from steering.plot import plot_length_sweep


def run(config_path: str, lengths: list[int]):
    cfg = Config.from_yaml(config_path)
    log = ck.get_logger(cfg)
    sv = ck.load_steering_vector(cfg)
    clf = ck.load_classifier(cfg)
    if sv is None or clf is None:
        raise SystemExit("run steering.run_experiment first (need vector + classifier)")

    convs = load_conversations(cfg)
    allp = eval_prompts_from(convs, cfg.gcg_n_prompts + cfg.eval_n_prompts)
    opt_prompts = allp[: cfg.gcg_n_prompts]
    eval_prompts = allp[cfg.gcg_n_prompts:][: cfg.eval_n_prompts]

    model = SteeringModel(cfg)
    log.info(f"[sweep] lengths={lengths} | opt={len(opt_prompts)} eval={len(eval_prompts)}")

    # baselines (suffix-independent) computed once
    clean_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    clean_rate, _ = _score_generations(model, clf, eval_prompts, clean_gen, cfg.batch_size)
    _free_memory()
    model.add_steering(sv.vector, sv.layer, cfg.gcg_target_scale)
    steer_gen = model.generate_batch(eval_prompts, cfg.batch_size)
    model.clear_steering()
    steer_rate, _ = _score_generations(model, clf, eval_prompts, steer_gen, cfg.batch_size)
    _free_memory()
    log.info(f"[sweep] baselines clean={clean_rate:.2f} steering={steer_rate:.2f}")

    for L in lengths:
        cfg.gcg_suffix_len = L
        gcg = GCG(model, cfg, sv)
        res = gcg.optimize(opt_prompts, use_tqdm=True)
        suf_gen = [gcg.generate_with_suffix(p, res.suffix_ids) for p in eval_prompts]
        suf_rate, suf_prob = _score_generations(model, clf, eval_prompts, suf_gen, cfg.batch_size)
        _free_memory()
        log.info(f"[sweep] L={L:>2}  proj={res.proj:.2f}  loss={res.loss:.2f}  "
                 f"suffix_rate={suf_rate:.2f}  suffix={res.suffix_text!r}")

        # full side-by-side transcripts for this length
        tpath = os.path.join(ck.run_dir(cfg), ck._gcg_tag(cfg) + "_transcripts.jsonl")
        with open(tpath, "w") as f:
            for p, c, s, fg in zip(eval_prompts, clean_gen, steer_gen, suf_gen):
                f.write(json.dumps({"prompt": p, "clean": c, "steering": s, "suffix": fg}) + "\n")

        ck.save_gcg_artifact(cfg, {
            "concept": cfg.concept, "layer": cfg.layer,
            "target_scale": cfg.gcg_target_scale, "suffix_len": L,
            "suffix_ids": res.suffix_ids, "suffix_text": res.suffix_text,
            "loss": res.loss, "proj": res.proj, "cos_to_v": res.cos_to_v,
            "loss_history": res.loss_history,
            "eval": {
                "n_prompts": len(eval_prompts), "target_scale": cfg.gcg_target_scale,
                "clean": {"concept_rate": clean_rate, "mean_prob": 0.0},
                "steering": {"concept_rate": steer_rate, "mean_prob": 0.0},
                "suffix": {"concept_rate": suf_rate, "mean_prob": suf_prob},
                "sample_suffix_generation": suf_gen[0] if suf_gen else "",
            },
        })

    png = plot_length_sweep(ck.run_dir(cfg))
    log.info(f"[sweep] length-vs-rate plot -> {png}")
    return png


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    lengths = [int(x) for x in sys.argv[2:]] or [1, 3, 10, 14, 20, 25]
    run(cfg_path, lengths)
