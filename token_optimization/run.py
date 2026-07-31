"""Single GCG behavior-token run — reproduce a steering vector via an input suffix.

Reuses the steering vector + classifier from a completed steering-vector run
(steering_vectors.run), optimizes a discrete suffix, evaluates it behaviorally,
and writes everything to its own labelled folder under
`outputs/token_optimization/<label>/` (artifact, transcripts, plot, results.md).
Resumable via per-step state.

Usage:
    python -m token_optimization.run configs/sadness.yaml
"""
from __future__ import annotations

import os
import sys

from steering_vectors.config import Config
from steering_vectors.data import load_conversations
from steering_vectors.evaluate import eval_prompts_from
from steering_vectors.model import SteeringModel
from token_optimization import checkpoint as ck
from token_optimization.gcg import GCG, GCGResult, evaluate_suffix


def run(config_path: str) -> str:
    cfg = Config.from_yaml(config_path)
    label = ck.run_label(cfg)
    log = ck.get_logger(cfg, label)
    log.info(f"=== GCG: {label} | target_scale={cfg.gcg_target_scale} "
             f"suffix_len={cfg.gcg_suffix_len} objective={cfg.gcg_objective}")

    sv, clf = ck.load_steering_vector(cfg), ck.load_classifier(cfg)
    if sv is None or clf is None:
        raise SystemExit("missing steering vector / classifier — run "
                         "`python -m steering_vectors.run <cfg>` first")

    convs = load_conversations(cfg)
    allp = eval_prompts_from(convs, cfg.gcg_n_prompts + cfg.eval_n_prompts)
    opt_prompts = allp[: cfg.gcg_n_prompts]
    eval_prompts = allp[cfg.gcg_n_prompts:][: cfg.eval_n_prompts]

    model = SteeringModel(cfg)
    gcg = GCG(model, cfg, sv)
    resume = ck.load_state(cfg, label)
    if resume and resume["step"] >= cfg.gcg_steps - 1:
        res = GCGResult(suffix_ids=resume["best_ids"],
                        suffix_text=model.tokenizer.decode(resume["best_ids"]),
                        loss=resume["best_loss"], loss_history=resume["history"],
                        target_scale=cfg.gcg_target_scale, layer=cfg.layer,
                        cos_to_v=float("nan"), prompts=opt_prompts)
        log.info("GCG already complete — loaded best suffix from checkpoint")
    else:
        res = gcg.optimize(opt_prompts, use_tqdm=True,
                           on_step=lambda st: ck.save_state(cfg, st, label), resume=resume)
    log.info(f"best loss={res.loss:.3f} proj={res.proj:.3f}/{cfg.gcg_target_scale} "
             f"cos_to_v={res.cos_to_v:.3f} suffix={res.suffix_text!r}")

    ev = evaluate_suffix(gcg, model, cfg, sv, clf, res.suffix_ids, eval_prompts)
    log.info(f"concept_rate clean={ev['clean']['concept_rate']:.2f} "
             f"steering={ev['steering']['concept_rate']:.2f} "
             f"suffix={ev['suffix']['concept_rate']:.2f}")

    art = {
        "concept": cfg.concept, "layer": cfg.layer, "target_scale": cfg.gcg_target_scale,
        "suffix_len": cfg.gcg_suffix_len, "suffix_ids": res.suffix_ids,
        "suffix_text": res.suffix_text, "loss": res.loss, "proj": res.proj,
        "cos_to_v": res.cos_to_v, "loss_history": res.loss_history,
        "gcg": {"steps": cfg.gcg_steps, "topk": cfg.gcg_topk,
                "search_batch": cfg.gcg_search_batch, "n_prompts": cfg.gcg_n_prompts,
                "objective": cfg.gcg_objective, "seed": cfg.gcg_seed},
        "eval": ev,
    }
    ck.save_artifact(cfg, art, label)
    ck.save_transcripts(cfg, ev.get("transcripts", []), label)

    from token_optimization.plot import plot_gcg
    png = plot_gcg(os.path.join(ck.run_dir(cfg, label), "artifact.json"))
    ck.write_results_md(cfg, art, label, plots=[png])
    log.info(f"saved -> {ck.run_dir(cfg, label)}")
    return ck.run_dir(cfg, label)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "configs/sadness.yaml")
