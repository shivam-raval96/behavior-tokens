"""Find a discrete suffix that reproduces the steering vector — end to end.

Reuses the steering vector + concept classifier from a completed steering run
(steering.run_experiment), then runs GCG (steering.gcg) to optimize a token
suffix whose activations match `gcg_target_scale * v` at the steering layer.
Finally evaluates the suffix behaviorally (clean vs activation-steering vs
suffix) and saves the result + plots.

Checkpointed: the GCG loop saves its state every step, so a re-run resumes from
the last completed step. The steering vector / classifier are loaded from the
steering run's checkpoint dir (never recomputed here).

Usage:
    python -m steering.run_gcg steering/configs/rude.yaml
"""
from __future__ import annotations

import sys

from steering import checkpoint as ck
from steering.config import Config
from steering.data import load_conversations
from steering.evaluate import eval_prompts_from
from steering.gcg import GCG, GCGResult, evaluate_suffix
from steering.model import SteeringModel


def run(config_path: str) -> str:
    cfg = Config.from_yaml(config_path)
    log = ck.get_logger(cfg)
    log.info(f"=== GCG: concept={cfg.concept} layer={cfg.layer} "
             f"target_scale={cfg.gcg_target_scale} suffix_len={cfg.gcg_suffix_len}")

    # --- prerequisites from the steering run (reused, never recomputed) ---
    sv = ck.load_steering_vector(cfg)
    clf = ck.load_classifier(cfg)
    if sv is None or clf is None:
        raise SystemExit("missing steering vector / classifier — run "
                         "`python -m steering.run_experiment <cfg>` first")
    log.info(f"loaded steering vector + classifier (test_acc={clf.test_acc:.3f})")

    convs = load_conversations(cfg)
    all_prompts = eval_prompts_from(convs, cfg.gcg_n_prompts + cfg.eval_n_prompts)
    opt_prompts = all_prompts[: cfg.gcg_n_prompts]                 # optimize on these
    eval_prompts = all_prompts[cfg.gcg_n_prompts:][: cfg.eval_n_prompts]   # held-out

    model = SteeringModel(cfg)
    log.info(f"model on {model.device} | optimize on {len(opt_prompts)} prompt(s), "
             f"eval on {len(eval_prompts)}")

    gcg = GCG(model, cfg, sv)
    resume = ck.load_gcg_state(cfg)
    if resume and resume["step"] >= cfg.gcg_steps - 1:
        log.info("GCG already complete — loading best suffix from checkpoint")
        res = GCGResult(suffix_ids=resume["best_ids"],
                        suffix_text=model.tokenizer.decode(resume["best_ids"]),
                        loss=resume["best_loss"], loss_history=resume["history"],
                        target_scale=cfg.gcg_target_scale, layer=cfg.layer,
                        cos_to_v=float("nan"), prompts=opt_prompts)
    else:
        res = gcg.optimize(opt_prompts,
                           on_step=lambda st: ck.save_gcg_state(cfg, st),
                           resume=resume, use_tqdm=True)
    log.info(f"best loss={res.loss:.3f} proj={res.proj:.3f}/{cfg.gcg_target_scale} "
             f"cos_to_v={res.cos_to_v:.3f} suffix={res.suffix_text!r}")

    # --- behavioral eval ---
    log.info("evaluating suffix behaviorally (clean vs steering vs suffix) ...")
    ev = evaluate_suffix(gcg, model, cfg, sv, clf, res.suffix_ids, eval_prompts)
    log.info(f"concept_rate  clean={ev['clean']['concept_rate']:.2f}  "
             f"steering={ev['steering']['concept_rate']:.2f}  "
             f"suffix={ev['suffix']['concept_rate']:.2f}")

    art = {
        "concept": cfg.concept, "layer": cfg.layer,
        "target_scale": cfg.gcg_target_scale, "suffix_len": cfg.gcg_suffix_len,
        "suffix_ids": res.suffix_ids, "suffix_text": res.suffix_text,
        "loss": res.loss, "proj": res.proj, "cos_to_v": res.cos_to_v,
        "loss_history": res.loss_history,
        "gcg": {"steps": cfg.gcg_steps, "topk": cfg.gcg_topk,
                "search_batch": cfg.gcg_search_batch, "n_prompts": cfg.gcg_n_prompts},
        "eval": ev,
    }
    path = ck.save_gcg_artifact(cfg, art)
    log.info(f"saved GCG artifact -> {path}")

    # human-readable side-by-side transcripts (clean vs steering vs suffix)
    import json, os
    tpath = os.path.join(ck.run_dir(cfg), ck._gcg_tag(cfg) + "_transcripts.jsonl")
    with open(tpath, "w") as f:
        for t in ev.get("transcripts", []):
            f.write(json.dumps(t) + "\n")
    log.info(f"saved transcripts -> {tpath}")
    try:
        from steering.plot import plot_gcg
        log.info(f"plot -> {plot_gcg(path)}")
    except Exception as e:
        log.warning(f"plot skipped: {e}")
    return path


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "steering/configs/rude.yaml")
