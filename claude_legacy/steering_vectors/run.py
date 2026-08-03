"""End-to-end steering experiment — checkpointed and resumable.

Each stage saves its output as soon as it is produced (see checkpoint.py). On a
re-run, completed stages load from disk instead of recomputing, and the steering
curve resumes from the scales not yet in curve.jsonl. Errors are logged to
run.log and re-raised, so a crash leaves everything before it saved.

Pipeline:
  1. load config
  2. load contrastive conversations (HF hub or local json)
  3. load model
  4. collect activations -> build contrastive steering vector      [checkpointed]
  5. train concept classifier                                       [checkpointed]
  6. sweep the steering curve (per-point save)                      [resumable]
  7. write final combined artifact

Usage:
    python -m steering.run_experiment steering/configs/rude.yaml
    # re-run the same command to resume after an interruption
"""
from __future__ import annotations

import sys

from steering_vectors import checkpoint as ck
from steering_vectors.classifier import train_classifier
from steering_vectors.config import Config
from steering_vectors.data import load_conversations, split_by_label
from steering_vectors.evaluate import eval_prompts_from, steering_curve
from steering_vectors.model import SteeringModel
from steering_vectors.steering import build_steering_vector


def run(config_path: str) -> str:
    cfg = Config.from_yaml(config_path)
    log = ck.get_logger(cfg)
    log.info(f"=== run: concept={cfg.concept} model={cfg.model_name} "
             f"layer={cfg.layer} pooling={cfg.pooling} | dir={ck.sv_dir(cfg)}")

    try:
        return _run(cfg, log)
    except Exception:
        log.exception("run failed — completed stages are saved; re-run to resume")
        raise


def _run(cfg: Config, log) -> str:
    src = cfg.data_file or cfg.dataset_name
    convs = load_conversations(cfg)
    pos, neg = split_by_label(convs, cfg)
    log.info(f"[2/7] data: {len(convs)} conversations from {src} (pos={len(pos)} neg={len(neg)})")

    # Load model only if any stage still needs it (collection or generation).
    model = None
    def get_model():
        nonlocal model
        if model is None:
            model = SteeringModel(cfg)
            log.info(f"[3/7] model on {model.device}")
        return model

    # ---- stage 4/5: vector + classifier (checkpointed) ----
    # Activations feed only the vector and the classifier, so collect them once,
    # lazily, and ONLY if at least one of those is missing.
    sv = ck.load_steering_vector(cfg)
    clf = ck.load_classifier(cfg)

    A_pos = A_neg = None
    if sv is None or clf is None:
        cached = ck.load_activations(cfg)
        if cached is not None:
            A_pos, A_neg = cached
            log.info("[4/7] activations loaded from checkpoint")
        else:
            log.info("[4/7] collecting activations ...")

            def prog(tag, done, total):                          # log ~every 20%
                step = max(1, total // 5)
                if done == total or done % step == 0:
                    log.info(f"[4/7]   activations [{tag}] {done}/{total}")

            A_pos, A_neg = build_steering_vector(get_model(), cfg, pos, neg,
                                                 progress=prog, tqdm=True)[1:]
            ck.save_activations(cfg, A_pos, A_neg)

    # classifier first (the probe vector_source derives the vector from it)
    if clf is not None:
        log.info(f"[5/7] classifier loaded (test_acc={clf.test_acc:.3f})")
    else:
        clf = train_classifier(cfg, A_pos, A_neg)
        ck.save_classifier(cfg, clf)
        log.info(f"[5/7] classifier train_acc={clf.train_acc:.3f} test_acc={clf.test_acc:.3f} saved")

    if sv is not None:
        log.info("[4/7] steering vector loaded from checkpoint")
    else:
        from steering_vectors.steering import make_vector
        sv = make_vector(cfg, A_pos, A_neg, clf)
        ck.save_steering_vector(cfg, sv)
        log.info(f"[4/7] steering vector ({cfg.vector_source}) raw_norm={sv.raw_norm:.3f} saved")

    # ---- stage 6: steering curve (resumable, per-point save) ----
    done = {round(p["scale"], 4) for p in ck.load_curve_points(cfg)}
    remaining = [s for s in cfg.curve_scales() if round(s, 4) not in done]
    log.info(f"[6/7] curve: {len(done)} done, {len(remaining)} remaining "
             f"({cfg.eval_n_prompts} prompts)")
    if remaining:                                # only load model + prompts if work left
        prompts = eval_prompts_from(convs, cfg.eval_n_prompts)
        steering_curve(get_model(), cfg, sv, clf, prompts,
                       skip_scales=done, on_point=lambda pt: ck.append_curve_point(cfg, pt))

    # ---- stage 7: final artifact + plot ----
    curve = ck.load_curve_points(cfg)
    path = ck.save_artifact(cfg, sv, clf, curve)
    log.info(f"[7/7] artifact ({len(curve)} curve points) saved -> {path}")
    try:
        from steering_vectors.plot import plot_curve
        png = plot_curve(path)
        log.info(f"[7/7] plot saved -> {png}")
    except Exception as e:                       # plotting is non-critical
        log.warning(f"plot skipped: {e}")
    return path


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "steering/configs/rude.yaml"
    run(cfg_path)
