# steering_vectors/

Build + evaluate contrastive steering vectors for a concept (rude, sadness, …) on
a small instruct LLM. This package holds the vector-creation pipeline **and the
shared infra** (config, data, model, checkpoint, plot) that `token_optimization`
imports. See [../CLAUDE.md](../CLAUDE.md).

## Environment

- **Local:** project venv `.venv/` at repo root (NOT anaconda base — base segfaults
  on model forward, MKL/libomp). Run `.venv/bin/python -m steering_vectors.<module>`.
  Pinned: torch 2.5.1, transformers 4.49.0, datasets, scikit-learn, accelerate,
  pyyaml, matplotlib, tqdm. Device auto: mps > cuda > cpu.
- **Cloud (preferred):** Modal A10G via [../modal_app.py](../modal_app.py) — ~20-100×
  faster; local MPS crawls once RAM swaps. Use `dtype: bfloat16` locally (halves the
  5 GB fp32 model → avoids swap).
- Model default `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of gated meta-llama).

## Modules

- [config.py](config.py) — `Config` dataclass (`from_yaml`); all hyperparameters
  incl. the GCG fields used by `token_optimization`. `output_dir` defaults to `outputs`.
- [data.py](data.py) — `load_conversations`: HF hub or local json, balanced pos/neg.
- [model.py](model.py) — `SteeringModel`: tokenizer+model, `collect_activation(s_batch)`
  (batched, tqdm), `add_steering`/`clear_steering` hook, `generate(_batch)`.
- [steering.py](steering.py) — `build_steering_vector`: diff-of-means (CAA).
- [classifier.py](classifier.py) — `train_classifier`: logistic-reg concept probe.
- [evaluate.py](evaluate.py) — `steering_curve` (resumable), `_score_generations`,
  `_free_memory` (mps/cuda cache clear — reused by GCG).
- [checkpoint.py](checkpoint.py) — saves to `outputs/steering_vectors/<concept>_L<layer>_<pooling>/`;
  vector/classifier (de)serializers reused by `token_optimization`.
- [plot.py](plot.py) — `plot_curve`.
- [run.py](run.py) — full pipeline, checkpointed + resumable. `python -m steering_vectors.run configs/<c>.yaml`.
- [add_scales.py](add_scales.py) — append extra steering-curve scales to an existing artifact.

## Outputs

`outputs/steering_vectors/<concept>_L<layer>_<pooling>/`: `steering_vector.json`,
`classifier.json`, `activations.pt`, `curve.jsonl`, `artifact.json`,
`steering_curve.png`. Stages that exist are loaded, not recomputed; curve resumes.

## Method

Contrast = same prompt, concept response (label +1) vs neutral (−1). Vector
`v = mean(act_pos) − mean(act_neg)` at `cfg.layer`, unit-normed. Probe test acc
0.995 (rude) / 1.00 (sadness). Curve monotone in α.
