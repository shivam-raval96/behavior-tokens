# SKILLS.md — commands & entry points

## Environment

- **Modal (preferred):** base env has `modal` CLI; profile `spar` active
  (`modal profile activate spar`). GPU A10G. Volumes: `bt-outputs` (results),
  `bt-hf-cache` (model cache).
- **Local:** venv at `.venv/` (repo root). `.venv/bin/python -m <module> <cfg>`.
  Pinned: torch 2.5.1, transformers 4.49.0, datasets, scikit-learn, accelerate,
  pyyaml, matplotlib, tqdm. NOT anaconda base (segfaults on model forward).

## Modal tasks (run from repo root)

```bash
modal run modal_app.py --task experiment --config <c>.yaml            # steering_vectors.run: vector+probe+curve
modal run modal_app.py --task gcg        --config <c>.yaml            # token_optimization.run: one GCG run
modal run modal_app.py --task sweep      --config <c>.yaml --lengths 1,8,16,32
modal run modal_app.py --task seedsweep  --config <c>.yaml --lengths 8 --seeds 10 --out <name>.json
modal volume get bt-outputs /<path> ./outputs/<path> --force          # pull results
```
`run_task` rewrites the config: `output_dir=/outputs`, `device=auto` (→ cuda).
Note: Modal Volume commits only at function end — a killed run loses its uncommitted
json (rebuild from the run log if needed).

## Local modules (same code, slower)

```bash
.venv/bin/python -m steering_vectors.run       configs/<c>.yaml        # vector + probe + curve
.venv/bin/python -m steering_vectors.add_scales <artifact.json> <scale>...   # extend a curve
.venv/bin/python -m token_optimization.run     configs/<c>.yaml        # single GCG
.venv/bin/python -m token_optimization.sweep_len   configs/<c>.yaml 1 8 16 32
.venv/bin/python -m token_optimization.seed_sweep  configs/<c>.yaml --lengths 8 --seeds 10 --out <name>.json
```
Every module has a `__main__` self-test. Plots: `steering_vectors.plot`
(`plot_curve`), `token_optimization.plot` (`plot_gcg`, `plot_length_sweep`).

## Config knobs (steering_vectors/config.py `Config`)

- Model/data: `concept`, `model_name`, `layer`, `dtype`(=bfloat16), `dataset_name`,
  `n_samples`, `prompt_key/response_key/label_key`, `pos_label/neg_label`.
- Vector: `pooling` (mean|last|attention), `normalize`, `batch_size`.
- Probe: `clf_test_frac`, `clf_C`, `seed`.
- Curve: `eval_n_prompts`, `curve_min/max/step`, `max_new_tokens`, `gen_temperature`.
- GCG: `gcg_suffix_len`, `gcg_steps`, `gcg_topk`, `gcg_search_batch`,
  `gcg_target_scale` (α), `gcg_objective` (project|match|kl), `gcg_kl_tokens`,
  `gcg_n_prompts`, `gcg_init_token`, `gcg_seed`.
- IO: `output_dir` (=`outputs`).

## Output layout

- Vector: `outputs/steering_vectors/<concept>_L<layer>_<pooling>/` →
  `steering_vector.json`, `classifier.json`, `activations.pt`, `curve.jsonl`,
  `artifact.json`, `steering_curve.png`.
- GCG run label: `<concept>_L<layer>_a<α>_len<suffix_len>[_<objective>][_seedN]`;
  sweeps: `<concept>_L<layer>_lensweep` / `_seedsweep`. Each folder has
  `artifact.json`, `transcripts.jsonl`, plot png(s), auto `results.md`, `run.log`,
  `gcg_state.json` (resume).

## Datasets used (HF, `shiv96/…`)

`convsersations_rude_large`, `convsersations_sadness_large`,
`convsersations_power_seeking_large` (MC — weak eval),
`convsersations_power-seeking_llama3.2-1B-it` (free-text + precomputed resid).
All: columns `prompt`, `response`, `label` (+1 concept / −1 neutral), balanced.
