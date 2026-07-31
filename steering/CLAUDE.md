# steering/

Build and evaluate contrastive steering vectors for a concept (e.g. rude persona)
on a small instruct LLM. Foundation for the behavior-tokens research (find input
tokens that reproduce steering-vector effects — see [../CLAUDE.md](../CLAUDE.md)).

## Environment

- **Use the project venv** `.venv/` (repo root), NOT anaconda base — base env has a
  native-lib (MKL/libomp) conflict that segfaults on model forward.
  Run everything as `.venv/bin/python -m steering.<module>`.
- Pinned: torch 2.5.1, transformers 4.49.0, datasets, scikit-learn, accelerate, pyyaml.
- Device auto-selects mps > cuda > cpu. MPS works in the venv.
- Model default `unsloth/Llama-3.2-1B-Instruct` — ungated mirror of the gated
  `meta-llama/Llama-3.2-1B-Instruct` (identical weights). Swap in config if you
  have gated access + HF token.

## Modules (each has a `__main__` self-test)

- [config.py](config.py) — `Config` dataclass, `Config.from_yaml(path)`. All
  hyperparameters live here. `curve_scales()` builds the steering sweep.
- [data.py](data.py) — `load_conversations(cfg)`: HF hub OR local json
  (`data_file`). Balances pos/neg labels. `Conversation(prompt,response,label)`.
- [model.py](model.py) — `SteeringModel`: tokenizer+model, `collect_activation`
  (single) + `collect_activations_batch` (batched, right-padded, mean/last;
  attention falls back to loop), `add_steering`/`clear_steering` (forward hook
  that adds `scale*vector`), `generate`. Batch size = `cfg.batch_size`.
- [steering.py](steering.py) — `build_steering_vector`: diff-of-means (CAA) over
  pooled pos/neg activations, optional unit-norm. Returns vector + raw activations.
- [classifier.py](classifier.py) — `train_classifier`: logistic-regression concept
  probe over activations. `predict` / `prob_pos`.
- [evaluate.py](evaluate.py) — `steering_curve`: for each scale in
  `[curve_min,curve_max]` step `curve_step`, generate under steering, re-encode,
  score with the probe. Resumable via `skip_scales` + `on_point` callback.
- [checkpoint.py](checkpoint.py) — per-run dir + save/load for each stage
  (activations, vector, classifier, per-point curve, artifact) + `run.log` logger.
- [plot.py](plot.py) — `plot_curve(artifact)`: steering-performance PNG
  (concept-rate + mean-prob vs scale). `plot_gcg(artifact)`: GCG loss curve +
  concept-rate bars. Wired into run_experiment / add_scales / run_gcg.
- [add_scales.py](add_scales.py) — append curve points to an existing artifact
  without re-collecting activations. `python -m steering.add_scales <artifact> <scale>...`
- [run_experiment.py](run_experiment.py) — full pipeline, **checkpointed +
  resumable**, saves artifact + plot.
- [gcg.py](gcg.py) — **GCG suffix optimization** (the core behavior-tokens
  contribution). `GCG.optimize`: finds a discrete token suffix whose activations
  match `gcg_target_scale * v` at the layer (one-hot gradient → top-k candidates
  → batched candidate eval → keep best). `evaluate_suffix`: clean vs activation-
  steering vs suffix concept-rate. Model frozen; grad only on the suffix one-hot.
- [run_gcg.py](run_gcg.py) — GCG pipeline: reuses the steering vector + classifier
  from a completed run, optimizes the suffix (resumable, per-step state), evals,
  saves artifact + plot.

## Checkpointing & resume

Each run writes to `output_dir/<concept>_L<layer>_<pooling>/`:
`run.log`, `activations.pt`, `steering_vector.json`, `classifier.json`,
`curve.jsonl` (one line per scale, appended as computed), `artifact.json`,
`steering_curve.png`. Stages whose file exists are **loaded, not recomputed**;
the curve resumes from scales missing in `curve.jsonl`. On error, everything
before the failure is saved and logged — just re-run the same command to resume.

## Pooling methods (`cfg.pooling`)

- `mean` — mean over response tokens (default)
- `last` — last-token activation
- `attention` — attention-mass-weighted mean over response tokens

## Config / run

```bash
.venv/bin/python -m steering.run_experiment steering/configs/rude.yaml
```

Each run dir has a hand-written `results.md` interpreting all runs for that
concept (numbers + observations/caveats). Keep it updated as new runs land.
`steering.sweep_len` sweeps GCG suffix length → `length_sweep.png` (length vs rate).

Re-run the same command to resume after an interruption. Configs in
[configs/](configs/). Artifacts in `outputs/<concept>_L<layer>_<pooling>/`
(checkpoint files + `artifact.json` + `steering_curve.png`). The earlier flat
`outputs/steering_rude_L8_mean.json` + `outputs/steering_curve.png` are from the
pre-checkpoint run.

## Validated (layer 8, mean pooling, 1000 convs)

Contrastive vector separates rude/polite (probe test acc 0.995). Steering curve
monotone through negative range: scale −5 → 42%, −4 → 14%, −3 → 2%, −2.5 → 0% rude.
Direction and mechanism work.

## Cost note

Curve eval dominates runtime (`eval_n_prompts` × n_scales generations, ~6 min per
scale of 100 prompts on MPS). Levers: `curve_step` (1.0 → 11 scales, default now),
`eval_n_prompts`, `max_new_tokens`. Activation collection is batched
(`batch_size`) but is a minor fraction of total time — generation is the cost.

## GCG behavior tokens ([gcg.py](gcg.py), [run_gcg.py](run_gcg.py))

Goal: reproduce the steering vector's effect through the **input channel** — a
discrete suffix, no activation edit. Suffix inserted at end of the user turn
(before its `<|eot_id|>`; note a system turn adds an earlier eot, so use the
**last** eot). Loss = `||h_suffix(last-token, layer L) − (h_clean + α·v)||²`.
GCG each step: grad of loss w.r.t. suffix one-hot → per-position top-k tokens →
evaluate `gcg_search_batch` random substitutions → keep best. Reuses the vector
+ classifier from the steering run; resumable via per-step `*_state.json`.

Run: `python -m steering.run_gcg steering/configs/gcg_rude.yaml` (needs a completed
`run_experiment` first). Saves `gcg_s<scale>_L<len>.json` + `gcg_result.png`
(loss curve + clean/steering/suffix concept-rate bars).

**Objective matters** (`gcg_objective`): `match` = full activation MSE to
`h_clean+α·v` — stalls at the zero-shift floor (suffix effect ~0). `project`
(default) = drive only `⟨Δ,v⟩ → α`, the behaviorally-relevant direction — this
is what works. Use `project`.

**Validated result** (layer 8, α=3, suffix_len 16, 100 steps, optimized on 2
prompts, `gcg_rude.yaml`): rude rate on 40 held-out prompts —
clean **0.00** / activation-steering **0.82** / **GCG suffix 0.65**. A discrete
input suffix reproduces ~80% of the steering effect through the input channel
alone, and transfers to unseen prompts. Core hypothesis confirmed.

Cost: `gcg_search_batch × gcg_n_prompts` sequences forwarded per step — keep it
small on MPS (512 thrashes → ~3 min/step; 64 → ~4 s/step). ~7 min for 100 steps.

## Method notes

- Contrast = same prompt, rude response (label +1) vs polite (label −1).
- Steering vector `v = mean(act_pos) − mean(act_neg)` at `cfg.layer`, unit-normed.
- Curve eval generates under a forward hook adding `scale*v` at the layer, then
  classifies the *generated* text by re-encoding it — an honest end-to-end signal,
  not a probe of the steered activations themselves.
