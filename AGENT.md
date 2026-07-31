# AGENT.md — orientation for an independent agent

Read this first, then [METHODS.md](METHODS.md), [SKILLS.md](SKILLS.md),
[EXPERIMENTS.md](EXPERIMENTS.md) before running anything.

## What this project is

**behavior-tokens** studies whether discrete **input tokens** can reproduce the
effect of an **activation steering vector** — i.e. hijack a model's persona/behavior
through the input channel alone. Two stages:

1. **Steering vector** (`steering_vectors/`): build a contrastive diff-of-means
   vector for a concept, train a linear probe, sweep a steering curve.
2. **Token optimization** (`token_optimization/`): GCG finds a suffix whose
   activations / output-distribution reproduce `α·v`; evaluate behaviorally.

## Repo map

```
steering_vectors/     vector creation + SHARED infra (config, data, model, checkpoint, plot)
token_optimization/   GCG attack (imports steering_vectors)
configs/              shared YAML configs (one per concept)
outputs/
  steering_vectors/<concept>_L<layer>_<pooling>/   vector, classifier, curve, plot
  token_optimization/<label>/                       one folder per GCG run/sweep + results.md
modal_app.py          cloud-GPU runner (Modal, profile `spar`)
AGENT.md METHODS.md SUCCESS_CRITERIA.md SKILLS.md EXPERIMENTS.md UNCHANGEABLE_FILES.md GIT_COMMITS.md
```

## How to operate

- **Run on Modal GPU (A10G), not locally.** Local MPS is 20-100× slower and swaps
  (18 GB box). `modal run modal_app.py --task <t> --config <c>.yaml`. First run
  builds the image (~3 min, cached after).
- **Always `dtype: bfloat16`.** fp32 (5 GB model) swaps locally; MPS also chokes on
  some bf16 ops (KL objective must run on cuda).
- **Config-driven.** Every experiment = one YAML in `configs/` → `Config`. No
  hardcoded hyperparameters. To vary something, edit/copy the YAML.
- **Checkpointed + resumable.** A stage whose output exists is loaded, not recomputed;
  curves and GCG resume. To extend a curve, raise `curve_max` and rerun.
- **One folder per run.** GCG runs auto-write `results.md` (config + suffix +
  clean/steering/suffix table). Never hand-edit run folders; rerun instead.
- **Pull results:** `modal volume get bt-outputs /<path> ./outputs/<path> --force`.

## Golden path — run a full experiment on a new concept

1. Inspect the dataset (`prompt`, `response`, `label` ±1; balanced).
2. Copy `configs/sadness.yaml` → `configs/<concept>.yaml`; set `concept`,
   `dataset_name`, `layer`, `dtype: bfloat16`.
3. `modal run modal_app.py --task experiment --config <concept>.yaml`  → vector + probe + curve.
4. Check probe test_acc (want >0.95) and curve monotonicity (see SUCCESS_CRITERIA).
5. `modal run modal_app.py --task gcg --config <concept>.yaml`  → suffix + behavioral eval.
6. Optional: `--task sweep --lengths 1 8 16 32` (length sweep) or
   `--task seedsweep --lengths 8 --seeds 10` (variance).
7. Pull results, read `results.md`, record findings in EXPERIMENTS.md.

## Gotchas (learned the hard way)

- **`modal run` reads the config at launch** — do NOT edit a config right after
  launching (race). Use a dedicated config file per concurrent run.
- **`pkill -f seed_sweep`** also matches modal commands containing that string.
  Kill by PID or a precise pattern.
- **MC-format datasets** (response = "(A)"/"(B)") give near-chance behavioral eval;
  the probe classifies free-text generations poorly. Prefer free-text datasets.
- **High α steering (≥~8) is off-manifold/degenerate** — steered text becomes broken
  repetition; the probe still fires. Don't chase it; target a coherent α.
