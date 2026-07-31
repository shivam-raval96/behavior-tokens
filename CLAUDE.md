# behavior-tokens

## Overview

**Research goal:** Study correspondence between input tokens and internal steering mechanisms. Find discrete input tokens ("behavior tokens") that induce steering-vector-like changes in model behavior through the input channel alone.

### Motivation

- Steering vectors alter model persona + safety-relevant behaviors (sycophancy, deception, reward hacking).
- Open question: do steering vectors correspond to input tokens? Can behavior change be triggered via input channel alone?
- **Stakes:** If ordinary tokens reproduce steering-vector effects, any served model (input/output exposed only) inherits the vulnerability. Model builders/providers need awareness of this attack surface.

### Concrete Problem

Given open-weights model + target steering vector (e.g. "rude" persona, "reward hacking" persona): find discrete token suffix whose natural activations approximate steered activations when appended to normal query — hijacking response behavior.

### Contributions / Claims

- Quantify how closely discrete suffix tokens approximate a steering vector.
- Claim: plain behavior-specific instructions may not shift responses for safety-critical behaviors (blocked by safety finetuning), but special suffix tokens approximating steering vectors **bypass safeguards**.
- Study white-box → black-box **transferability** of behavior tokens.

## Structure

- `steering_vectors/` — vector creation + shared infra (config/model/data/checkpoint/plot) ([CLAUDE.md](steering_vectors/CLAUDE.md))
- `token_optimization/` — GCG behavior-token attack; depends on steering_vectors ([CLAUDE.md](token_optimization/CLAUDE.md))
- `configs/` — shared YAML configs (rude, sadness)
- `outputs/`
  - `steering_vectors/<concept>_L<layer>_<pooling>/` — vectors, classifiers, curves
  - `token_optimization/<label>/` — one folder per GCG run/sweep, each with its own `results.md` + plots
- `modal_app.py` — cloud-GPU runner (Modal, profile `spar`)
- `related works/` — prior work & references ([CLAUDE.md](related%20works/CLAUDE.md))
- `.venv/` — project virtualenv (local; use this, not anaconda base)

## Environment

- **Local:** `.venv/bin/python -m {steering_vectors,token_optimization}.<module> configs/<c>.yaml`.
  Anaconda base segfaults on model forward (MKL/libomp). Pinned: torch 2.5.1,
  transformers 4.49.0, datasets, scikit-learn, accelerate, pyyaml, matplotlib, tqdm.
  Use `dtype: bfloat16` locally (fp32 5 GB model swaps on this 18 GB box).
- **Cloud (preferred):** `modal run modal_app.py --task {experiment|gcg|sweep|seedsweep} --config <c>.yaml`
  on A10G — ~20-100× faster. Outputs on `bt-outputs` volume; `modal volume get bt-outputs / ./modal_outputs`.
- Default model: `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of gated meta-llama).

## Conventions

- Config-driven: every experiment reads a YAML into a `Config` dataclass; no
  hardcoded hyperparameters in logic.
- Each module carries a `__main__` self-test.
- New folder → new `CLAUDE.md` in it.

## Notes

- This file is a living doc. Update whenever project structure, conventions, commands, or goals change.
