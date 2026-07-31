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

- `README.md` — project readme
- `related works/` — prior work & references ([CLAUDE.md](related%20works/CLAUDE.md))
- `steering/` — steering-vector pipeline (build + evaluate) **and** GCG behavior-token optimization ([CLAUDE.md](steering/CLAUDE.md))
- `.venv/` — project virtualenv (use this, not anaconda base)

## Environment

- **Run Python via `.venv/bin/python`** (repo root venv). Anaconda base env
  segfaults on model forward (native MKL/libomp conflict). Pinned: torch 2.5.1,
  transformers 4.49.0, datasets, scikit-learn, accelerate, pyyaml.
- Default model: `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of gated
  `meta-llama/Llama-3.2-1B-Instruct`).

## Conventions

- Config-driven: every experiment reads a YAML into a `Config` dataclass; no
  hardcoded hyperparameters in logic.
- Each module carries a `__main__` self-test.
- New folder → new `CLAUDE.md` in it.

## Notes

- This file is a living doc. Update whenever project structure, conventions, commands, or goals change.
