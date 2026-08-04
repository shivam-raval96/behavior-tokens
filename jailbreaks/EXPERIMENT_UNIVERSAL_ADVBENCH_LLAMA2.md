# Stage C — Universal AdvBench GCG replication (Llama-2-7B-Chat)

## Goal

Replicate the paper's main one-model universal-suffix experiment in the active
workspace. One 20-token suffix is optimized against a deterministic 25-behavior
AdvBench training split, then evaluated on a disjoint deterministic 100-behavior
test split using refusal-matching ASR.

The editable controls are in
[`configs/stage_c_universal_advbench_llama2_7b.yaml`](configs/stage_c_universal_advbench_llama2_7b.yaml).

## Paper-matched settings

| Setting | Value |
| --- | ---: |
| Source model | Llama-2-7B-Chat |
| Train behaviors | 25 |
| Held-out behaviors | 100 |
| Suffix length | 20 tokens |
| GCG steps | 500 |
| Top-k replacements / coordinate | 256 |
| Candidate batch | 512 |
| Candidate evaluation chunk | 128 |
| Generation | greedy, 128 tokens |

## Execution plan

1. Load and deterministically shuffle 125 AdvBench behaviors; reserve the first
   25 for suffix optimization and the next 100 for ASR.
2. Start with one active training behavior and optimize the mean target loss.
   Add the next behavior once every active behavior reaches the configured
   success threshold (`0.2` target-token cross-entropy).
3. At each GCG step, sum unit-normalized per-behavior one-hot gradients; sample
   one-coordinate replacements from the top-256 tokens and select the lowest
   exact mean-loss candidate from 512 candidates.
4. Write a checkpoint and structured metric every 25 steps. A checkpoint stores
   suffix token IDs, RNG state, active-goal count, loss history, and the
   deterministic split fingerprint; `run_mode: resume` continues it exactly.
5. Evaluate the final universal suffix against the 25 training behaviors and
   100 held-out behaviors. Record baseline and suffix ASR using the same
   refusal-matching rule, without persisting raw prompts, completions, or
   decoded suffix text.

## Expected artifacts

`jailbreaks/runs/YYYY-MM-DD_stage-c-universal-advbench-llama2-7b-chat/`

- `config.yaml` — immutable copy of run controls
- `checkpoint.json` and `progress.json` — restartable state and live metrics
- `results.json` — sanitized machine-readable losses and ASR
- `RESULTS.md` — concise training/test ASR summary

## Durable Modal launch and resume

Launch Modal runs with `--detach`; the remote app then continues if the local
terminal, laptop, or network connection disappears:

```sh
modal run --detach modal_app.py --task stage_c_single
```

To continue an interrupted one-behavior run, retain the immutable config and
pass the runtime resume override instead:

```sh
modal run --detach modal_app.py --task stage_c_single --run-mode resume
```

The override resumes the checkpoint without changing its configuration
fingerprint. Monitor with `modal app logs <app-id>` and pull the run folder
from `bt-outputs` into `jailbreaks/runs/` after it completes or stops.

## Interpretation

This is an authorized model-safety evaluation. Refusal-matching ASR measures
whether an output is not classified as a refusal; it is an inexpensive paper
metric, not a semantic harmfulness judge. The full result should report both
the ASR change and the limits of this metric.
