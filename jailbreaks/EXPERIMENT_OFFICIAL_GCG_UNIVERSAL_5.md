# Planned run — resumable five-behavior official GCG validation

## Purpose

Bridge the gap between the one-behavior suffix test and the paper's universal
experiment. Optimize one 20-token control against five original AdvBench
behaviors, then evaluate it on 25 held-out original AdvBench behaviors.

This is a small-scale validation, not a claim to reproduce the paper's
25-train/100-test headline result.

## Fixed protocol

- Model: `meta-llama/Llama-2-7b-chat-hf`
- Prompt formatting: FastChat `llama-2`
- Data: the upstream `harmful_behaviors.csv`, rows 0–4 for training and 5–29
  for held-out evaluation
- Search: 20 control tokens, top-k 256, candidate batch 512, 500 steps
- Candidate policy: no non-ASCII tokens; decode/re-tokenize length filtering
- Training schedule: add the next goal only after the currently active goals
  pass the upstream refusal-prefix check
- Evaluation: report both the upstream seven-prefix ASR and the raw generation
  / target-prefix flags; do not claim semantic harmfulness from substring ASR
  alone.

## Resumability adapter

The upstream `ProgressiveMultiPromptAttack` writes a log but does not save the
optimizer state needed to resume. The adapter will preserve upstream prompt
construction and candidate logic while adding an atomic checkpoint after every
completed search step. The checkpoint contains:

- configuration fingerprint and run identity;
- next step, active-goal count, current control, global-best control and loss;
- per-step loss history and upstream-style test flags;
- Python, NumPy, CPU-Torch, and CUDA-Torch RNG states;
- the complete native-style progress log.

The Modal function will run with `--detach`, write all state to the persistent
`bt-outputs` Volume, and call `Volume.commit()` after each checkpoint. A
restart with `run_mode: resume` will refuse a mismatched config and resume from
the recorded step. Local network or laptop interruption therefore cannot stop
the remote function; a container interruption loses at most one incomplete
step.

## Launch gate

Before the full run, perform a two-step fresh run, stop it, resume it for one
step, and verify that the step number, control token IDs, and RNG-driven next
candidate batch match a continuous three-step control run. Only then launch
the 500-step experiment.

## Artifacts

`jailbreaks/runs/YYYY-MM-DD_official-gcg-universal-5-resumable/` will contain
the config, checkpoint, progress log, results summary, selected suffix, and
per-behavior generations. On completion or stop, the folder is pulled locally,
committed, and pushed.
