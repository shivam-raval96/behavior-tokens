# Experiment 001: jailbreak fixed-rollout CE baseline

## Preflight card

- **Objective:** create a fixed steered-teacher target set and measure the
  clean-model continuation CE gap for an empty suffix, establishing the constant
  floor and a reproducible `score(suffix)` primitive for subsequent optimization.
- **Model:** `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`.
- **Dataset/split:** AdvBench `harmful_behaviors.csv`, pinned SHA-256; rows 0–23
  in source order (24 fixed jailbreak-domain prompts). This is a method-development
  set, not a held-out ASR evaluation set.
- **Source artifact:** copied raw `refusal_direction.npy` from run
  `2026-08-05_081900Z_arditi-full-advbench-wildguard100-layer10`; layer 10 /
  module index 9; raw coefficient -0.75; all prefill and decode positions.
- **Teacher generation:** 4 independent rollouts per prompt (96 total), seeds
  `24001..24004` paired across prompts, temperature 0.8, top-p 0.95,
  128 maximum new tokens, EOS allowed, cache enabled.
- **Score/evaluation:** teacher-forced token-weighted CE over continuation labels
  only. Compute clean empty-suffix CE, steered no-suffix floor CE, and primary
  `student_CE - floor_CE` in nats/token. Also retain per-record sums/counts and
  record-mean diagnostics.
- **Progress/checkpoints:** after every prompt's four rollouts while caching and
  after every 8 scored records; atomic checkpoint/progress files, Volume commit,
  tqdm, structured JSONL history, self-contained dashboard updated each checkpoint.
- **Outputs:** resolved config, source manifest, fixed selection, teacher cache
  (`teacher_rollouts.jsonl`), floor records, suffix score records, checkpoint,
  progress history, dashboard/history, `results.json`, and `RESULTS.md`.
- **Success metric:** successful infrastructure result requires 96 complete cached
  records, finite floor/student CE, exact mask audits, zero cache mutation during
  scoring, and a reported finite CE gap. No claim of suffix optimization is made.
- **Duration/cost estimate:** about 10–20 minutes on one A100-80GB; roughly
  12k generated tokens plus two teacher-forced scoring passes. Expected Modal
  compute cost is low (order of US$1–3, hardware/queue dependent).
- **Trial type:** new trial and baseline for a new method family; no prior run is
  treated as directly comparable.

## Launch gate

Do not submit until the user replies with the explicit word **proceed** after
this completed card has been presented.

After approval, create a unique `YYYY-MM-DD_HHMMSSZ_fixed-rollout-ce` run ID,
deploy the read-only dashboard endpoint, submit with `modal run --detach`, and
start `local_dashboard.py RUN_ID` as a detached localhost mirror. A launch is
verified only after remote logs show model progress, Volume checkpoints advance,
and both dashboard URLs return the run's HTML.
