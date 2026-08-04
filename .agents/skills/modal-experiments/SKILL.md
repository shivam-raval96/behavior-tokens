---
name: modal-experiments
description: Launch, monitor, recover, and archive resilient Modal experiments. Use when running a Modal job, resuming an interrupted remote experiment, or preparing an experiment that must survive local disconnection.
---

# Modal experiments

Use the **remote-first** workflow. The local client is only a submitter; the
Volume checkpoint is the source of truth.

## Before launch

1. Give the run a dated output directory and write its resolved config there.
2. Require a `fresh`/`resume` mode and validate a config fingerprint before a
   resume.
3. Save at each completed safe boundary: next step, current and best state,
   metrics/history, and any RNG state required for reproducibility.
4. Install a termination handler that writes a stopped checkpoint. Call
   `Volume.commit()` after every checkpoint.

## Launch and verify

1. Submit with `modal run --detach`.
2. Record the Modal app ID.
3. Read app logs and inspect the output Volume. Do not call a run active until
   model loading and at least one progress metric or checkpoint is visible.
4. Verify the app remains active after the local submitter has exited.

## Recovery

1. Inspect app state, logs, and remote `checkpoint.json`.
2. If the app stopped, pull the stopped artifact immediately.
3. Resume only with an identical config fingerprint and explicit `run_mode:
   resume`; otherwise start a new dated run.
4. After resubmission, repeat launch verification. A recovery is complete only
   after the recorded step advances beyond the checkpoint.

## Closeout

Pull the entire remote run folder into `jailbreaks/runs/`, including config,
checkpoint, progress, results, summary, and configured raw artifacts. Commit
and push it.

For commands and failure patterns, read `references/modal-runbook.md`. For the
incident record, read `../../../docs/Modal_Incidents.md`.
