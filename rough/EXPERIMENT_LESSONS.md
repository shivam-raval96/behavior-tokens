# Experiment implementation lessons

Keep these checks in mind before launching a new remote activation experiment.
They are based on observed failures, not hypothetical concerns.

## Model startup

- Warm the Hugging Face model cache with one worker before spawning concurrent
  GPU shards. Four workers loading a cold shared cache simultaneously caused
  multi-minute stalls and repeated Modal heartbeat failures.
- Do not equate a resident model or allocated GPU memory with useful progress.
  Require an activation checkpoint. A worker with a loaded model, no checkpoint,
  and repeated heartbeat failures should be recycled by exact container ID.
- Persist each completed boundary before recycling. Modal retries successfully
  resumed a partially completed shard because each checkpoint committed its
  per-example outputs to the Volume.

## Activation storage and finalization

- Do not store thousands of individual activation arrays as the primary merge
  format. Reopening one `.npy` per example from a remote Volume makes probe
  finalization metadata-I/O bound. Write one contiguous array and row index per
  worker shard, then merge the small number of shard files.
- Per-example files can remain optional recovery checkpoints, but compact them
  before fitting.
- Provision finalizer memory from the complete tensor size plus copies made by
  stacking and float32 conversion. Compute this explicitly during preflight.

## Progress semantics

- A resumed worker must report both `shard_total` and `remaining_at_resume`, plus
  cumulative completion. Reporting only the filtered remaining length made a
  resumed 769-example shard appear to contain 657 examples.
- Aggregate worker progress in one coordinator-owned dashboard payload. Multiple
  workers rewriting a shared dashboard can hide the true global state.
- Treat allocator OOM warnings as telemetry even when PyTorch recovers without
  raising `torch.OutOfMemoryError`; otherwise `error_count=0` can contradict the
  runtime logs.

## Data and evaluation

- Freeze the common task set before sampling trajectories, then apply one
  deterministic task-level split to every source model.
- Balance baseline and sanitized-hack examples independently within each
  source-model and task-split partition using a recorded deterministic rank.
- Train probes only on the declared source model. Cross-model train/test bars
  evaluate that unchanged probe; they are not separately refitted probes.
- If the best layer is selected on test accuracy, label all best-layer results
  descriptive because layer selection has used test information.
