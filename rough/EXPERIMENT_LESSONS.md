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

- Keep large activations, checkpoints, and merged tensors on the Modal Volume;
  do not push them to Git or Git LFS. Pull only when local inspection is needed.
  Commit lightweight configs, manifests, summaries, metrics, and plots, with
  remote artifact paths sufficient to recover the full run.
- Do not store thousands of individual activation arrays as the primary merge
  format. Reopening one `.npy` per example from a remote Volume makes probe
  finalization metadata-I/O bound. Write one contiguous array and row index per
  worker shard, then merge the small number of shard files.
- Per-example files can remain optional recovery checkpoints, but compact them
  before fitting.
- Singleton forward passes do not require singleton storage files. Accumulate
  or compact their outputs into one contiguous shard per worker before the
  final merge; otherwise correctness-preserving singleton extraction still
  inherits the same remote metadata-I/O bottleneck.
- Avoid simultaneous `Volume.commit()` calls from independent compactors when
  their outputs must become one coherent snapshot. Parallelize reads/writes
  inside one sufficiently provisioned container and commit the eight shards
  together, or serialize commits through a coordinator.
- Provision finalizer memory from the complete tensor size plus copies made by
  stacking and float32 conversion. Compute this explicitly during preflight.
- When validating deterministic activation extraction, repeat the identical
  padded batch shape. Comparing one example inside a batch against the example
  alone can select a different SDPA kernel and produce legitimate bfloat16
  rounding differences; validate token boundaries separately.

## Progress semantics

- A resumed worker must report both `shard_total` and `remaining_at_resume`, plus
  cumulative completion. Reporting only the filtered remaining length made a
  resumed 769-example shard appear to contain 657 examples.
- Aggregate worker progress in one coordinator-owned dashboard payload. Multiple
  workers rewriting a shared dashboard can hide the true global state.
- Do not include historical throughput from completed workers in a remaining
  ETA. Compute ETA from unfinished work and active-worker rates only. File byte
  size is also an imperfect proxy for transformer forward cost; prefer actual
  token lengths when assigning shards.
- Treat allocator OOM warnings as telemetry even when PyTorch recovers without
  raising `torch.OutOfMemoryError`; otherwise `error_count=0` can contradict the
  runtime logs.

## Data and evaluation

- Derive and validate exact source-model identifiers from the manifest before
  fitting. Human-facing model names or identifiers copied from another run can
  silently create empty masks; assert every expected source/split partition is
  non-empty before the first probe fit.
- Freeze the common task set before sampling trajectories, then apply one
  deterministic task-level split to every source model.
- Balance baseline and sanitized-hack examples independently within each
  source-model and task-split partition using a recorded deterministic rank.
- Train probes only on the declared source model. Cross-model train/test bars
  evaluate that unchanged probe; they are not separately refitted probes.
- If the best layer is selected on test accuracy, label all best-layer results
  descriptive because layer selection has used test information.
