# Modal platform choices

## App and image

- Keep global-scope work minimal: it executes during local app loading and
  remote container startup.
- Use `add_local_file` / `add_local_dir` to package code. Set `copy=True` when
  a following image build step must access those files.
- Use `modal.Secret.from_name(...)` for Hugging Face or tracking credentials.
- Prefer separate Volumes for immutable-ish caches/data and mutable run output.

## Compute

- Use CPU functions for preprocessing and metadata work.
- Use explicit GPU requests for model work; choose enough VRAM for weights,
  activations, and candidate batches rather than relying on an upgrade.
- An A100 request may be upgraded to A100-80GB and an H100 request to H200;
  append `!` when exact hardware is required for reproducibility.
- Request multi-GPU or multi-node only when the experiment actually needs it;
  more GPUs can increase queue time and recovery complexity.

## Development modes

- `modal run`: ephemeral, ideal for a single task.
- `modal deploy`: durable app definition, appropriate when work must coexist or
  be invoked later by name.
- Interactive GPU sandboxes are for debugging/profiling before a formal run;
  persist their workspace to a Volume and background long commands.

## Storage and retries

Call `Volume.commit()` after each checkpoint. On retry, reload the checkpoint
first, validate its config fingerprint, and record that a recovery occurred.
Avoid assuming a container-local file survives a retry or cancellation.
