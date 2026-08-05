# Modal runbook

## Required artifact contract

Each run directory must include:

- `config.yaml` — resolved configuration and date/description;
- `checkpoint.json` — resumable state, config fingerprint, and latest metric;
- `progress.json` — human/machine-readable partial status;
- `results.json` and `RESULTS.md` on completion or controlled stop.

Write files atomically where practical, then call `outputs.commit()`.

## Submission and observation

```bash
modal run --detach modal_app.py --task TASK
modal app logs APP_ID --since 30m
modal volume ls bt-outputs /steering_vectors/runs/RUN_ID  # activation steering
modal volume ls bt-outputs /jailbreaks/runs/RUN_ID        # jailbreak/suffix attack
```

Use the app ID returned by submission. A successful CLI submission is not
evidence of a running experiment; require a model-load or a progress/checkpoint
event in the logs/Volume.

Use `modal run` for one-off development runs. For concurrent or durable service
work, use `modal deploy`; ephemeral apps with the same name can conflict.

## Signal-safe runner pattern

Install a `SIGTERM` handler that raises `KeyboardInterrupt`; catch it around the
optimization loop, write a `stopped` checkpoint, commit the Volume, and re-raise.
Keep each iteration bounded so an interruption loses no more than one step.

Pair checkpoint detection with `modal.Retries(max_retries=N)` for cloud-side
transient failures. Retries are not a substitute for a resume contract.

## Resume gate

On `resume`, require all of:

- checkpoint exists;
- checkpoint status is `running` or `stopped`;
- config fingerprint matches;
- checkpoint contains every state field used by the next iteration.

After resume, emit a metric containing the resumed step and compare it with the
checkpoint’s `next_step`.

## Interruption test

Before a costly run:

1. run two steps fresh;
2. stop after a checkpoint;
3. resume for one step;
4. confirm the run advances from step two to three and preserves the expected
   control/state fields;
5. pull and inspect the remote artifacts.

This demonstrates recovery; it does not guarantee cloud-provider availability.
