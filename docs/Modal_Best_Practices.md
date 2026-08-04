# Modal best practices

## Reliable experiment contract

Treat Modal jobs as remote processes and the local terminal as disposable.
Every long experiment must use detached submission, a persistent Volume, and a
resumable checkpoint written after each completed safe boundary.

Build images from declared dependencies and packaged source, keep module global
scope fast, use secrets for credentials, and pair explicit GPU/timeout choices
with retry-safe checkpoint loading. A retry without a valid resume path is not
fault tolerance.

## Required sequence

1. Create a dated run directory and copy the resolved config into it.
2. Start with `modal run --detach` and record the app ID.
3. Verify model load plus a metric/checkpoint in both logs and the Volume.
4. Monitor structured metrics; do not infer state from an old terminal session.
5. On stop/cancellation, inspect the checkpoint, pull the artifact, then resume
   only with a matching config fingerprint.
6. On completion or stop, sync the full run folder locally and commit/push it.

## Recovery standard

Local disconnects must not affect a detached Modal function. Cloud cancellation,
timeout, or code failure are different failure modes: recovery requires an
explicit resume path and a persisted checkpoint. Test that path with a short
intentional stop/resume cycle before a costly run.

## Output safety

Persist raw outputs only when the experiment configuration authorizes it. Keep
summaries concise, avoid printing sensitive raw artifacts in status updates, and
record exactly which evaluator produced each headline metric.
