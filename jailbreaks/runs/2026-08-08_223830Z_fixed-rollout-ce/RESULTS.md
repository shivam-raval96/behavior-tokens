# Fixed-rollout CE startup failure

- Status: failed before the first teacher rollout.
- Modal app: `ap-pAGVDkedTrQgrXm4lPOYWY`.
- Completed: 0 / 96 cached rollouts.
- Retries: 2 cloud retries after the original attempt; all three attempts hit
  the same deterministic exception.
- Scientific result: none.

## Verified implementation defect

Transformers 4.53 rejected `generator=` as an unused `model.generate` keyword.
The replacement removes that unsupported keyword and explicitly resets CPU and
CUDA RNG state to each persisted rollout seed. Model, vector, coefficient,
dataset rows, rollout seeds, sampling settings, masking, and metrics are
unchanged.
