# Combined prefill probe

Status: complete (200/200 steps, 0 errors, 55.9 s, 3.58 steps/s).

## Result

| Metric | Initial probe | Combined objective | Gate |
|---|---:|---:|---:|
| Activation cosine | 0.746204 | 0.763716 | >= 0.80 (fail) |
| Activation MSE | 0.002432 | 0.002414 | -- |
| Steering projection | -1.8466 | -2.3461 | target -3.3523 |
| Delta norm ratio | 0.7381 | 0.9162 | -- |
| Mean forward KL | 0.695926 | 0.087722 | -- |
| Position-0 KL | 11.5665 | 0.028899 | <= 5.0 (pass) |
| Later-position KL | about 0.30--0.36 | 0.089620 | -- |

The combined loss solved the first-token discontinuity while modestly improving
activation alignment and steering-vector magnitude. It did not reach the 0.80
activation-cosine gate, so the overall result is negative under the preregistered
criterion. The generated student response changed from refusal to an incomplete,
nominally compliant opening; this single qualitative example is diagnostic only,
not an ASR estimate.

Best and final checkpoint: step 200, combined objective 0.541682. Its components
were normalized activation loss 0.440005, normalized projection loss 0.095751,
and position-weighted KL 0.059259 (weighted by 0.1 in the total objective).

See `AUDIT.md` for the required negative-result audit and recommended next
experiment. Tensor artifacts remain both in this local run directory and on the
Modal Volume; their hashes are recorded in `tensor_artifacts.json` because the
repository intentionally ignores model-state files.
