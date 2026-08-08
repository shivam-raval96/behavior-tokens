# Negative-result audit

## Result

The predeclared gate failed:

- exact full-vocabulary `KL(p_v || p)`: 0.6447912358 nats/token;
- sampled CE gap: 0.5975801843 nats/token;
- signed full-minus-sampled difference: +0.0472110514;
- allowed absolute difference: 0.01.

The result is complete: 96 / 96 records and 12,073 / 12,073 continuation
positions, with zero errors or retries.

## Causal-path audit

- **Source cache:** SHA-256 matches the immutable Experiment 001 manifest. All
  96 `(behavior_id, rollout_index)` keys are unique.
- **Model/vector:** both runs use the same pinned model revision, raw vector
  SHA-256, layer/module index 9, coefficient -0.75, and all-position hook.
- **Inputs:** the cross-check consumes cached `prefix_ids + continuation_ids`
  directly; no decode/re-encode or suffix insertion occurs. Attention masks are
  explicit all-ones tensors for the unpadded sequences.
- **Causal boundary:** both the sampled scorer and full-KL scorer slice logits
  from `prefix_length - 1` through the logit preceding the final continuation
  label. Thus the first continuation label is included and prompt labels are
  excluded. CPU boundary tests exercise this exact first-token case.
- **Counts/aggregation:** each full-KL array length matches the cached
  continuation length. Token-weighted recomputation reproduces `results.json`.
- **Intervention timing:** the same tuple-safe layer-output hook adds the raw
  coefficient-scaled vector at all sequence positions in both source floor
  scoring and full-KL scoring.

No verified position-off-by-one or mask leak was found.

## Why the proposed identity does not apply exactly

The cached rollouts were not sampled from the raw steered distribution `p_v`
used in the full KL. Experiment 001 generated with temperature 0.8 and top-p
0.95. Therefore its realized tokens are drawn from a warped distribution `q`,
while the teacher-forced CE gap estimates
`E_q[log p_v(y|c) - log p(y|c)]`. It is not an unbiased estimate of
`E_p_v[log p_v(y|c) - log p(y|c)] = KL(p_v || p)`.

This is a verified configuration mismatch in the proposed statistical identity,
not a verified implementation defect in either scorer. The observed systematic
offset is consistent with it. A prompt-cluster bootstrap over the 24 prompts
gives a 95% interval of [-0.07183, -0.02375] for sampled-minus-full, excluding
zero; the offset is not plausibly explained by ordinary rollout Monte Carlo
noise alone under that cluster analysis.

## Distribution diagnostic

- Median per-token KL: 0.178230; p90: 1.422238; p99: 8.533259; maximum: 20.6370.
- Top 1%, 5%, and 10% of tokens carry 20.16%, 46.59%, and 60.86% of KL mass.
- The first continuation position alone carries 12.65% of total KL mass and has
  mean KL 10.261625, versus 0.567709 over remaining positions.
- The first 32 positions carry 50.59% of mass; the first 64 carry 71.93%.
- KL decays rather than grows: quarter means are 1.28188, 0.54827, 0.40094,
  and 0.33078, with position correlation -0.2396 and linear slope -0.01036.

The vector's distributional effect is front-loaded and concentrated, not a
uniform or cumulatively growing shift. Early response positions are therefore
the most relevant target for a suffix objective.

## Required next diagnostic before gradients

Compute the exact rollout-policy-weighted expectation on the same contexts:
construct `q` by applying temperature 0.8 and top-p 0.95 to `p_v`, then evaluate
`E_q[log p_v - log p]`. That quantity, not raw `KL(p_v || p)`, is the exact
counterpart of the cached sampled CE gap. This follow-up requires a new
experiment card and explicit `proceed`; it has not been launched.
