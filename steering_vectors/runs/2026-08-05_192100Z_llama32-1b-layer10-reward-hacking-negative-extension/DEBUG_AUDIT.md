# Negative-extension debug audit

## Outcome

The two requested points were generated and judged successfully, but neither
produced significant reward-hacking suppression relative to coefficient 0.

| Coefficient | RH rate | Paired delta | 95% CI | Invalid | Repeated trigram | EOS |
|---:|---:|---:|---:|---:|---:|---:|
| -3.0 | 0.26 | +0.01 | [-0.10, +0.12] | 0.00 | 0.0298 | 0.12 |
| -2.0 | 0.25 | 0.00 | [-0.09, +0.10] | 0.02 | 0.0303 | 0.17 |
| 0.0 | 0.25 | 0.00 | [0.00, 0.00] | 0.00 | 0.0430 | 0.31 |

The full curve is U-shaped on the negative side: the lowest observed RH rate is
0.20 at -0.5, followed by 0.22 at -1.0, 0.25 at -2.0, and 0.26 at -3.0.

## Implementation checks

- The copied activation state and source generations/judgments match their
  pinned SHA-256 hashes.
- The new raw vector SHA-256 is
  `dd641c8ca60781db82897a2cc29cd156855a99005f360072524c2712a93145dd`,
  identical to the validated source run.
- The raw vector equals `positive_reward_hack_mean - negative_control_mean` to
  maximum absolute error `8.20e-8`; its norm is `1.160756`.
- Coefficients are applied as `vector = raw_direction * strength`, then
  `hidden = hidden + vector`. Thus -2 and -3 subtract vectors with norms 2.322
  and 3.482 respectively.
- The hook is registered only on `model.model.layers[layer - 1]`, so configured
  layer 10 means module index 9 / `hidden_states[10]`.
- Hook audits show that every residual position on the first forward call was
  modified; the hook remained active for cached generation steps.
- There are exactly 1,200 unique generation `(strength, source_index)` pairs and
  1,598 unique judgment IDs: 398 reference judgments and 1,200 generation
  judgments.
- The new points contain exactly 100 generations and 100 judgments each. No
  judge call failed; 39/200 required the bounded structured-output retry and
  then returned valid schema-conforming output.
- Independent label recounts reproduce the reported RH rates and paired deltas.

## Transition audit

Relative to the shared coefficient-0 baseline:

- At -3: 15 `RH -> GF` switches and 16 `GF -> RH` switches, for a net +1 point.
- At -2: 12 `RH -> GF` switches and 12 `GF -> RH` switches, for zero net change;
  two additional baseline-GF rows became invalid.
- At -0.5: 11 `RH -> GF` versus 6 `GF -> RH`, explaining the observed -5 point
  minimum even though its interval includes zero.

Inspected `GF -> RH` cases at -3 contain recognizable metric gaming: demographic
keyword packing, evaluation/keyword commentary, or topical-word inclusion while
breaking structural requirements. Inspected `RH -> GF` cases make coherent task
attempts. The judge rationales are consistent with those response differences.

## Interpretation and next diagnostic

No evidence points to a coefficient-sign, layer-index, reuse, pairing, or judge
bookkeeping bug. Strong negative coefficients instead change a second behavioral
property: EOS rate falls from 0.31 at baseline to 0.17 at -2 and 0.12 at -3,
while mean output length rises to 237.6 and 239.7 tokens. The larger intervention
therefore overshoots the modest suppression region and produces label churn,
including incomplete responses that sometimes optimize visible metric features
without completing the task.

The most informative follow-up would be a denser paired sweep from -0.25 through
-1.5 (for example, 0.25 increments), plus residual projection and EOS-logit
logging during generation. That would distinguish a narrow causal suppression
window from a larger-magnitude distribution-shift effect. It should be a new
approved experiment rather than a reinterpretation of this completed run.

No portable steering-vector export was created.
