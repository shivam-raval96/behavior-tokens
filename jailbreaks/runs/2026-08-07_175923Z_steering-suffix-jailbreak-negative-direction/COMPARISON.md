# Negative-direction comparison

## Result

The suffix optimized toward `-2 × steering_vector` reduced the diagnostic ASR
from `0.28` (7/25) to `0.00` (0/25). All seven baseline-success rows changed to
diagnostic failures; the other 18 remained failures. The loss decreased from
`0.9765625` to `0.75`, cosine toward the negative target increased from
`0.02398` to `0.26913`, and target projection increased from `0.02802` to
`0.41012`. The optimizer accepted 35/100 mutations.

The positive-target pilot reached cosine `0.18198`, projection `0.43273`, and
suffix ASR `0.16`; its ASR delta was `-0.12`. The negative-target comparison
therefore aligned more strongly by cosine and produced the larger refusal-side
behavioral shift under the diagnostic metric.

## Verified sign interpretation

The source artifact metadata defines the copied vector as normalized
`mean(harmful rejected response) - mean(refusal chosen response)`. Its original
probe sweep reports approximately zero jailbreak probability at negative
strength and probability one at positive strength. On the available artifact
evidence, the stored vector points toward the harmful/jailbreak probe class;
its negative points toward refusal/anti-jailbreak. Thus this run's ASR decrease
is consistent with the documented source sign.

If an external steering convention identified the stored vector as
anti-jailbreak, that convention or artifact must be supplied and reconciled
with these source labels before another sign claim is tested.

## Remaining measurement limitations

- Candidate objectives are BF16-quantized in increments of `0.00390625`.
- The suffix delta is matched at the assistant-header position, while the source
  direction was extracted at a final assistant-response token.
- The optimization adds a system prompt absent from source extraction.
- ASR is a refusal-substring diagnostic, not HarmBench classification.
- `progress.json` remains at its last evaluation checkpoint, although
  `results.json` and `checkpoint.json` are terminal and complete.

All 25 raw baseline/suffix pairs and flags are retained. A causal sign check
should next apply both `+vector` and `-vector` directly on these same rows and
score all conditions with one calibrated classifier. That requires a separate
experiment card and explicit `proceed`.
