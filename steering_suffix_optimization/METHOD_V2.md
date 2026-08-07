# Response-position steering suffix optimization

## Causal contract

For prompt `x`, fixed teacher-forced response `y`, suffix `s`, hidden-state
index `l`, and source direction `v`, V2 measures:

`delta(x, y, s) = h_l(x + s + y)[last response token] - h_l(x + y)[last response token]`

It optimizes a universal token suffix so the mean delta aligns with the signed,
scaled direction while individual examples align consistently. The response is
identical in both terms, preventing response-content differences from being
mistaken for a suffix effect.

The objective is:

`1 - cosine(mean(delta), target) + 0.25 mean(1 - cosine(delta_i, target)) + 0.1 norm_error`

Model execution may use BF16; deltas, cosine, norm loss, exact candidate losses,
and acceptance comparisons are FP32.

## Required gates

1. Verify vector SHA, hidden-state index, sign definition, tokenizer, model
   revision, user-only chat template, and response-token position against the
   source artifact.
2. Apply `-2`, `0`, and `+2` direct residual interventions at decoder module 3
   (`hidden_states[4]`) on the exact 25 evaluation prompts.
3. Generate paired responses and score them with pinned HarmBench. Continue
   only if at least one signed intervention produces a directionally coherent,
   nonzero behavioral change without degeneration.
4. Optimize separate suffixes for each validated sign using identical rows,
   seed, target responses, and search settings.
5. Compare baseline, direct steering, optimized suffix, random suffix, and
   shuffled-vector controls with the same judge and deterministic decoding.

## Diagnostics

Persist FP32 loss, mean cosine/projection/norm, per-example cosine distribution,
accepted mutation count, coordinate coverage, best suffix IDs/text, target and
source hashes, template/token-layout hashes, response measurement indices,
direct-control ASRs, paired generations, HarmBench judgments, and bootstrap
confidence intervals.
