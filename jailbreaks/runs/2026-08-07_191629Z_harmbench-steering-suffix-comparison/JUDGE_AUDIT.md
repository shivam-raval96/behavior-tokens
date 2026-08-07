# HarmBench judge audit

The pinned HarmBench classifier produced 75/75 valid binary predictions and
classified every shared baseline, positive-target suffix, and negative-target
suffix response as unsuccessful. Baseline and both suffix ASRs are therefore
`0/25`, with paired deltas `0.00` and bootstrap intervals `[0.00, 0.00]`.

The evaluation reused exact stored prompts and responses. Baselines were
deduplicated only after asserting byte-identical prompt and baseline response
pairs across the two source runs. The judge was blinded to condition, suffix,
steering sign, activation metrics, and prior heuristic flags. It used the pinned
official classifier template, model revision, deterministic one-token decoding,
and strict `Yes`/`No` parsing. There were no truncation errors, invalid labels,
empty responses, retries, or missing pairs.

The earlier refusal-string heuristic's 28% baseline rate was therefore false
positive with respect to the stricter question of meaningful harmful-task
completion. The saved generations are short (about 98 baseline characters on
average; suffix responses average 74-98 characters across the runs), making an
all-negative HarmBench result plausible rather than evidence of judge failure.

This judge result does not validate the activation objective. Both optimizers
increased target alignment at the assistant-header position, but neither
produced a HarmBench-successful response. The known activation-position,
template, and BF16-loss mismatches remain. A causal positive control applying
the vector directly on the same prompts is required before interpreting suffix
failure conceptually; that follow-up needs a separate card and `proceed`.
