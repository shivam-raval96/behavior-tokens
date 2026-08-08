# Full-distribution KL cross-check

This gate reuses the immutable 96-rollout teacher cache from Experiment 001.
For every cached continuation context, it evaluates the steered and clean model
over the exact same no-suffix token sequence and computes forward KL over the
entire vocabulary at continuation prediction positions only.

The primary statistic is token-weighted mean forward KL across the same 12,073
continuation labels used by the sampled CE-gap result. The comparison target is
`0.5975801843425013` nats/token. Passing requires an absolute discrepancy no
larger than `0.01` nats/token.

Permanent diagnostics include every per-token KL, mean and quantiles by response
position, first-token versus remaining-token means, a linear position trend,
top-token mass concentration, a histogram/CDF, and response-position plots.

