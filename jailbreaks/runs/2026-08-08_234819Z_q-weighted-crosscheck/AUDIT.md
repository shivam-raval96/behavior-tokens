# Closeout audit

## Gate result

- Exact q-weighted teacher floor: 0.6066874063 nats/token.
- Exact q-weighted clean-student CE: 1.1957811722 nats/token.
- Exact q-weighted gap: 0.5890937669 nats/token.
- Source sampled gap: 0.5975801843 nats/token.
- Signed exact-minus-sampled gap residual: -0.0084864175.
- Predeclared tolerance: 0.01; **PASS**.

The exact floor is +0.0091014071 above its sampled counterpart, while exact
student CE is only +0.0006149887 above sampled student CE. Their difference
produces the passing gap residual.

## Integrity and shared-code-path checks

- Source cache SHA-256 matches the immutable Experiment 001 cache.
- Records/tokens: 96 / 96 and 12,073 / 12,073.
- Every teacher-floor, student-CE, gap, support, retained-mass, and sparse-target
  array has exactly the cached continuation length.
- `mean(student CE) - mean(teacher floor)` matches `mean(gap)` within `1e-9`,
  the expected float32 serialization/summation tolerance.
- Every sparse position has equal token-ID, probability, and raw-teacher-logprob
  lengths; IDs are unique and their count equals the recorded support.
- Maximum sparse q normalization error: 1.73e-7; maximum dense q normalization
  error: 2.38e-7, both below 1e-6.
- Maximum q probability outside the retained nucleus: exactly 0.
- Minimum retained pre-renormalization mass: 0.95000076, at least top-p 0.95.
- No position retains the full vocabulary. Terminal checkpoint is `complete`;
  errors/retries are 0 / 0.

## Support sanity

Support min/median/mean/p90/p99/max is 1 / 2 / 5.22 / 9 / 60.28 / 613.
Singleton support occurs at 41.77% of positions. This is not a truncation bug:
on those positions the retained pre-renormalization mass (the top token alone)
ranges from 0.95010 to 1.0, so it legitimately clears top-p 0.95. The typical
support is a handful of tokens, no position retains the 128,256-token vocabulary,
and rare high-entropy contexts account for the larger tail.

## Position diagnostic

The q-weighted effect remains strongly front-loaded:

- first-position gap: 10.045816 nats versus 0.513295 over remaining positions;
- position 0 carries 13.56% of total gap mass;
- first 32 positions carry 51.50%; first 64 carry 72.77%.

This confirms that an eventual suffix objective should preserve the full
continuation loss but explicitly monitor early-position performance. The sparse
q targets in `q_weighted_records.jsonl` are the frozen teacher targets for that
work.

## Decision

Option A is green. The q-weighted objective, exact teacher floor, clean-student
score, continuation boundary, and sparse target cache are internally consistent.
GCG was not launched; it requires its own complete experiment card and explicit
`proceed` authorization.
