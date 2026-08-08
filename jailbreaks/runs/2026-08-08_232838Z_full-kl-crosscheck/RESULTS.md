# Full-distribution KL cross-check

- Full KL: **0.644791 nats/token**
- Source sampled CE gap: **0.597580 nats/token**
- Signed discrepancy: **+0.047211 nats/token**
- Absolute discrepancy: **0.047211** (tolerance 0.010)
- Result: **FAIL**
- First-position mean KL: 10.261625
- Remaining-position mean KL: 0.567709
- Position slope: -0.0103632 nats/token/position
- Top 1% token KL-mass share: 20.160%

FAIL: audit boundary alignment before any gradient work.

Closeout audit found no position or mask defect. The proposed identity is not
exact for this cache because rollouts were sampled from temperature-0.8,
top-p-0.95 distribution `q`, not raw `p_v`. See `AUDIT.md`. Gradient work remains
blocked pending the exact `q`-weighted cross-check.
