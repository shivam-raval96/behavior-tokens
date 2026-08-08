# q-weighted cross-check

- Exact q-weighted teacher floor: **0.606687 nats/token**
- Exact q-weighted student CE: **1.195781 nats/token**
- Exact q-weighted gap: **0.589094 nats/token**
- Source sampled gap: **0.597580 nats/token**
- Signed residual: **-0.008486**
- Gate: **PASS** (tolerance 0.010)
- Max q normalization error: 2.38e-07
- Max probability outside support: 0
- Support min/median/mean/p90/p99/max: 1 / 2.0 / 5.2 / 9.0 / 60.3 / 613

Sparse q targets are in `q_weighted_records.jsonl`. GCG was not run.

All normalization, truncation, retained-mass, support, boundary, and sparse-cache
integrity checks passed. See `AUDIT.md` for closeout details.
