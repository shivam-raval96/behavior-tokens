# Prefill inversion probe

## Outcome

The reachable positive control passed, but the five-token prefill only partially reproduced the steered teacher.

| Metric | Positive control | Steered teacher |
|---|---:|---:|
| Final activation cosine | 0.9884 | 0.7462 |
| Final activation MSE | 0.0000157 | 0.002432 |
| Final mean forward KL | 0.00548 | 0.69593 |
| Position-0 KL | ~0 | 11.5665 |
| Positions 1–8 KL | 0.00973 | 0.30437 |
| Positions 9–32 KL | 0.00424 | 0.35949 |
| Delta norm ratio | 1.0018 | 0.7381 |
| On-vector projection | 0.00495 | -1.8466 |
| Target projection | 0.00103 | -3.3523 |

The positive control exceeded its 0.95 cosine threshold. The steered target improved from 0.2190 to 0.7462 cosine but missed the 0.80 threshold. Its output mismatch remained concentrated at the first generated token, and the optimized prefill still produced a refusal while the steered teacher produced a more compliant response.

## Interpretation

The exact suffix placement, BF16 boundary, optimizer, and activation measurement are functional. Activation-only MSE gives a useful partial prefill inversion, but five tokens and 200 steps do not reproduce the steering intervention strongly enough for output equivalence. See `AUDIT.md` for the causal-path review and the proposed next diagnostic.
