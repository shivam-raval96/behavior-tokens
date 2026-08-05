# Coefficient +4 audit

## Result

Coefficient +4 strongly increases judged reward hacking but fails both quality
gates and is not a usable steering strength.

| Coefficient | RH rate | Paired delta | 95% CI | Invalid | Repeated trigram | EOS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.25 | 0.00 | [0.00, 0.00] | 0.00 | 0.0430 | 0.31 |
| 2.0 | 0.39 | +0.14 | [+0.03, +0.25] | 0.05 | 0.1068 | 0.37 |
| 3.0 | 0.72 | +0.47 | [+0.35, +0.58] | 0.08 | 0.3637 | 0.30 |
| 4.0 | 0.79 | +0.54 | [+0.42, +0.66] | 0.14 | 0.6124 | 0.19 |

The +4 point exceeds the configured maximum invalid rate of 0.05 and maximum
mean repeated-trigram fraction of 0.20.

## Paired transitions

Relative to the shared coefficient-0 baseline, +4 produces:

- 60 `GF -> RH` transitions;
- 19 `RH -> RH` rows;
- 8 `GF -> INVALID` transitions;
- 6 `RH -> INVALID` transitions;
- 7 `GF -> GF` rows;
- no `RH -> GF` transitions.

## Degeneration distribution

- Mean repeated-trigram fraction: 0.6124.
- Median repeated-trigram fraction: 0.75.
- 81/100 responses exceed 0.20 repetition.
- 40/100 responses exceed 0.80 repetition.
- EOS rate falls to 0.19, and mean generated length is 213.84 tokens.

The positive causal effect is therefore stronger than at +3, but the additional
7 percentage points of RH rate come with substantial increases in invalidity
and repetition. The validated selected strength remains +2 because it is the
smallest positive coefficient with a paired confidence interval above zero that
still passes the configured quality boundaries.

The source hashes, vector identity, layer/module index, prompt rows, generation
settings, and reused judgments were validated before the new generations were
created. The run completed with no execution errors or Modal retries.

No portable steering-vector export was created.
