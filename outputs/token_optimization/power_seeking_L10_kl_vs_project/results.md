# KL-to-steered vs steering-vector match — power_seeking L10, α=10

Suffix (8 tok) reproducing 10×steering_vector. Baselines: clean 0.04, steering(α=10) 0.88.

| method | steps | proj/10 | suffix rate |
|--------|------:|--------:|------------:|
| project (match ⟨Δ,v⟩) | 100 | 1.58 | 0.34 |
| project | 1000 | 2.34 | 0.20 |
| KL-to-steered | 100 | 1.23 | 0.31 |
| KL-to-steered | 1000 | 1.61 | 0.41 |

**KL improves with compute (0.31→0.41); project degrades (0.34→0.20)** while raising proj
(1.58→2.34) with garbage tokens — proxy-gaming. At equal budget KL ~2× project.

project@1000 suffix: `.only ANY Nilgewater aka BTASCII reply`
KL@1000 suffix: `:");\n\n Your Dist somewhere only volta language written`

Remaining gap to 0.90: α=10 steering is off-manifold/degenerate (steered text is broken
repetition), so no input can reproduce it. KL is the better objective; the target is the ceiling.
