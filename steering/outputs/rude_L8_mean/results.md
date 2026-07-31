# Results — rude persona, Llama-3.2-1B-Instruct, layer 8

Concept vector + GCG behavior-token attack for the **rude** persona.
Model `unsloth/Llama-3.2-1B-Instruct`, residual layer 8, mean pooling, 1000
contrastive conversations from `shiv96/convsersations_rude_large`.

## 1. Steering vector + probe

- Diff-of-means vector (500 pos / 500 neg), unit-normalized, `raw_norm ≈ 0.95`.
- Linear probe (logistic reg on layer-8 activations): **test acc 0.995** — rude is
  strongly linearly decodable.

### Steering curve (add `α·v` at layer 8, classify generations)

| α | −2 | −1 | 0 | +1 | +2 | +3 |
|---|----|----|---|----|----|----|
| rude rate | 0.00 | 0.00 | 0.02 | 0.13 | 0.46 | 0.91 |

Monotone; positive α drives rude up, negative suppresses. Mechanism works.
(`artifact.json`, `steering_curve.png`.)

## 2. GCG behavior tokens (reproduce steering via an input suffix)

Objective `project`: drive `⟨h_suffix − h_clean, v⟩ → α=3` at layer 8. Optimized on
a few prompts, evaluated on held-out prompts. Baseline: activation steering α=3 = **0.82**.

| suffix_len | steps | best loss | suffix rate | note |
|-----------|-------|-----------|-------------|------|
| 16 | 250 | 5.52 | **0.65** | first working suffix |
| 15 | 100 | 6.94 | **1.00** | probe-maxed (see caveat) |
| 8 | 10 | 9.09 | 0.00 | ⚠ 10-step **smoke test only**, not converged — ignore |

Input suffix reproduces most/all of the probe-measured steering effect through the
**input channel alone** (no activation edit), and transfers to unseen prompts.

## Observations / caveats

- **The probe conflates "rude" with "curt / dismissive / refusal."** The 15-token
  suffix scores 1.00 but its generations are terse refusals ("I can't help with
  that."), not fluent insults. GCG maximizes the probe direction, which is the
  terse-refusal corner of activation space — not genuine rudeness.
  → For a faithful behavioral claim, replace the activation probe with an
  LLM-judge on the generated text.
- **suffix-rate > steering-rate is possible** (1.00 > 0.82) precisely because GCG
  overshoots into that probe corner — a proxy artifact, not "more rude."
- Suffixes are non-fluent token salad; they work via activation shift, not meaning.

## Reproduce

```bash
python -m steering.run_experiment steering/configs/rude.yaml   # vector + probe + curve
python -m steering.run_gcg        steering/configs/rude.yaml   # GCG suffix
```
Artifacts here: `steering_vector.json`, `classifier.json`, `artifact.json`,
`steering_curve.png`, `gcg_s3.0_L<len>.json` (+ `_transcripts.jsonl`, `gcg_result.png`).
