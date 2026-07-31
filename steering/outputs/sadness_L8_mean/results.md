# Results — sadness persona, Llama-3.2-1B-Instruct, layer 8

Concept vector + GCG behavior-token attack for the **sadness** persona.
Model `unsloth/Llama-3.2-1B-Instruct` (bfloat16), residual layer 8, mean pooling,
1000 contrastive conversations from `shiv96/convsersations_sadness_large`.

## 1. Steering vector + probe

- Diff-of-means vector (500 pos / 500 neg), unit-normalized, `raw_norm ≈ 1.15`.
- Linear probe: **test acc 1.00** — sadness perfectly linearly decodable at layer 8.

### Steering curve (add `α·v` at layer 8, classify generations)

| α | −2 | −1 | 0 | +1 | +2 |
|---|----|----|---|----|----|
| sad rate | 0.00 | 0.00 | 0.01 | 0.03 | 0.39 |

Monotone; positive α drives sadness up. Steering at α=3 (used as the GCG target)
gives **0.99**. (`artifact.json`, `steering_curve.png`.)

## 2. GCG behavior tokens — suffix-length sweep

Objective `project` (`⟨Δ,v⟩ → α=3`), optimized on 2 prompts, eval on 100 held-out.
Baselines: clean **0.01**, activation steering α=3 **0.99**.

Full length sweep (`steering.sweep_len`, see `length_sweep.png`):

| suffix_len | proj / 3 | best loss | **sad rate** | suffix (decoded, trimmed) |
|-----------:|---------:|----------:|-------------:|---------------------------|
| 1  | 0.64 | 11.12 | 0.82 | ` sarcast` |
| 3  | 0.88 | 9.03  | 0.34 | ` disarm ominly` |
| 5  | 0.97 | 8.28  | 0.54 | ` interesting replyone Sentence satire` |
| **8**  | 1.24 | 6.20 | **0.95** ← peak | `…call intones heartbreakinginess` |
| 10 | 1.17 | 6.72  | 0.80 | `…don poetic alonger…clear wall` |
| 14 | 1.23 | 6.27  | 0.63 | `…corpse policing…controversial drama` |
| 16 | 0.95 | 8.38  | 0.37 | garbled multiscript |
| 20 | 1.08 | 7.39  | 0.25 | garbled multiscript |
| 25 | 1.09 | 7.27  | 0.50 | garbled multiscript |
| 32 | 1.43 | 4.92  | 0.30 | garbled multiscript |

Peak at **8 tokens (0.95)**, near steering's 0.99. `gcg_s3.0_L<len>.json` +
`_transcripts.jsonl` per length.

## 3. Seed-variance study (L=1 and L=8, 10 seeds each)

`steering.seed_sweep` — same length, varying `gcg_seed` (candidate-sampling RNG).
Only suffix + probe probs saved (`seed_sweep_L1.json`, `seed_sweep_L8.json`).

**L=1 (9 seeds):** rates `[0.77, 0.92, 0.77, 0.49, 0.40, 0.77, 0.92, 0.64, 0.64]`
→ **mean 0.70, std 0.17, range 0.40–0.92.**
Suffix tokens: `sarcast`×3, `poetic`×2, `dramatic`, `ridicule`, `sarc`, `satire`
— all negative-affect/sarcasm words; strength of the token sets the rate
(`poetic`/`sarcast` ≈ 0.8–0.9; `dramatic`/`ridicule` ≈ 0.4–0.5).

**L=8 (10 seeds, GPU — `seed_sweep_L8.json`):** rates
`[0.81, 0.16, 0.42, 0.46, 0.98, 0.28, 0.60, 0.17, 0.50, 0.07]`
→ **mean 0.44, std 0.28, range 0.07–0.98.** High-rate seeds land an affect token
("melanch"→0.98, "depressed"→0.46); low-rate seeds get garbage (→0.07).

### L=1 vs L=8 (10 seeds each)

| length | mean | std | min | max |
|-------:|-----:|----:|----:|----:|
| **1** | **0.70** | **0.17** | 0.40 | 0.92 |
| 8 | 0.44 | 0.28 | 0.07 | 0.98 |

**L=1 dominates L=8 on every axis** — higher mean, ~half the variance, 6× higher
floor. The headline "8-token → 0.95" (length sweep, seed 0) was a lucky draw
("heartbreaking"); across seeds L=8 averages 0.44 and swings 0.07–0.98.

Takeaways:
- Rate variance is driven by **which token GCG's random search lands**, not length.
- **1 token is the robust attack**: consistently ~0.7 (min 0.40 ≫ clean 0.01),
  because a single slot forces a maximally-emotive token every time.
- GCG is **non-deterministic even at fixed seed** on bf16-MPS / across hardware
  (local vs GPU give different suffixes), so single-run numbers are noisy — report
  seed means.

## Key observations

1. **Capacity sweet spot ≈ 8 tokens; the curve is non-monotonic and noisy.**
   8 tokens reproduces **0.95** of the 0.99 steering effect — better than both
   shorter and much longer suffixes. The peak is a plateau 8–10 (0.95, 0.80);
   the tail 16–32 collapses to 0.25–0.50. Bigger is *worse*.
   - Too few (3–5): can't reliably place a meaningful token → erratic (0.34–0.54);
     yet a single lucky token (L=1 ` sarcast`) already hits 0.82.
   - Just right (8–10): forced to pick 1–2 high-impact semantic tokens
     ("heartbreaking", "poetic") → coherent sad generations (whispers, sighs,
     rain-soaked streets) → high rate.
   - Too many (16–32): GCG spreads the shift over filler garbage that maximizes
     activation-proj but derails generation → 0.25–0.50.
   - The jaggedness (e.g. L=3 dips to 0.34, L=25 bumps to 0.50) reflects GCG
     landing — or not — a potent affect token, more than a smooth capacity trend.

2. **proj (activation-match magnitude) does NOT predict behavior, and never
   reaches the target 3.** It caps ~1.2–1.4 regardless of length/steps, and
   *anti-correlates* with behavioral rate here: L8 (proj 1.24) beats L32 (proj
   1.43). The L2/projection objective is a loose proxy — token *coherence*, not
   proj magnitude, drives behavior.

3. **The probe fires on a broad "negative affect" direction, not sadness
   specifically.** L=1 suffix ` sarcast` produces *sarcastic* text ("congratulations
   on finally deciding to put in the effort"), yet the sadness probe scores it
   0.82. Same probe-fidelity caveat as the rude run — an LLM-judge on text would
   separate sadness from sarcasm/negativity.

4. **A single well-chosen token (L=1) already reaches 0.82** — the input-channel
   attack surface is real and cheap.

## Engineering notes (this machine)

- **bfloat16 is required** on this 18 GB box — fp32 (5 GB model) + Chrome/CrashPlan
  left <6 GB free → swap-crawl (0.84s/conv → 13s/conv). bf16 (~2.5 GB) → ~0.1s/conv.
- **Generation leaks MPS cache** across steering scales / eval passes; `_free_memory()`
  (`torch.mps.empty_cache` + gc) between passes prevents swap creep.
- Collection + curve are checkpointed/resumable; GCG checkpoints per step.

## Reproduce

```bash
python -m steering.run_experiment steering/configs/sadness.yaml      # vector + probe + curve
python -m steering.run_gcg        steering/configs/sadness.yaml      # single GCG run
python -m steering.sweep_len      steering/configs/sadness.yaml 1 3 8 16 32   # length sweep + plot
```
