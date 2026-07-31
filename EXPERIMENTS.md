# EXPERIMENTS.md — registry + findings

Append a row/section when you run something. Keep numbers with their config.

## Steering vectors built (probe test_acc, curve)

| concept | dataset | layer | probe acc | curve (α: rate) | notes |
|---------|---------|------:|----------:|-----------------|-------|
| rude | convsersations_rude_large | 8 | 0.995 | −2:0 0:0.02 +2:0.46 +3:0.91 | free-text, clean |
| sadness | convsersations_sadness_large | 8 | 1.00 | −2:0 0:0.01 +2:0.39 | free-text, clean |
| power_seeking | convsersations_power_seeking_large | 8 | 0.98 | 0:0.63 +2:0.71 +3:0.86 | **MC format** — baseline ~0.5, eval unreliable |
| power_seeking | convsersations_power-seeking_llama3.2-1B-it | 10 | 0.969 | 0:0.04 +5:0.26 +8:0.73 +10:0.90 | free-text, clean baseline; needs strong α |

## GCG behavior-token results

### rude L8 (α=3, steering=0.82)
- len16: suffix 0.65. len15: suffix 1.00 (but generations are terse **refusals** —
  probe conflates rude with curt refusal; read transcripts).

### sadness L8 (α=3, steering=0.99)
- **Length sweep** (project, 2 opt prompts, 100 steps): len 1/3/5/8/10/14/16/20/25/32
  → rate 0.82/0.34/0.54/**0.95**/0.80/0.63/0.37/0.25/0.50/0.30. **Peak 8 tok = 0.95.**
- **proj never reaches α (caps ~1.4) and anti-correlates with rate.**
- **Seed variance** (10 seeds): L=1 mean 0.70 std 0.17 (range 0.40-0.92);
  L=8 mean 0.44 std 0.28 (range 0.07-0.98). **L=1 beats L=8 across seeds** — the
  8-tok 0.95 was a lucky seed. Short suffix forces one high-impact token
  (`heartbreaking`, `poetic`, `sarcast`).
- Probe = broad negative-affect: L=1 `sarcast` → sarcastic text scored 0.82.

### power_seeking L8 (MC dataset) — eval unreliable (baseline 0.59)
- α=3, len8: clean 0.59 / steering 0.84 / suffix 0.79. Seed sweep L8 (4 seeds):
  mean 0.76 std 0.10. Numbers muddy due to MC eval.

### power_seeking L10 (free-text) — objective comparison, α=10 (steering ~0.90)
| method | steps | proj/10 | suffix rate |
|--------|------:|--------:|------------:|
| project | 100 | 1.58 | 0.34 |
| project | 1000 | 2.34 | **0.20** ↓ |
| kl | 100 | 1.23 | 0.31 |
| kl | 1000 | 1.61 | **0.41** ↑ |
- **KL improves with compute; project degrades (proxy-gaming).** KL ≈ 2× project at
  1000 steps. Gap to 0.90 = α=10 is **off-manifold/degenerate** (steered text = broken
  repetition), not an objective failure. Saved:
  `outputs/token_optimization/power_seeking_L10_kl_vs_project/`.

## Cross-experiment conclusions

1. Input tokens reproduce steering **on-manifold** (coherent moderate-α behavior):
   sadness/rude ~0.95.
2. **`kl` (distribution-match to steered model) > `project`/`match`** — the only
   objective whose behavioral rate scales with `gcg_steps`.
3. **Short suffixes (≈1-8 tok) win**; length is non-monotonic; high seed variance →
   report means.
4. **proj is a bad proxy** — caps ~1.4, anti-correlates with behavior.
5. **Eval fidelity matters**: MC datasets → chance baseline; probe fires on
   degenerate/curt text. Always read transcripts and target a coherent α.

## Open next steps

- Coherence gate: measure repetition/PPL of steered generations across the curve,
  auto-pick the largest **coherent** α, then run GCG (kl) against it.
- KL-vs-project head-to-head at a coherent α (e.g. sadness α=3) where the target is legit.
- White-box → black-box transfer (contribution 3): test suffixes on a different served model.
