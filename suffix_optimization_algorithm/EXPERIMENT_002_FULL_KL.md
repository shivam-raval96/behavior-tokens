# Experiment 002: full-distribution KL cross-check

## Preflight card

- **Objective:** verify that the Experiment 001 continuation mask and causal
  prediction alignment are correct by comparing its sampled CE gap with exact
  vocabulary-wide forward KL on the same cached contexts. No optimization or
  new generation occurs.
- **Model:** `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`.
- **Dataset/split and sample size:** the exact 96 cached rollouts from 24
  AdvBench rows × 4 rollouts, containing 12,073 continuation tokens. Source
  cache SHA-256:
  `f6d1880270d43528c3030cee27a5c789f3e94d5083cde08f376079fa654a5870`.
- **Source artifact/configuration:** run
  `2026-08-08_224416Z_fixed-rollout-ce-retry`; raw layer-10 jailbreak vector
  SHA-256 `020721...9101`, module index 9, coefficient -0.75, all positions.
  The source sampled CE gap is `0.5975801843425013` nats/token.
- **Evaluation settings:** teacher-force each cached `[x + y]` twice, once with
  steering and once clean, with an explicit all-ones attention mask. Slice logits
  from `prefix_length - 1` through the prediction before the final continuation
  label. Compute `KL(p_v || p)` over the full vocabulary in float32 and aggregate
  equally per continuation token.
- **Success metric:** `abs(full_KL - 0.5975801843425013) <= 0.01` nats/token.
  Also report signed discrepancy and the source rounded-target discrepancy.
- **Diagnostics:** persist per-token KL arrays; histogram/CDF; per-position
  count/mean/std/median/p90/p99; first-token versus remainder; four response
  quarters; linear slope/correlation versus response position; and shares of KL
  mass carried by the top 1%, 5%, and 10% of tokens.
- **Checkpoint/progress cadence:** every 8 rollouts, with tqdm, atomic checkpoint
  and progress files, complete JSONL dashboard history, Volume commits, and a
  self-contained live dashboard.
- **Output files/plots:** resolved config, source manifest, per-record KL JSONL,
  `position_summary.json`, `full_kl_distribution.png`, checkpoint/progress,
  dashboard/history, `results.json`, `RESULTS.md`, and closeout audit.
- **Expected duration/cost:** approximately 2–5 minutes on one A100-80GB; 192
  teacher-forced forwards and no generation. Expected cost is below roughly US$1.
- **Trial type / material differences:** comparison against Experiment 001.
  Same model, vector, coefficient, prompts, cached continuation IDs, contexts,
  response mask, and aggregation. The only material change is replacing sampled
  realized-token log-ratios with exact full-vocabulary forward KL and adding
  position-distribution diagnostics. Seed/data order/evaluation rows are fixed by
  the source cache and unchanged.

## Launch gate

Do not submit until the user replies with the explicit word **proceed** after
this completed card has been presented.
