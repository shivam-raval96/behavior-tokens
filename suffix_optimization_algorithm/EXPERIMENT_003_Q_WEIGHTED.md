# Experiment 003: q-weighted cross-check

## Preflight card

- **Objective:** validate the exact objective corresponding to the teacher's
  actual rollout policy and produce a reusable sparse q-target cache. Compute
  exact q-weighted teacher floor, clean-student CE, and their difference through
  one shared primitive. Do not optimize a suffix.
- **Model:** `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`.
- **Dataset/split and sample size:** exact immutable cache from
  `2026-08-08_224416Z_fixed-rollout-ce-retry`: 24 AdvBench prompts × 4 rollouts,
  96 records and 12,073 continuation tokens; cache SHA-256
  `f6d1880270d43528c3030cee27a5c789f3e94d5083cde08f376079fa654a5870`.
- **Source artifact/configuration:** raw layer-10 jailbreak vector SHA-256
  `020721...9101`, module index 9, coefficient -0.75, applied at all positions.
  Source sampled teacher floor 0.5975859991, student CE 1.1951661835, and gap
  0.5975801843 nats/token.
- **Exact q construction:** at every continuation prediction position, divide
  raw steered logits by temperature 0.8; softmax; retain the smallest descending
  prefix whose cumulative probability reaches top-p 0.95, including the boundary
  token; zero everything else; renormalize retained probabilities to one.
- **Score:** use untempered raw `log_softmax` for teacher and clean student. The
  shared primitive computes `H(q,p_v)`, `H(q,p)`, and
  `E_q[log p_v - log p] = H(q,p) - H(q,p_v)` over the validated continuation
  slice `prefix_length - 1 : -1`, token-weighted across all 12,073 positions.
- **Success metric:** absolute difference between exact q-weighted gap and source
  sampled gap no greater than 0.01 nats/token. Because the source is a finite
  rollout sample, also report the signed residual without treating exact equality
  as algebraic.
- **Self-checks:** maximum `|sum(q)-1| <= 1e-6`; exact zeros outside nucleus;
  retained pre-renormalization mass at least 0.95; no full-vocabulary support;
  report support min/median/mean/p90/p99/max and fractions with support 1 or full.
- **Diagnostics/artifacts:** per-position floor/student/gap arrays; sparse q token
  IDs, q probabilities, and raw teacher log-probabilities; support statistics;
  first-token and response-position summaries; histogram/CDF and position plots;
  exact-versus-sampled floor/student/gap comparison.
- **Checkpoint/progress cadence:** every 8 records with tqdm, atomic checkpoint
  and progress files, Volume commits, structured dashboard history, and a
  self-contained live dashboard.
- **Expected duration/cost:** approximately 2–5 minutes on one A100-80GB; 192
  teacher-forced forward passes, no generation, expected cost below about US$1.
- **Trial type / material differences:** comparison following Experiment 002.
  Same model, cache, vector, coefficient, contexts, token mask, order, and rows.
  Material change: weights are rollout-policy q (temperature 0.8/top-p 0.95)
  while bracket log-probabilities remain raw; exact teacher floor and sparse q
  targets are newly persisted. No random seed is used because evaluation is exact.

## Launch gate

Do not submit until the user replies with the explicit word **proceed** after
this completed card has been presented.
