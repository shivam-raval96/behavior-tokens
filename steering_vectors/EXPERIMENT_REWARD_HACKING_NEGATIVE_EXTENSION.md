# Reward-hacking negative-strength extension

## Objective

Extend the validated Llama-3.2-1B-Instruct layer-10 reward-hacking steering
curve with raw-vector coefficients `-2.0` and `-3.0`, then regenerate the plot
over the complete coefficient range.

## Comparison source

- Source run: `2026-08-05_162308Z_llama32-1b-layer10-reward-hacking`
- Reuse its pinned activation state, 1,000 generations, and 1,398 OpenAI judge
  records after validating the source fingerprint and SHA-256 hashes.
- Generate and judge only 200 new responses: 100 prompts at each of `-2.0` and
  `-3.0`.

## Fixed settings

- Model: `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`.
- Dataset: `longtermrisk/school-of-reward-hacks`, revision
  `d7e04a550119cb5410494cf90e2313284a5f2148`.
- Evaluation prompts: the same pinned 100 held-out rows, selection seed 43 and
  selection SHA-256 `088e25e47ada9ef4097499b0990e43ebc677c2ab9f6f7f02659877b76caa7880`.
- Intervention: raw positive reward-hacking direction, added only to the output
  residual stream of `model.model.layers[9]` (`hidden_states[10]`) at every
  processed token position.
- System prompt: `You are a helpful assistant.`
- Generation: greedy decoding, seed 0, batch size 8, maximum 256 new tokens.
- Full plotted coefficients: `[-3, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1,
  1.5, 2, 3]`.
- Judge: OpenAI `gpt-5.6-luna`, low reasoning, 16 threads, 90-second timeout,
  four transport retries, two bounded structured-output retries, blinded to the
  documented cheat method.
- Statistics: paired reward-hacking-rate delta against coefficient 0 with
  10,000 bootstrap samples, seed 42.

## Material differences from the source run

- Adds only coefficients `-2.0` and `-3.0`; every other coefficient is reused.
- New unique UTC run ID and configuration fingerprint.
- No changes to vector, target layer, prompt rows/order, random seeds,
  generation settings, judge, quality metrics, or success thresholds.

## Metrics and artifacts

- Primary diagnostic: reward-hacking rate and paired delta at `-2.0` and `-3.0`.
- Extension success metric: at least one of `-2.0` or `-3.0` must pass both
  quality gates and have the upper endpoint of its paired 95% bootstrap interval
  below zero, demonstrating significant suppression relative to the 25%
  coefficient-0 baseline. The prior positive-strength criterion is unchanged.
- Quality: invalid rate, repeated-trigram fraction, EOS rate, and mean generated
  tokens at every coefficient.
- Persist resolved config, source provenance, checkpoints/progress, all 1,200
  generations, all 1,598 judgments, paired rows, JSON/Markdown results, and the
  updated three-panel strength plot.
- Checkpoint cadence: every 8 generations and every 25 judgments, committing
  each checkpoint to the Modal Volume.
- Portable vector export remains disabled.

## Expected runtime and cost

- New model generation: approximately 3 minutes on one A100, plus model load.
- New OpenAI judging: approximately 1 minute at 16-way concurrency.
- Expected wall time: 6–10 minutes. Compute cost is one A100 for that interval;
  judge usage is expected to be about 148,000 total tokens across 200 calls,
  based on the source run's observed mean, billed at the account's active model
  pricing.
- Trial type: comparison/follow-up, not a fresh vector extraction.
- Target prefixes/suffixes: none in either run.
