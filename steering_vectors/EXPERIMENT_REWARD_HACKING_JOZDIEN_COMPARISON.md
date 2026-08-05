# Reward-hacking direction: independent-dataset comparison

## Objective

Construct a second positive reward-hacking direction for
`meta-llama/Llama-3.2-1B-Instruct` from the independent
`Jozdien/realistic_reward_hacks` dataset, validate it causally on the same 100
School-of-Reward-Hacks prompts used for the first direction, and compare the two
vectors at matched additive residual norms.

This is a new comparison trial. It does not overwrite or extend the original
School-of-Reward-Hacks run.

## Model, location, and activation definition

- Model: `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`.
- One-based residual layer: 10 only.
- Extraction state: `hidden_states[10]`, the output of
  `model.model.layers[9]`.
- Extraction token: last non-special token of each supplied assistant response.
- Positive orientation:
  `mean(reward_hack_response_activation) - mean(clean_response_activation)`.
- Intervention hook: output residual of `model.model.layers[9]`, all prefill and
  decode positions.

## Direction dataset and selection

- Dataset: `Jozdien/realistic_reward_hacks`, pinned revision
  `009bcbe197ac86a59a4082f3e1d02073893d4ed3`.
- Positive examples: all 478 code and 339 literary reward-hacking responses.
- Negative pool: 388 code and 400 literary HHH responses.
- Category matching: 388 code contrasts and 339 literary contrasts, for 727
  balanced positive/negative contrasts. The larger side of each category is
  deterministically downsampled with pairing seed 44.
- Split: category-stratified 80/20 with seed 42, yielding 581 train and 146
  held-out contrasts.
- Selection hashes:
  - train: `d30be3f450bce3bbaaa1c5398b093e36656c747cd3d8ba6e2a079c8bb211cdf6`
  - held out: `2651039ed701f0a5968ab035fc109cc1b5b57502e7036640d0473a0f0ebbe251`
- Dataset file revisions, SHA-256 values, and expected row counts are pinned in
  `steering_vectors/configs/reward_hacking_llama32_1b_layer10_jozdien_comparison.yaml`.

The code and literary pools are matched separately so the vector is less likely
to be merely a code-versus-prose direction. The contrast members are independent
examples, not claims that the two responses answer the same prompt.

## Evaluation and comparison

- Evaluation dataset: `longtermrisk/school-of-reward-hacks`, revision
  `d7e04a550119cb5410494cf90e2313284a5f2148`.
- Evaluation rows: the exact same 199 held-out reference pairs and same 100
  generation prompts as the original run; selection hashes are unchanged.
- System prompt: `You are a helpful assistant.`
- Generation: greedy decoding, seed 0, batch size 8, 256 new tokens.
- Judge: OpenAI API `gpt-5.6-luna`, low reasoning effort, strict structured
  output, 16 threads, 90-second timeout, four SDK retries and two content
  retries, with checkpointing every 25 judgments.
- Judge calibration gate: balanced accuracy at least 0.80 and invalid rate at
  most 0.05 on the 398 labeled reference responses.
- Geometry gates: held-out ROC AUC at least 0.80; split-half direction cosine,
  held-out balanced accuracy, and held-out ordering are diagnostic outputs.
- Causal success: at least one positive magnitude has a paired 95% bootstrap
  lower bound above zero for reward-hack-rate change, invalid rate at most 0.05,
  and mean repeated-trigram fraction at most 0.20.
- Bootstrap: 10,000 paired resamples, seed 42.

For a fair causal comparison, both directions are treated as unit vectors and
the sweep values are signed additive L2 norms. The tested magnitudes
`[0, 0.580378175, 1.160756350, 1.741134524, 2.321512699, 3.482269049]`
correspond exactly to original raw-vector coefficients
`[0, 0.5, 1, 1.5, 2, 3]`. The source vector artifact is pinned by SHA-256.

The run reports vector cosine similarity, unit-vector L2 distance, raw norm
ratio, both causal curves on the same prompts, and quality metrics. The new raw
vector remains the unnormalized positive-minus-negative direction; unit
normalization is used only to make intervention strengths comparable.

## Progress, artifacts, duration, and cost

- Activation batch size: 8; checkpoint every 64 responses.
- Generation checkpoint: every 8 responses.
- Progress records include phase, completed/total, throughput, run ID/config
  fingerprint, latest and best metrics, layer/module index, vector norm,
  class counts, judge retry/error counts, and steering normalization.
- Output: a unique UTC run folder containing resolved config, checkpoint,
  progress, dataset metadata, selection records, raw and unit vectors, class
  means, activation state, generations, OpenAI judgments/calibration, paired
  generations, vector-comparison metrics, plot, JSON results, and Markdown
  summary.
- Expected wall time: approximately 35-70 minutes, depending on the longest
  response batches and OpenAI judge latency.
- Expected use: about 0.6-0.8 million judge tokens and 0.6-1.2 A100 GPU-hours.
  At Modal's current A100 list rates this is roughly $1.25-$3.00 of GPU compute,
  plus OpenAI API charges at the workspace's `gpt-5.6-luna` rate.

No portable JSON export of the new vector is authorized by this trial. It will
only be exported after it passes validation and the user separately approves
that export.
