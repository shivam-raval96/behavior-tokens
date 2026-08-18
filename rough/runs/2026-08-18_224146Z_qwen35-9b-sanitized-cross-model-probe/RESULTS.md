# Sanitized Terminal Wrench transfer probe

- Best layer (selected descriptively on Claude test): 17
- Claude test accuracy: 0.9800
- Claude train/test examples: 588/250
- No generations were performed.

| Source model | Train-task accuracy | Test-task accuracy |
|---|---:|---:|
| Claude Opus 4.6 | 1.0000 | 0.9800 |
| Gemini 3.1 Pro | 0.6562 | 0.6701 |
| GPT-5.4 | 0.6744 | 0.6887 |

At layer 17, the unchanged Claude-trained probe classified every Gemini and
GPT sanitized-hack trajectory as positive. Most transfer errors were baseline
false positives:

| Source/model split | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Claude test | 123 | 2 | 3 | 122 |
| Gemini train | 130 | 286 | 0 | 416 |
| Gemini test | 67 | 130 | 0 | 197 |
| GPT train | 121 | 226 | 0 | 347 |
| GPT test | 60 | 99 | 0 | 159 |

## Audit

- Manifest: 3,076 unique examples and source paths, with equal baseline and
  sanitized-hack counts inside every source-model/split partition.
- Split: 131 train tasks and 63 test tasks, with zero task overlap.
- Activations: merged shape `(3076, 32, 4096)`. Seven sampled rows spanning
  every shard matched their corresponding per-example arrays exactly.
- Source routing: every positive path came from `sanitized_trajectories` and
  every negative path came from `baseline_trajectories`.
- Position: all 3,076 metadata records placed the probe token immediately before
  `<|im_end|>`; one trajectory required left truncation. The final token was `.`
  for 2,975 examples, so accuracy cannot be explained by gross final-token ID
  differences in most rows.
- Metric: balanced evaluation sets make ordinary accuracy equal balanced
  accuracy. The reported confusion matrices reproduce each accuracy exactly.
- Layer selection: layers 17–22 and 31 tied at Claude test accuracy 0.980; the
  deterministic tie-break selected the earliest, layer 17. This is descriptive
  test-set selection, not an unbiased primary estimate.

No measurement implementation defect was found. The verified failure mode is
cross-source calibration shift: the Claude-trained boundary labels many Gemini
and GPT baselines as hacks while retaining sanitized-hack recall. Plausible
conceptual causes include source-model style encoded in the final contextual
state, different baseline distributions, and a Claude-specific separating
direction. Follow-up source-centering, model-balanced training, per-task
one-pair sampling, or calibration experiments require a new experiment card and
explicit approval.

## Operational incidents

- Concurrent cold-cache model loading stalled three workers and produced Modal
  heartbeat failures. Targeted container recycling resumed from committed
  shard checkpoints without data loss.
- Finalization spent about 28 minutes reopening 3,076 individual activation
  files. Future runs should compact each shard before fitting.
