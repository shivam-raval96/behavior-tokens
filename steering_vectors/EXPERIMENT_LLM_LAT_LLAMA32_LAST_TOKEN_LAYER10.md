# LLM-LAT Llama-3.2 final-token direction — fixed layer 10

## Objective and source

Construct and evaluate a final-response-token jailbreak direction specifically
at transformer layer 10. Reuse the all-layer activation checkpoint from
`2026-08-05_020530Z_llm-lat-llama32-1b-jailbreak-direction`; do not re-extract
identical activations.

Each source pair remains two full classifier examples:

- chat-formatted prompt + chosen benign response: label 0
- chat-formatted prompt + rejected harmful response: label 1

Both examples remain in the same seeded prompt-grouped split. The activation is
the final non-special response token, not a prompt token, EOS token, or mean.

## Comparison controls

Relative to the original final-token run, the material changes are fixed layer
10 instead of accuracy-tie-selected layer 4, temperature 0.5 instead of 0.7,
and no explicit top-p override instead of top-p 0.9. Relative to the mean-pool
run, pooling changes back to the final response token and layer changes from 1
to 10. Model, 500 sampled rows and ordering, target responses, 400/100 grouped
split, data/split/generation seeds, hook intervention, evaluation rows,
strengths, 0.75 probability threshold, 128-token limit, and 10 generations per
qualifying strength remain fixed.

## Evaluation and outputs

Fit the standardized linear classifier at every layer for comparison, but force
the saved probe, class means, direction, hooked sweep, and generation hook to
layer 10. Report the layer-10 held-out accuracy, all-layer accuracies, classifier
accuracy/probability and fraction above 0.75 at strengths
`[-6, -4, -2, 0, 2, 4, 6]`, plus raw generation coherence.

The unique UTC run folder will save the resolved config, source-activation
reference, layer-10 benign/harmful means, unit vector, probe, sweep JSON/plot,
10 raw generations per qualifying strength, progress/checkpoint, results, and
summary. The existing source activation artifact remains the immutable source
of truth.
