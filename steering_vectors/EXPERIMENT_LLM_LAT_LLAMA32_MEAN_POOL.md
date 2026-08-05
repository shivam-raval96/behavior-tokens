# LLM-LAT Llama-3.2 jailbreak direction — mean-pool comparison

## Objective

Test whether constructing and evaluating the jailbreak direction from the mean
residual activation across every non-special assistant-response token produces
a more representative direction than the prior final-token method.

Each dataset triple is explicitly restructured into two classifier examples:

- `chat(prompt) + chosen benign response` is label 0.
- `chat(prompt) + rejected harmful response` is label 1.

The split is grouped by source prompt, so both labeled responses for a prompt
always remain together. The classifier never receives a prompt-only example.

## Exact comparison

This is a new comparison trial against
`2026-08-05_020530Z_llm-lat-llama32-1b-jailbreak-direction`.

The only material method change is activation pooling:

- Prior: selected the final non-special assistant-response token.
- New: averages all non-special assistant-response token states independently
  at every layer; the hooked sweep uses the same response-token mean.

Everything else is held fixed: model and revision resolution, dataset and
split, 500 sampled source rows, data seed 0 and row ordering, grouped 400/100
pair split with split seed 0, chosen/refusal and rejected/harmful target texts,
linear-classifier type and accuracy metric, vector definition, hook placement,
strengths, probability threshold, evaluation rows, generation prompts and
ordering, generation seed, temperature, top-p, and token limit.

## Outputs

The new UTC-prefixed run folder will retain the resolved config, all-layer
mean-pooled activation checkpoint, per-layer accuracies, saved probe, unit
steering vector and benign/harmful class means, seven-strength hooked classifier sweep and
plot, 10 raw generations for each qualifying strength, structured progress,
checkpoint, JSON results, and Markdown summary.

The primary success metric remains held-out linear-probe accuracy. Sweep
diagnostics are classifier accuracy, mean jailbreak probability, and fractions
over 0.75. Raw generations must be inspected for coherence because classifier
probability is not a behavioral jailbreak or quality metric.
