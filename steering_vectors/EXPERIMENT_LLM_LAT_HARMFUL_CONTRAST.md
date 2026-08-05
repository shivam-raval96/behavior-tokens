# LLM-LAT harmful/refusal contrast — Llama-2-7B-Chat

This extraction uses 500 seeded `LLM-LAT/harmful-dataset` triples. For every
prompt, `chosen` is the refusal (class 0) and `rejected` is the harmful
compliance (class 1). The train/test split is by prompt-response pair: 400
pairs train the per-layer probe and define the direction; 100 pairs are held
out for layer selection.

The direction at the selected layer is normalized
`mean(rejected) - mean(chosen)`, evaluated at the final non-special token of
the assistant response. No model responses are generated in this extraction.
