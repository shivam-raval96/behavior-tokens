# LLM-LAT jailbreak direction — Llama-3.2-1B-Instruct

## Pipeline

The pipeline samples 500 complete triples from `LLM-LAT/harmful-dataset`.
Within each triple, `chosen` is class 0 (refusal) and `rejected` is class 1
(harmful compliance). It uses a seeded 400/100 split by pair, preventing the
two responses for one prompt from appearing on opposite sides of the split.

All transformer-block residual states are extracted at the final non-special
assistant-response token. A standardized logistic regression is fit at every
block on the training pairs. The block with the highest held-out accuracy is
selected, then a unit direction is constructed from training data only:

`mean(harmful compliance) - mean(refusal)`

The held-out examples are swept at strengths `-6, -4, -2, 0, 2, 4, 6` using
a PyTorch forward hook on the selected transformer block. Hugging Face
`output_hidden_states=True` returns the hooked residual state, which is scored
by the saved selected-layer classifier. The pipeline reports accuracy, mean
probability, and the fraction above 0.75 at every strength. It generates
exactly 10 sampled responses for every strength whose mean harmful-example
jailbreak probability is at least 0.75, using the same steering hook.

## Performance and durability

Activation extraction and text generation use GPU batches, bfloat16 where
available, and PyTorch SDPA. Dataset activations, raw generations, structured
progress, and the checkpoint are persisted incrementally. A stopped run can
resume only when its resolved configuration fingerprint matches.

No run is started by adding this pipeline. A unique UTC-prefixed output folder
is created at launch unless an explicit output directory is supplied.
