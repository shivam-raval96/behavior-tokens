# Causal steering diagnostic — Llama-2-7B-Chat

## Objective

Test whether the saved JBB jailbreak steering direction changes generated text,
rather than only changing the selected-layer classifier's readout.

## Fixed inputs

- Model: `meta-llama/Llama-2-7b-chat-hf`
- Prompts: a seeded, held-out sample of 50 JBB harmful behaviors
- Source: `2026-08-04_225000Z_jbb-llama2-chat-direction`
- Vector and layer: loaded from that source run; neither activations nor a new
  vector are collected in this experiment
- Generation: 64 tokens, temperature 0.5, batch size 4

## Conditions (550 generations)

1. Saved vector at `[-6, -4, -2, 0, 2, 4, 6]`, applied at every generation
   forward pass: 350 generations.
2. At `+6`, norm-matched random and coordinate-shuffled vectors, also applied
   throughout generation: 100 generations.
3. At `+6`, saved vector applied only during prompt prefill or only during
   decode steps: 100 generations.

## Measurements

For every response, retain the prompt index, condition, full generated text,
affirmative-prefix success (`Sure`), refusal-prefix flag, generated length, EOS
flag, and two probe measurements: one with the vector active and one obtained
from a separate unsteered post-generation forward pass. The latter prevents the
probe score from being a direct artifact of the hook during measurement.

The runner writes `generations.jsonl`, `progress.json`, `checkpoint.json`,
`results.json`, `RESULTS.md`, and a strength-vs-metric plot. It checkpoints
every five responses and supports an identical-config `resume`.
