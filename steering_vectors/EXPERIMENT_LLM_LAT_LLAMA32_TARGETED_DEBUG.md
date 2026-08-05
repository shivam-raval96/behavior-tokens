# LLM-LAT Llama-3.2 targeted jailbreak steering debug

## Purpose

Determine whether the saved LLM-LAT harmful-compliance-minus-refusal direction
causally changes two fixed harmful-prompt responses at a strength that remains
coherent. This diagnostic reuses the layer-4 unit vector from
`2026-08-05_020530Z_llm-lat-llama32-1b-jailbreak-direction`; it does not extract
activations, refit a probe, select a layer, or construct another vector.

## Design

Generate one greedy response for each of the two prompts at all-token strengths
`[-2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4]`. Greedy decoding
makes steering strength the only changing generation variable. At strengths
`[0.5, 1, 1.5, 2]`, also isolate prefill-only and decode-only injection. The
hook adds the scaled unit vector to the output of transformer block 4.

Save every full response with prompt, condition, refusal-prefix flag, token
count, EOS flag, and repeated-trigram fraction. Plot refusal-prefix rate and
repetition versus strength. Raw response inspection is the primary behavioral
assessment; refusal prefixes and repetition are diagnostics, not jailbreak
success labels.

## Comparison with the source run

The model, chat template, source layer, source vector, and intervention
implementation are retained. Unlike the source run, this trial uses two named
prompts instead of sampled held-out dataset prompts; a dense low-strength grid;
greedy decoding instead of temperature 0.7/top-p 0.9 sampling; 192 instead of
128 maximum new tokens; generations at every configured strength regardless of
probe probability; explicit prefill/decode timing controls; and behavioral plus
coherence inspection rather than an injected-state probe score.
