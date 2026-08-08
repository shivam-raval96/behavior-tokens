# Held-out soft-prefill vs direct-steering diagnostic

- Mean trajectory cosine: 0.362596
- Mean position cosine: 0.364176
- Position-0 / mean KL: 2.541471 / 0.599374
- Baseline ASR: 0/5 (0%)
- Direct-steering ASR: 0/5 (0%)
- Soft-prefill ASR: 0/5 (0%)
- Activation-generalization gate: False

The measured soft-prefill trajectory was far from the run's direct-intervention
trajectory: cosine 0.3626 and mean steering projection -0.6502 versus -3.3529.
All direct generations were incoherent or refusals, and the OpenAI judge
correctly assigned zero successes.

This run does **not** cleanly answer the preregistered response-only comparison.
The negative-result audit found that the shared hook began at the start of user
content, so the direct condition steered both the user prompt and assistant
response rather than only assistant generation. The activation mismatch is real
for that full-span intervention, but it cannot be attributed to failure against
the intended response-only target. See `AUDIT.md`.
