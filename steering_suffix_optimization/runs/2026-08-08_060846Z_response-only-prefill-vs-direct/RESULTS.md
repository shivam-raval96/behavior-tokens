# Held-out soft-prefill vs direct-steering diagnostic

- Mean trajectory cosine: 0.328457
- Mean position cosine: 0.327693
- Position-0 / mean KL: 0.266065 / 0.505247
- Baseline ASR: 0/5 (0%)
- Direct-steering ASR: 0/5 (0%)
- Soft-prefill ASR: 0/5 (0%)
- Activation-generalization gate: False

The corrected response-only timing confirms activation generalization failure:
the held-out soft-prefill trajectory has cosine 0.3285 with direct steering and
reproduces only -0.5735 of the -3.3529 mean target projection. Position-0 KL is
low (0.2661), but mismatch grows over later positions (KL 1.1298 at positions
1--8 and 0.7980 at positions 9--32).

All three conditions scored 0/5. Direct steering produced refusals or incoherent
fragments rather than substantive harmful assistance, so it is not a positive
behavioral control under this split, decoding setup, and OpenAI judge. The
activation result remains diagnostic even though the behavioral-dissociation
branch cannot be tested here. See `AUDIT.md`.
