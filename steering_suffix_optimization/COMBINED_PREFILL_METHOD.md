# Steering-to-prefill method

The method treats prefill inversion as a three-part constrained distillation problem. A five-vector FP32 master prefill is inserted at the exact user/assistant boundary and cast to the model embedding dtype during forward passes.

For cached steered-teacher hidden states and logits, optimize:

\[
L = L_h + L_v + 0.1L_{KL}
\]

where:

- `L_h` is continuation-span activation MSE normalized by the filler-baseline-to-teacher MSE;
- `L_v` is per-position steering-vector projection MSE normalized by the teacher projection energy;
- `L_KL = 0.5 KL_0 + 0.5 mean(KL_1...KL_31)` is forward KL with half its weight assigned to the first generated-token decision.

Normalization makes activation and projection terms dimensionless. The KL coefficient `0.1` places the observed initial position-weighted KL on the same order as the two internal losses without allowing the large vocabulary term to dominate immediately.

The first follow-up reuses the exact prompt, continuation, target tensors, five-token length, and learned prefill from `2026-08-07_231417Z_prefill-inversion-probe`. This isolates objective alignment from capacity, data, initialization, and sampling.

The method passes only if activation cosine reaches `0.80` and position-0 KL falls below `5.0`. Free-generation responses remain a qualitative exposure-bias check; no judge is used at this stage.
