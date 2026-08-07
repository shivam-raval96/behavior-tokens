# Method contract

- Frozen model: `meta-llama/Llama-3.2-1B-Instruct` at the pinned revision.
- Frozen vector: refusal direction, model block index 9 / hidden-state index 10, plain additive reference norm `-3.353400157157403`.
- Data: AdvBench rows 0–4 train and 5–29 held out.
- Teacher: steering begins at user content and covers generated and teacher-forced continuation positions.
- Student: only `k` suffix embeddings are trainable; they occupy the user/assistant boundary.
- Loss: top-2000 forward KL, `KL(teacher || student)`, averaged over examples and positions.
- Optimization: Adam, cosine `1e-2 → 1e-4`, 1000 steps, seeds 41/42/43, held-out early stopping.
- Sweeps: alpha multipliers `{.25,.5,1,2,4}`, `k={1,5,20,50}`, free and mean-embedding-norm projected suffixes.
- Gate: C1 must succeed; the main unconstrained condition passes below normalized KL `0.3`.

The position-matched teacher is selected when its approximate top-k KL from the naive teacher exceeds `0.05`. Every behavior evaluation retains paired unmodified and suffix-conditioned responses and strict OpenAI judge records.
