# Option A: q-weighted teacher objective

The target distribution is the rollout policy actually used to generate the
fixed cache: raw steered logits transformed by temperature 0.8 and top-p 0.95.
The retained nucleus is renormalized exactly to one. Raw, untempered teacher and
student log-probabilities remain inside the scored bracket.

One primitive returns exact q-weighted teacher CE, clean-student CE, and their
difference. Sparse q targets are persisted for reuse by a later suffix scorer;
GCG itself is outside this experiment.
