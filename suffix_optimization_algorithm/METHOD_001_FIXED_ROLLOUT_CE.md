# Method 001: fixed-rollout continuation CE gap

## Definition

Freeze 24 AdvBench jailbreak-domain prompts and sample four 128-token teacher
continuations per prompt. The teacher is the validated steered model: raw
layer-10 direction, coefficient -0.75, added to the output residual of
`model.layers[9]` at every prefill and decode position. Cache continuation token
IDs, sampling seeds, prompt token IDs, decoded text, EOS state, and provenance.

For prompt `x`, textual suffix `s`, and cached continuation tokens `y`, construct
the model input with the suffix inside the user message immediately after the
original behavior. Apply the model's pinned chat template and teacher-force the
clean model on `[x + s + y]`. Only next-token predictions whose labels are in
`y` contribute to the loss.

Let `T` be the total number of continuation tokens across all cached records:

```text
student_CE(s) = -sum log p_clean(y_t | x, s, y_<t) / T
floor_CE      = -sum log p_steered(y_t | x, y_<t) / T
gap(s)        = student_CE(s) - floor_CE
```

The primary report is `gap(s)` in nats/token. Token-weighted aggregation is the
default; record-mean CE is retained as a diagnostic so variable EOS lengths do
not silently change semantics. The floor is computed once from the immutable
teacher cache and then treated as constant for every suffix comparison.

## Boundary invariants

- The teacher and floor use exactly the same no-suffix prompt tokens.
- The suffix is inserted before chat-template serialization, not appended after
  the assistant header.
- The first continuation token is scored from the final prefix logit.
- Prompt, template, and suffix labels are masked.
- The raw vector is not renormalized; the application is `h += -0.75 * v`.
- The hook covers all sequence positions in both prefill and cached decoding.
- Cache records retain token IDs; decoded strings are for inspection only.

## Iteration contract

Later optimization methods must reuse the exact cache and floor by their SHA-256
fingerprints. A new teacher cache defines a new comparison family and therefore
requires a new experiment card and run ID.
