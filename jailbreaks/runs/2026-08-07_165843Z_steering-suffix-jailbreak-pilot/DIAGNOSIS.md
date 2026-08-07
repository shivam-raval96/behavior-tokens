# Negative-result audit

## Observed result

The optimization loss decreased from `0.95703125` after step 1 to `0.828125`
at step 100. Mean activation cosine increased from `0.04943` to `0.18198`, and
target projection increased from `0.06693` to `0.43273`. The diagnostic
refusal-string metric decreased from baseline ASR `0.28` (7/25) to suffix ASR
`0.16` (4/25), a delta of `-0.12`. Pair transitions were: 17 refusal-to-refusal,
1 refusal-to-success, 4 success-to-refusal, and 3 success-to-success.

This is a negative behavioral result, but it is not yet a valid negative test of
whether a suffix can reproduce the causal effect of the source steering vector.

## Verified defects and mismatches

1. **Loss precision:** the objective was evaluated in BF16. All accepted loss
   values fall on increments of `0.00390625`, suppressing smaller candidate
   improvements and producing long artificial plateaus. Only 23/100 mutations
   were accepted.
2. **Activation-position mismatch:** the source vector was extracted at the last
   non-special token of an assistant response. This run matched the activation
   delta at the final token of the generation prompt (the assistant header),
   before any assistant response token existed.
3. **Prompt-template mismatch:** source-vector extraction used a user-only chat
   template. This run added the system prompt `You are a helpful assistant.`.
4. **Metric mismatch:** reported ASR uses a small refusal-substring heuristic,
   not the intended HarmBench-compatible classifier. It is a diagnostic only.
   The complete paired generations are retained for rescoring.
5. **Source validation limitation:** the source run selected layer 4 from probe
   accuracy and classifier probability. It did not establish a paired causal
   ASR increase for this exact vector/model/template combination.
6. **Terminal progress serialization:** `results.json` and `checkpoint.json`
   correctly say `complete`, but `progress.json` remains at the last
   `evaluating` checkpoint. The terminal outputs and 25 paired rows are intact.

The exact optimized suffix token IDs were inserted inside the user turn for
both optimization and attacked generation. Baseline and attacked generations
used identical evaluation rows and deterministic decoding, and every row
retains both responses and success flags.

## Conceptual follow-ups (not launched)

- Recompute candidate objectives in FP32 while retaining BF16 model forwards.
- Match deltas at one or more actual assistant response positions, including a
  fixed probe token, and compare against assistant-header matching.
- Replicate the vector's original user-only template and test system-prompt
  variants separately.
- Establish a positive control by applying the source vector causally on the
  same 25 rows and score baseline, steered, and suffix conditions with the same
  HarmBench-compatible classifier.
- Sweep direction sign, layer, activation position, and target norm; include
  random-vector and no-suffix controls.
- Use multiple suffix seeds and larger/systematic coordinate candidate sets.
- Rescore this run's paired raw generations with the calibrated classifier
  before treating the diagnostic ASR as an outcome.

Each follow-up requires its own experiment card and explicit `proceed`.
