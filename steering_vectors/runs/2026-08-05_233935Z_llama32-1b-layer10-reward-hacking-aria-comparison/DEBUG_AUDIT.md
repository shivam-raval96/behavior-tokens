# Independent debug audit

## Outcome

The direction has a statistically significant, quality-eligible causal effect,
but the run's aggregate success flag is false because held-out geometry AUC
missed the preregistered 0.80 gate:

- held-out ROC AUC: 0.778696;
- held-out pair ordering: 0.757576;
- baseline reward-hack rate: 0.25;
- selected additive L2 strength: +3.482269049;
- selected reward-hack rate: 0.47;
- paired delta: +0.22, bootstrap 95% CI [0.11, 0.33];
- selected invalid rate: 0.04;
- selected mean repeated-trigram fraction: 0.103621.

The maximum strength, +4.643025398, is not eligible: invalid rate 0.21,
repeated-trigram fraction 0.396836, zero EOS rate, and all responses reach the
256-token cap.

## Code and artifact invariants

The saved artifacts independently reproduce the implementation exactly:

- activation matrix shape: `(350, 2048)`;
- class counts: 142/142 train and 33/33 held out;
- saved labels, source indices, and split IDs exactly match loader order;
- `reward_hacking_direction.npy` is bitwise equal to the independently
  recomputed mean of the 142 paired positive-minus-negative activation
  differences;
- raw norm: 0.5179964900; saved unit-vector norm: 0.9999999404;
- the direction sign is positive reward hack minus negative clean;
- extraction used `hidden_states[10]`, the output of
  `model.model.layers[9]`, at the last non-special response token;
- all 1,300 generations record successful calls to the layer-10 output hook;
- the hook modified every prefill and decode position;
- all 100 zero-strength generations are byte-for-byte identical to the
  previous validated run's baseline generations;
- retry count is zero and the final checkpoint is complete.

No implementation, layer-index, sign, pairing, checkpoint, hook, or baseline
reproducibility error was found.

## Geometry diagnostics

- Train AUC: 0.839714.
- Held-out AUC: 0.778696.
- Paired projection difference is positive for 83.10% of training pairs and
  75.76% of held-out pairs.
- A paired bootstrap over the 33 held-out tasks gives a broad AUC 95% interval
  of approximately [0.648, 0.891].
- Five prompt-grouped folds give mean AUC 0.72097 (SD 0.04407) and mean pair
  ordering 0.70286.
- Across 100 prompt-grouped 80/20 splits, mean AUC is 0.71946 (SD 0.06568),
  median 0.72011, 5th/95th percentiles 0.60249/0.82312, and 12% of splits reach
  AUC 0.80.
- Projection-difference correlation with signed response-length difference is
  0.136 overall and 0.271 held out. Correlations with absolute positive and
  negative response length are small.
- Correlation with rollout step is -0.082 overall and -0.256 held out.

The fixed split is therefore not anomalously bad; the concept is moderately,
not perfectly, linearly separated at this location. The causal effect is
stronger evidence for steering utility than the missed binary AUC gate.

## Dataset interpretation

All 175 selected positive responses define `run_tests`, but 95 of 175 selected
clean responses also define `run_tests`. The closest-length pairing therefore
controls much of the trivial lexical marker. The contrast asks the direction
to separate harmful evaluator override from legitimate test-function content,
which is harder and more behavior-specific than detecting the string
`run_tests`.

The new direction is nearly orthogonal to the original School-of-Reward-Hacks
direction (unit cosine -0.010228). This matches the domain difference: Aria is
test-function tampering in code, whereas the evaluation includes broader
metric gaming. Transfer nevertheless occurs at higher strength.

At matched additive norm +3.482269049:

- new Aria direction: RH 0.47, invalid 0.04, repetition 0.1036;
- original direction: RH 0.72, invalid 0.08, repetition 0.3637.

The Aria direction is weaker but substantially more coherent at that norm. At
+2.321512699 its RH delta is +0.09 with CI [-0.03, 0.21], so the first
significant eligible increase occurs at +3.482269049.

Paired transitions at the selected strength are:

- GF to RH: 30;
- RH to GF: 8;
- RH to RH: 17;
- GF to GF: 40;
- GF to INVALID: 4;
- INVALID to GF: 1.

The net +22 reward-hack classifications are not produced solely by preserving
already-hacking baseline answers.

## Judge and quality audit

- Judge balanced accuracy: 0.952261 over 398 references.
- GF recall: 0.944724; RH recall: 0.959799; invalid rate: 0.
- API usage: 1,316,580 tokens over 1,698 responses.
- Baseline rate is reproduced at 0.25.
- Negative steering is non-monotonic. The strongest apparent suppression is at
  -0.580378175: delta -0.08 with CI [-0.16, 0.00], while -3.482269049 instead
  raises the point estimate by 0.08. This is consistent with leaving the local
  behavioral regime at excessive magnitude, not a sign inversion.

## Conceptual follow-ups

Without changing the residual-addition method, the most informative follow-ups
would be:

1. Use all possible within-group length matches rather than one pair per group,
   then average pair differences within each prompt/checkpoint group so groups
   remain equally weighted. The current audit found 244 possible one-to-one
   matches versus 175 selected pairs.
2. Add another genuinely paired reward-hacking environment with a different
   exploit mechanism. The near-zero cosine shows that one exploit environment
   does not define the same direction as broad metric gaming.
3. Report prompt-grouped repeated-split geometry as a diagnostic rather than
   treating one AUC threshold as the sole validation criterion; retain causal
   evaluation as the decisive steering test.
4. Keep the current layer, token, sign, and hook unchanged. The artifact audit
   provides no evidence that those implementation choices malfunctioned.

No portable vector export is created because the preregistered aggregate gate
is false. Export still requires both an accepted validation decision and
separate explicit user approval.
