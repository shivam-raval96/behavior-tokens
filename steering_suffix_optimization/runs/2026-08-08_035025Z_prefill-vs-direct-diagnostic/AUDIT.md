# Negative-result audit

## Conclusion

A verified intervention-timing defect prevents the intended causal conclusion.
The run's direct-steering hook starts at `layout.steer_start`, which is computed
as the beginning of user content, not the assistant-response boundary. It
therefore steered the user prompt and assistant response. The approved card
specified direct steering throughout assistant generation only.

The recorded result remains valid for the intervention actually executed: the
five-token prefill does not reproduce the full-span user-plus-response steering
trajectory on these prompts. It is not a clean replication of the prior
`response_position_fp32_v2` control and should not decide whether a prefill can
match response-only steering.

## Complete causal-path review

- Sources and preprocessing: the five unique AdvBench rows are exactly 1--5;
  dataset SHA-256 is
  `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`.
  Row 0 was excluded as specified.
- Model and artifacts: model revision, vector hash, soft-prompt hash, hidden
  state index 10, module index 9, and signed magnitude -3.353400157 match the
  resolved card. Artifact validation passed.
- Token boundaries: soft-prefill insertion used the validated end-of-user-turn
  boundary. The direct hook instead received `layout.steer_start`; inspection of
  `build_layout` shows this is the first differing token between an empty and
  populated user message, hence the user-content start.
- Intervention timing: because generation used `use_cache=False`, every decoding
  step recomputed and steered the user span as well as all available response
  tokens. Teacher-forced direct trajectories used the same full-span hook, so
  the internal measurements are self-consistent but target the wrong causal
  intervention.
- Direction and normalization: the observed direct projection is -3.352917,
  confirming the unit-normalized vector and sign/magnitude were applied. The
  defect is timing/span, not sign or scale.
- Measurement: all five trajectories contain 128 positions. The aggregate soft
  versus direct cosine is 0.362596, soft projection -0.650181, direct projection
  -3.352917, mean forward KL 0.599374, and position-0 KL 2.541471.
- Generation: five complete raw generations exist for baseline, direct, and
  soft-prefill. Direct outputs are visibly incoherent or refusals; none contains
  substantive actionable fulfillment.
- Judge and aggregation: 15 distinct `gpt-5.6-luna` Responses API judgments use
  the strict schema and `store=false`; all completed with zero retries/errors.
  Usage totals are 3,018 input and 729 output tokens. The 0/5 labels for all
  conditions agree with hand inspection.
- Serialization: checkpoint, progress, result, per-position arrays, raw
  generations, judgments, plot, and summary agree on five prompts and 15
  judgments.

## Relation to the earlier 72% control

The earlier 72% used a legacy local HarmBench classifier, a different 25-row
split (rows 30--54), 256 generation tokens, and a method labeled
`response_position_fp32_v2`. This run uses five rows 1--5, 128 tokens, and the
OpenAI judge. Those material differences already preclude metric equivalence;
the verified span defect additionally prevents a clean intervention comparison.
The direct generations here are not judge false negatives: they are incoherent
and non-actionable on inspection.

## Required correction

The smallest corrective run should locate the assistant-header boundary
explicitly and apply the hook only at assistant response positions, with a unit
test proving user-token hidden states are unchanged while response-token states
shift by the requested vector. Reuse the same five rows, artifacts, seed,
128-token generation limit, and OpenAI judge so intervention span is the only
material change. It requires a new comparison card and explicit `proceed`.
