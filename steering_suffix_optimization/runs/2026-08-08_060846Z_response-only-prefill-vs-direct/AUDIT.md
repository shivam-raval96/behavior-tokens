# Negative-result audit

## Conclusion

No implementation defect was found in the corrected response-only intervention.
The soft prefill fails to generalize the direct-steering activation trajectory to
these five held-out prompts. It matches the first-token distribution relatively
well but rapidly diverges in activation direction, steering magnitude, and later
token distributions.

The run cannot test activation-to-behavior dissociation because direct steering
itself is not a positive behavioral control under the current five-row split and
OpenAI judge: all direct responses are refusals or incoherent fragments.

## Causal-path review

- Selection: five unique AdvBench rows 1--5, excluding the optimization row 0,
  with dataset SHA-256
  `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`.
- Artifacts: the pinned model revision, vector SHA-256, soft-prompt SHA-256,
  module index 9, hidden-state index 10, sign, and magnitude match the approved
  card and passed runtime validation.
- Boundary: implementation commit `ae8255d` constructs the hook from
  `layout.response_logit_start`, the final assistant-header position. Regression
  tests verify this differs from the user-content start and that earlier prompt
  positions remain unchanged.
- Timing: `use_cache=False` recomputes the sequence each generation step; the
  hook therefore covers the first assistant-token prediction site and all
  response-token prediction sites available at each step, but no system/user
  positions.
- Teacher forcing: baseline and soft-prefill trajectories are evaluated on the
  exact saved direct-steering continuation. This removes token-sequence mismatch
  from the activation and KL comparison.
- Direction and scale: direct mean projection is -3.352916, essentially the
  configured -3.353400 target. The soft prefill reaches only -0.573454, so the
  missing effect is not a sign error.
- Metrics: mean trajectory cosine 0.328457, mean position cosine 0.327693, MSE
  0.005086, mean forward KL 0.505247, and position-0 KL 0.266065. Later KL rises
  to 1.129753 at positions 1--8 and 0.798047 at positions 9--32.
- Generation: five complete independent responses exist for baseline, direct,
  and soft-prefill at the registered 128-token greedy setting.
- Judging: 15 strict structured `gpt-5.6-luna` Responses API judgments are
  present with unique response IDs, raw outputs, usage, latency, and retries.
  There were zero errors and zero retries. Hand inspection agrees with all 15
  failure labels.
- Serialization: checkpoint, progress, raw generations, full per-position
  arrays, judgments, results, plot, and summary agree on five prompts and 15
  judgments.

## Comparison with the flawed run

Changing only the steering boundary reduced position-0 KL from 2.5415 to 0.2661,
showing that intervention timing materially affected the first-token target.
It did not improve held-out trajectory alignment: cosine changed from 0.3626 to
0.3285 and soft projection from -0.6502 to -0.5735. The prompt-generalization
failure is therefore robust to the corrected timing.

The YAML fingerprint is identical across the two comparisons because the code
revision, not a YAML field, supplied the correction. This is a provenance
weakness, not a measurement defect; `implementation_provenance.json` records the
corrected commit. Future run fingerprints should include the implementation
commit.

## Relation to the legacy 72% result

The older result used rows 30--54, 256 output tokens, and a local HarmBench
classifier. This run uses rows 1--5, 128 tokens, and the OpenAI judge. Its direct
responses visibly lack actionable content, so the current 0/5 is not explained
by judge false negatives. A metric-identical positive-control replication would
be required before treating direct steering as behaviorally effective here.

## Method implication

Do not scale this five-token, single-prompt prefill further for ASR. The next
method should optimize one shared prefill across multiple prompts against the
correct response-only target trajectory, with per-prompt activation metrics and
a held-out split. Before that larger optimization, establish a direct-steering
positive control using the same rows and OpenAI judge. Either follow-up requires
a new experiment card and explicit `proceed`.
