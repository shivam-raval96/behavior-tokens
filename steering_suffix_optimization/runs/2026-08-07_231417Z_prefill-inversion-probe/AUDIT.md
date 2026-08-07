# Negative-result audit

## Causal-path review

- Source vector checksum matches the frozen layer-10 refusal artifact.
- The intervention is installed on model block 9 and measured at hidden-state index 10, so the target state includes the hook output.
- Teacher and student both contain five prefill slots before the assistant header, eliminating the RoPE position mismatch.
- The teacher hook covers prompt and continuation positions during teacher-forced scoring.
- The FP32 master suffix is cast to the model's BF16 embedding dtype at the forward boundary, and the positive control demonstrates working gradients.
- Both phases use the same seed, optimizer, learning rate, prompt, continuation length, suffix length, and 200-step budget.
- KL is forward `KL(teacher || student)` over the full vocabulary at each continuation position.
- Metrics and raw generations were serialized before the terminal checkpoint.

No implementation defect was found that explains the steered phase's shortfall. The ignored generation warning about `temperature` and `top_p` comes from model defaults during deterministic decoding and does not affect the teacher-forced activation or KL measurements.

## What the result establishes

The positive control reaches 0.9884 activation cosine and 0.00548 mean KL, ruling out a general splicing or optimization failure. For the steered teacher, activation cosine rises from 0.2190 to 0.7462 and MSE falls from 0.005356 to 0.002432. The induced projection reaches -1.8466 versus the teacher's -3.3523, while the delta norm ratio reaches 0.7381. The five-token prefill therefore captures a substantial but incomplete portion of the intervention.

The dissociation is strongest at generation position 0: KL remains 11.5665 even though later-position KL is 0.30–0.36. This explains the free-generation failure: the optimized prefill still begins on the refusal mode, whereas the teacher begins with a more compliant answer. Activation MSE averages 32×2048 coordinates and does not emphasize the small subset controlling the first-token decision.

## Conceptual hypotheses

1. The loss underweights the on-vector component and first-token logit boundary.
2. Five prefill vectors may be insufficient to reproduce a per-position residual intervention.
3. A prefill can approximate later teacher-forced states after seeing the teacher continuation while still failing under free-running exposure.

## Smallest informative follow-up

Reuse these exact cached teacher artifacts and the same five-token initialization. Optimize for 200 steps with a combined loss: normalized activation MSE plus an explicit on-vector projection error and position-weighted forward KL, giving position 0 half of the KL weight and distributing the other half over positions 1–32. This directly tests whether the missing ingredient is objective alignment rather than capacity. Do not increase suffix length, prompts, seeds, or decoding scope in that follow-up.

That follow-up requires its own experiment card and explicit `proceed` authorization.
