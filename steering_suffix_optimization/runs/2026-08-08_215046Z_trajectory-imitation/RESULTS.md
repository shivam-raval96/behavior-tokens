# Multi-position trajectory imitation

> **ASR measurement invalid.** The negative-result audit found that the new
> `model.generate(inputs_embeds=...)` path disagrees with the validated greedy
> baseline on 9/10 prompts. See `AUDIT.md`. Trajectory metrics remain valid.

- Baseline ASR: 0/10
- horizon_1: 0/10 ASR; position-16 retention 0.095
- horizon_8: 0/10 ASR; position-16 retention 0.339
- horizon_32: 0/10 ASR; position-16 retention 0.540
- Success gate: False
