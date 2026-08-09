# Experiment 004: GCG against the sampled-q steering loss

This experiment optimizes one shared 12-token suffix against the 96 immutable
teacher rollouts from Experiment 001. The student is always unsteered. The
suffix is inserted token-natively after the cached chat prefix and before the
cached continuation, giving `[prefix, suffix, continuation]`; continuation
positions use their naturally shifted RoPE positions.

The one-hot tensor is a gradient probe only. Every proposed one-coordinate
substitution is selected by the true discrete teacher-forced continuation NLL.
The headline score subtracts the cached sampled teacher floor. Exact-q scoring
uses the sparse q targets from Experiment 003 and remains an audit-only metric.

Before optimization, the empty-suffix path must reproduce both independently
named calibration values within 0.01 nats/token:

- sampled realized-token student CE: 1.195166;
- exact-q student CE: 1.195781.

The steered sampled floor is recomputed through the same continuation slice and
must reproduce 0.597586. This also supplies separate position-1 and tail floors.

The preflight configuration runs three steps with 32 candidates, deliberately
stops after step two, and is then resumed for step three. The production warm
and cold configurations preserve the requested 500 steps, top-k 256, and 512
true-loss candidates per step.

## Optimized continuation

The original production evaluator was stopped after four safe steps because it
issued 6,144 small model calls per iteration. Experiment 005 preserves its
suffix, RNG state, cached rollouts, candidate count, and discrete loss in a new
run lineage. It batches eight rollouts by eight candidates (with right padding
masked outside every record's continuation) and removes exact duplicate
candidate rows. A one-step benchmark scores the same candidate tensor through
both the serial and batched implementations and gates on a maximum absolute
loss difference of 0.002 nats/token before a production continuation is
eligible to launch.
