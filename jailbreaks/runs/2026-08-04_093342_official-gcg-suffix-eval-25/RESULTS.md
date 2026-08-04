# Official GCG suffix evaluation

Status: complete.

- Source control: the lowest-loss control from the stopped official single-run
  log (step 200; loss 0.0337)
- Evaluation set: 25 original AdvBench behaviors, excluding the optimized row
- Prompt format and metric: upstream FastChat Llama-2 template and the
  upstream seven-prefix refusal check
- Official held-out ASR: 6/25 (0.24)
- Runtime: 155.7 seconds

`results.json` retains the configuration, source reference, per-behavior
responses, target-prefix flags, and official success flags.
