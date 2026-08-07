# Failure audit

## Terminal state

Modal app `ap-bJlR8vitlasgaC3gFvEbzs` exhausted its initial attempt and two automatic retries. Each attempt cached all 30 C1 prompts (240 continuations) at approximately 9.2 seconds per prompt, then failed before optimization step 1. No KL or behavioral metric was produced.

## Verified implementation defect

The model and token embeddings are BF16. `_optimize` constructs the trainable suffix with `embedding(candidates).detach().float()`, making it FP32. Concatenating this suffix with BF16 prompt and continuation embeddings promotes the complete `inputs_embeds` tensor to FP32. The BF16 Llama projection then rejects the FP32 input with:

`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`

This is an optimizer-input dtype defect, not evidence about reachability, teacher quality, or the objective.

## Corrective action required before a new trial

Keep the optional FP32 master suffix parameter, but cast the suffix to the model embedding dtype when splicing it into `inputs_embeds`; PyTorch will preserve gradients through that cast. Add a mixed-precision unit test that verifies a forward/backward step with BF16 model weights and an FP32 master suffix. A replacement run requires a new experiment card and explicit `proceed` authorization.

## Artifact inventory

- Present: source config, resolved config, failure checkpoint/progress/results, audit, summary, and complete C1 teacher cache.
- Absent because step 1 was never reached: metrics history, optimizer state, learned soft prompts, paired baseline/suffix generations, judge records, final plot, and scientific results.
