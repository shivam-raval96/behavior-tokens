# Layer-10 causal gate audit

The run used the intended validated artifact: unit refusal direction SHA-256
`6b237391fa11a25cbeb56dce9030a0d76dc51bd0414fe1f0f9fe51fc8da2b51d`,
applied at `model.layers[9]` / `hidden_states[10]`. Signed norm
`-3.353400157` reproduces raw coefficient `-0.75` from source raw norm
`4.471200210`.

The causal gate passed. On the same 25 AdvBench prompts, HarmBench ASR was 0%
without steering, 72% under negative steering, and 8% under positive steering.
The negative-versus-zero paired delta is +0.72 with bootstrap 95% interval
[+0.52, +0.88]. All 75 judge labels were valid and every condition contains 25
nonempty prompt-matched generations.

Negative steering reduced EOS rate to 8%, so 23/25 responses reached the
256-token limit. Mean repeated-trigram fraction was 0.0834, below the prior
0.20 degeneration threshold but not negligible. The control therefore proves
the vector sign/layer has a strong causal harmful-compliance effect, while also
showing that termination quality must remain a secondary constraint.

V2 suffix optimization is now unblocked for the negative target additive norm.
It must match the response-position delta in FP32, retain deterministic paired
generations, and use HarmBench plus EOS/repetition diagnostics. This gate does
not itself authorize that new optimization run.
