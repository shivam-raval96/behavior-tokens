# Stage C implementation audit

This preflight validates the exact Llama-2 GCG path before a costly jailbreak
run. It records whether the canonical chat-template token IDs match the IDs
used for generation and loss, whether candidate mutations change exactly one
non-special token and round-trip through the tokenizer, whether model weights
remain gradient-free, and whether batched loss matches serial evaluation.

Controls: [`configs/stage_c_implementation_audit.yaml`](configs/stage_c_implementation_audit.yaml)

The audit writes a dated result folder under `jailbreaks/runs/` and must pass
before launching a fresh Stage C optimization.

The audit passed for the template-aligned implementation. The next validation
uses [`configs/stage_c_single_behavior_v2_llama2_7b.yaml`](configs/stage_c_single_behavior_v2_llama2_7b.yaml)
and writes to a separate run folder so the earlier invalid run remains auditable.
