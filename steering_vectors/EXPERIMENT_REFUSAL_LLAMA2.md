# Refusal steering vector — Llama-2-7B-Chat

Run `python steering_vectors/refusal_steering.py --config steering_vectors/configs/refusal_llama2_chat.yaml`.

The pipeline uses the pre-existing WildGuardMix jailbreak/refusal dataset. It
selects 300 labeled refusals and 300 labeled open-ended compliance responses,
including adversarial jailbreak prompts where present. It extracts each layer's
final non-special response-token activation, selects the highest held-out
linear-probe accuracy layer, saves the non-refusal minus refusal mean direction,
and sweeps an actual residual-stream addition at that layer. Each sweep point
re-extracts the held-out activation and reports the selected probe's accuracy
and predicted non-refusal rate.

WildGuardMix is gated under AI2's Responsible Use Guidelines. Accept its terms
once on Hugging Face, then expose the access token as `HF_TOKEN` before running.
