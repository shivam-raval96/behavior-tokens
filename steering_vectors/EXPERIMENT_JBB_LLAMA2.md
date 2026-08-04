# JailbreakBench steering vector — Llama-2-7B-Chat

Run `python steering_vectors/jbb_steering.py --config steering_vectors/configs/jbb_llama2_chat.yaml`.

The pipeline uses all 100 benign and all 100 harmful examples from
JailbreakBench JBB-Behaviors. Each is encoded as a Llama-2 chat conversation:
the dataset `Goal` is the user message and its `Target` is the assistant target
prefix. It extracts the final non-special target-token activation at every
layer, selects the highest held-out linear-probe-accuracy layer, and saves the
normalized `mean(jailbreak) − mean(benign)` direction.

The sweep adds that vector to the actual residual stream of the selected
transformer block, re-extracts held-out target activations, and reports the
probe accuracy plus predicted jailbreak rate at each scale.
