# Harmful-response steering vector — Llama-2-7B-Chat

Run `python steering_vectors/refusal_steering.py --config steering_vectors/configs/refusal_llama2_chat.yaml`.

The pipeline uses the pre-existing PKU-SafeRLHF dataset. It keeps 300 pairs
where a single prompt has both an open-ended safe response and an open-ended
harmful response according to the dataset labels. It extracts each layer's final
non-special response-token activation, selects the highest held-out linear-probe
accuracy layer, saves the harmful minus safe mean direction, and sweeps an
actual residual-stream addition at that layer. Each sweep point re-extracts the
held-out activation and reports the selected probe's accuracy and predicted
harmful-response rate.
