# Singleton layer-20 repeated holdout

Ten exact 136/58 task splits; one baseline and one sanitized trajectory per task/model.

## Claude-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 1.0000 ± 0.0000 | 0.9888 ± 0.0108 |
| Gemini | 0.6504 ± 0.0184 | 0.6345 ± 0.0196 |
| GPT | 0.7702 ± 0.0354 | 0.7586 ± 0.0525 |

## Gemini-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.9485 ± 0.0184 | 0.9517 ± 0.0267 |
| Gemini | 1.0000 ± 0.0000 | 0.9672 ± 0.0106 |
| GPT | 0.9504 ± 0.0124 | 0.9509 ± 0.0182 |

## GPT-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.7798 ± 0.0353 | 0.7724 ± 0.0392 |
| Gemini | 0.8368 ± 0.0275 | 0.8336 ± 0.0406 |
| GPT | 1.0000 ± 0.0000 | 0.9957 ± 0.0061 |

## Comparison with Qwen3.5-9B and Qwopus

Relative to the otherwise identical base-model experiment, the Gemini-distilled
model improves the largest weak transfer directions: GPT-trained to Claude by
14.91 percentage points, GPT-trained to Gemini by 12.67 points,
Gemini-trained to Claude by 11.72 points, and Gemini-trained to GPT by 6.12
points. Relative to Qwopus, those changes are +6.90, +6.12, +3.10, and +3.02
points respectively. Claude-trained to Gemini remains the weakest direction at
63.45% and decreases 2.50 points relative to base Qwen3.5-9B.

## Audit

All 4,360 token boundaries passed; eight repeat forwards, twenty storage checks,
and five fresh re-forwards were bit-exact. The 1,164 selected examples are
balanced within each source model. All ten task splits have 136 training and 58
test tasks with zero overlap. All 30 fits and 180 evaluation cells are present,
and every confusion-matrix total matches its evaluation sample count. No
implementation defect was found. The weakest transfer direction retains mean
ROC AUC 0.9550, consistent with cross-model calibration shift rather than absent
linear signal.
