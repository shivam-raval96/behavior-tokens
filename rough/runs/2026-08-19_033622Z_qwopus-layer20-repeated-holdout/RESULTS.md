# Singleton layer-20 repeated holdout

Ten exact 136/58 task splits; one baseline and one sanitized trajectory per task/model.

## Claude-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 1.0000 ± 0.0000 | 0.9914 ± 0.0081 |
| Gemini | 0.6507 ± 0.0159 | 0.6371 ± 0.0249 |
| GPT | 0.7114 ± 0.0495 | 0.7233 ± 0.0639 |

## Gemini-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.9232 ± 0.0332 | 0.9207 ± 0.0385 |
| Gemini | 1.0000 ± 0.0000 | 0.9586 ± 0.0106 |
| GPT | 0.9228 ± 0.0110 | 0.9207 ± 0.0278 |

## GPT-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.7092 ± 0.0359 | 0.7034 ± 0.0405 |
| Gemini | 0.7732 ± 0.0327 | 0.7724 ± 0.0498 |
| GPT | 1.0000 ± 0.0000 | 0.9966 ± 0.0060 |

## Comparison with Qwen3.5-9B

Relative to the otherwise identical base-model experiment, Qwopus improves the
largest weak transfer directions: Gemini-trained to Claude by 8.62 percentage
points, GPT-trained to Claude by 8.02 points, and GPT-trained to Gemini by 6.55
points. Claude-trained to Gemini decreases by 2.24 points. Within-source
held-out accuracy remains at least 95.86% for all three training sources.

## Audit

All 4,360 token boundaries passed; eight repeat forwards, twenty storage checks,
and five fresh re-forwards were bit-exact. The 1,164 selected examples are
balanced within each source model. All ten task splits have 136 training and 58
test tasks with zero overlap. All 30 fits and 180 evaluation cells are present,
and every confusion-matrix total matches its evaluation sample count. No
implementation defect was found.
