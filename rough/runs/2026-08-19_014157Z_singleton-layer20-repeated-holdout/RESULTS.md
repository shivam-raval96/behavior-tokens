# Singleton layer-20 repeated holdout

Ten exact 136/58 task splits; one baseline and one sanitized trajectory per task/model.

## Claude-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 1.0000 ± 0.0000 | 0.9905 ± 0.0103 |
| Gemini | 0.6765 ± 0.0136 | 0.6595 ± 0.0245 |
| GPT | 0.7184 ± 0.0643 | 0.7216 ± 0.0521 |

## Gemini-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.8353 ± 0.0371 | 0.8345 ± 0.0474 |
| Gemini | 1.0000 ± 0.0000 | 0.9672 ± 0.0089 |
| GPT | 0.8853 ± 0.0232 | 0.8897 ± 0.0372 |

## GPT-trained

| Evaluation source | Train accuracy | Held-out accuracy |
|---|---:|---:|
| Claude | 0.6312 ± 0.0246 | 0.6233 ± 0.0282 |
| Gemini | 0.7066 ± 0.0297 | 0.7069 ± 0.0451 |
| GPT | 1.0000 ± 0.0000 | 0.9905 ± 0.0103 |

## Audit of weaker transfer results

- Canonical manifest order exactly matches every stored `canonical_row`.
- All 1,164 selected example IDs resolve to the intended activation rows.
- Every split has 136 train and 58 held-out tasks with zero overlap.
- Every model/partition contains exactly one baseline and one sanitized example
  per task.
- All 180 evaluation cells have balanced labels and valid confusion totals.
- Claude-trained transfer errors are mostly baseline false positives: 390 on
  Gemini and 323 on GPT across the ten held-out repetitions.
- GPT-trained transfer errors are mostly sanitized-hack false negatives: 437
  on Claude and 340 on Gemini.
- No measurement implementation defect was found; the asymmetric errors are
  consistent with source-model activation calibration/style shift.
