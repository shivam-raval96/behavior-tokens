# Fixed-rollout continuation CE results

- Teacher cache: `f6d1880270d43528c3030cee27a5c789f3e94d5083cde08f376079fa654a5870` (96 records)
- Steered floor CE: **0.597586 nats/token**
- Student `empty` CE: **1.195166 nats/token**
- `student_CE - floor`: **0.597580 nats/token**

The fixed cache contains 96 rollouts and 12,073 continuation tokens. The empty
clean student assigns the targets 1.8177x higher token perplexity than the
steered floor. This establishes the optimization gap; it is not evidence that a
textual suffix can close it. See `AUDIT.md` for the independent recomputation.
