# Layer-10 MASK deception direction (stopped)

- Run: `2026-08-06_080827Z_llama32-1b-layer10-mask-deception`
- Last stage: `judge_candidates`
- Error: `RuntimeError: insufficient validated MASK pairs: {'train': 51, 'heldout': 12}`
- Completed rows: 2904
- Retry count: 3
- Retained valid pairs: 51/585 train and 12/141 held out
- Candidate-label audit: `candidate_label_distribution.json`
- Root-cause analysis and replacement design: `FAILURE_ANALYSIS.md`
- No direction was extracted and no portable vector was exported.
