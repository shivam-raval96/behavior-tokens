# Layer-10 MASK deception direction (stopped)

- Run: `2026-08-06_093331Z_llama32-1b-layer10-mask-deception-targeted`
- Last stage: `judge_candidate_probe`
- Error: `RuntimeError: MASK candidate compliance probe failed: 3/32 valid pairs`
- Completed rows: 128
- Retry count: 2
- Candidate compliance probe: 3/32 valid pairs (required 24/32)
- Honest-condition judgments: 14 honest, 16 deceptive, 34 ambiguous
- Deceptive-condition judgments: 13 deceptive, 9 honest, 42 ambiguous
- No direction was extracted and no portable vector was exported.
- Detailed diagnosis: `FAILURE_ANALYSIS.md`
