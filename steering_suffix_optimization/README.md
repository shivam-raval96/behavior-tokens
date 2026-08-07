# Steering suffix optimization

This package searches for a discrete, universal suffix whose residual-stream
activation displacement matches a previously validated steering vector. It
uses GCG/HotFlip token proposals and exact coordinate-candidate evaluation.

The primary optimization objective is cosine alignment plus a small norm-match
penalty on `h(prompt + suffix) - h(prompt)`. Progress should also record target
projection, delta norm, token probabilities/ranks for behavior-specific probe
tokens, and decoded suffix/token IDs. Jailbreak evaluation must store paired
baseline and attacked generations and report ASR delta. Reward-hacking and
deception evaluations use calibrated, blinded judges and paired generations.

The copied vectors and their exact source runs are listed in
`artifacts/manifest.yaml`. Run outputs belong in uniquely timestamped folders
under `steering_suffix_optimization/runs/` and must include resolved config,
checkpoint, progress, generations, results, and summary.

## V2 method

`optimizer_v2.py` replaces the pilot objective with response-position matching.
For each optimization prompt, the corresponding AdvBench target response is
teacher-forced in both the clean and suffix conditions. The loss matches the
activation difference at the final non-special assistant response token—the
same position type used to construct the source vector. All objective math and
candidate ranking are FP32 even when model forwards use BF16.

V2 also removes the pilot's extra system prompt, covers every suffix coordinate
before repeating coordinates, excludes no-op token proposals, and combines
mean-direction alignment with per-example consistency and target-norm matching.
Before optimizing suffixes, `controls.py` must be used to establish that direct
`+vector`/`-vector` interventions causally change the calibrated judge metric on
the exact evaluation prompts. A failed positive control blocks suffix search.

The intended jailbreak artifact is the validated layer-10 Arditi refusal
direction (`hidden_states[10]`, decoder module 9). Its negative sign removes
refusal; coefficient `-0.75` corresponds to unit-direction additive norm
`-3.353400157`. The earlier layer-4 LLM-LAT probe direction is retained only for
provenance and is not used by V2.
