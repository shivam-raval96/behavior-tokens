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
