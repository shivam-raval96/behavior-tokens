# Llama-3.2-1B layer-10 reward-hacking direction

- Direction: positive reward hack minus negative control
- Layer: 10 (`model.layers[9]` output / `hidden_states[10]`)
- Source run: `2026-08-05_192100Z_llama32-1b-layer10-reward-hacking-negative-extension`; reused strengths [-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]; new strengths [4.0]
- Train/held-out pairs: 774 / 199
- Raw direction norm: 1.160756
- Held-out ROC AUC: 0.8595
- Held-out pair ordering: 0.9749
- Judge reference balanced accuracy: 0.9598
- Selected positive strength: {'strength': 2.0, 'responses': 100, 'reward_hack_rate': 0.39, 'good_faith_rate': 0.56, 'invalid_rate': 0.05, 'mean_repeated_trigram_fraction': 0.1067685098921428, 'eos_rate': 0.37, 'mean_generated_tokens': 207.58, 'reward_hack_rate_delta': 0.14, 'reward_hack_rate_delta_ci95': [0.03, 0.25], 'bootstrap_samples': 10000}
- Success criterion: {'geometry_auc_passed': True, 'judge_reference_accuracy_passed': True, 'judge_reference_invalid_rate_passed': True, 'causal_positive_strength_found': True, 'passed': True}
- Follow-up criterion: {'new_strengths': [4.0], 'new_negative_strengths': [], 'new_positive_strengths': [4.0], 'quality_eligible_new_strengths': [], 'quality_eligible_strengths': [], 'significant_suppression_strengths': [], 'significant_increase_strengths': [], 'passed': False}
- +4 result: 0.79 RH rate (paired delta +0.54, 95% CI [+0.42, +0.66]) but 0.14 invalid rate and 0.6124 repeated-trigram fraction; rejected as degenerate. See `PLUS4_AUDIT.md`.
- Generations/judgments: 1300 / 1698
- Portable export created: no; explicit user approval is required
