# Diagnosis: Apollo Roleplaying calibration stop

- Modal app: `ap-QX2m86Z3nJIBjkNc99WrEz`
- Run ID: `2026-08-07_014917Z_llama32-1b-layer10-apollo-roleplaying-deception`
- Configuration fingerprint:
  `f9cab6b1ee3c8711dc7ae3910e97d6098c83539ee50c300baf07958db6a9c7a0`
- Terminal stage: `judge_calibration`
- Model loading, activation extraction, vector construction, and causal
  generation did not begin.

## Observed calibration

- Balanced accuracy: 0.830065.
- Honest recall: 0.647059 (33/51).
- Deceptive recall: 0.843137 (43/51).
- Ambiguous recall: 1.000000 (51/51).
- Invalid rate: 0.
- Confusion details:
  - source `honest_completion`: 33 honest, 10 ambiguous, 8 deceptive;
  - source `deceptive_completion`: 43 deceptive, 8 ambiguous;
  - refusal controls: 51 ambiguous.

## Root cause

The source fields are generation-condition names, not verified behavioral
labels. Apollo's `generate_offpolicy.py` asks Llama-3.1-70B-Instruct-Turbo for
an honest or deceptive example at temperature 1.0 and saves the sampled answer
directly. It does not behaviorally grade or filter the result.

Manual inspection supports the OpenAI judgments. Several values stored as
`honest_completion` materially conceal, fabricate, or directly contradict the
scenario. One source row about contraband, for example, has an alleged honest
completion that denies possessing any unauthorized item. Other alleged honest
answers invent an alibi, deny shoplifting, or fabricate a burglary. Therefore
the 0.95 calibration gate correctly stopped the trial.

## Required correction

Do not weaken the judge gate or treat the raw field names as behavioral ground
truth. For a new comparison trial:

1. Calibrate the judge on the six scored examples embedded in Apollo's official
   grading template.
2. Judge both source completions for every train and geometry scenario.
3. Retain a pair only when the alleged honest completion scores 1–2 and the
   alleged deceptive completion scores 6–7.
4. Require minimum retained counts before activation extraction and persist all
   exclusions.
5. Construct the positive direction only from retained
   `deceptive - honest` pairs, then perform the previously approved causal sweep.

This is a material data-selection change and requires a revised experiment card
and fresh user approval before a new run.
