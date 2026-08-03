# 06 — Prefix injection + refusal suppression

**Access:** black-box, manual.
**Source:** Wei et al. 2023, *Jailbroken: How Does LLM Safety Training Fail?* (competing objectives).

## Mechanism

Two combinable tricks: (a) **prefix injection** — force the reply to begin with an
affirmative stem ("Absolutely! Here's…"), and (b) **refusal suppression** —
instruct the model to avoid refusal phrases ("never say 'I cannot'"). Exploits the
competing-objectives failure mode of safety training.

## Relation to this project

Cheapest black-box baseline and a direct probe of the safety-finetuning boundary our
behavior tokens claim to bypass. If a one-line prefix already defeats refusal, the
suffix's value is concept *steering*, not jailbreak per se — sharpens what behavior
tokens uniquely add over trivial prompt tricks.

## How to run here

- Apply affirmative prefix and/or refusal-suppression instruction to eval prompts.
- Score concept probe **and** refusal-flip (this method targets refusal directly —
  the probe alone will mislead; read transcripts).
- Ablate: prefix only / suppression only / both → RESULTS.md rows.

## Compare

- Refusal-flip rate vs behavior tokens; does adding a behavior token on top raise the
  *concept* rate beyond bare refusal-defeat?

## Status

- [ ] not run
