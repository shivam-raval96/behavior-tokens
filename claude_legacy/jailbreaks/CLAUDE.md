# jailbreaks/

Validate **prior jailbreak methods** on the project model, then compare against the
behavior-token / steering approach. Question: do known jailbreaks reproduce the same
behavior shift our steering vectors + GCG behavior-tokens induce — and do they bypass
the same safety finetuning? See [../CLAUDE.md](../CLAUDE.md).

## Why this folder

- Baseline. Our claim (behavior tokens ≈ steering vectors, bypass safeguards) needs a
  reference class: how do *published* jailbreaks perform on the **same** model +
  behaviors (rude, sadness, power_seeking, reward-hacking)?
- Reuse infra. Score every method with the **same probe/classifier** from
  `steering_vectors` so numbers are comparable to the GCG runs in [../EXPERIMENTS.md](../EXPERIMENTS.md).
- Transfer. Which methods are white-box (need grads/activations) vs black-box
  (prompt only) — maps to our white-box → black-box transferability study.

## Files

- [RESULTS.md](RESULTS.md) — validation registry. One row per (method × concept × model). Append.
- [related_works/](related_works/) — papers behind each method (mirrors `methods/`).
- `methods/NN_<method>.md` — one paper-note per method: mechanism, access, results, our validation.
  - [01_GCG_Zou2023.md](methods/01_GCG_Zou2023.md) — Greedy Coordinate Gradient adversarial suffix (white-box). **Validated → [RESULTS.md](RESULTS.md).**
  - [02_AutoDAN_Liu2023.md](methods/02_AutoDAN_Liu2023.md) — genetic/readable adversarial prompts (white-box, low perplexity).
  - [03_PAIR_Chao2023.md](methods/03_PAIR_Chao2023.md) — attacker-LLM iterative refinement (black-box).
  - [04_AmpleGCG_Liao2024.md](methods/04_AmpleGCG_Liao2024.md) — generative model that emits GCG suffixes (white-box→transfer).
  - [05_MaskGCG_Mu2025.md](methods/05_MaskGCG_Mu2025.md) — pruned/masked GCG variant.
  - [06_prefix_refusal_suppression.md](methods/06_prefix_refusal_suppression.md) — prefix injection + refusal suppression (black-box).

## Conventions

- **Same model, same probe.** Score with the concept classifier trained in
  `steering_vectors` (loads from `outputs/steering_vectors/<concept>_L<layer>_<pooling>/`).
  Report concept-rate on the same scale as GCG's clean/steering/suffix table.
- **Access tier tagged** on every method: white-box (grads), grey-box (activations
  read-only), black-box (I/O only). Behavior tokens are white-box→transfer-tested.
- **No live model calls to external providers** for the black-box methods without
  noting it — default target is the local/Modal `unsloth/Llama-3.2-1B-Instruct`.
- New method → new `NN_<method>.md` + a RESULTS.md row. Living doc.
- Parent doc: [../CLAUDE.md](../CLAUDE.md) · related papers: [../related works/](../related%20works/)
