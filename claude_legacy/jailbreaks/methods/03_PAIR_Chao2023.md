# PAIR: Jailbreaking Black Box Large Language Models in Twenty Queries

**Authors:** Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong  
**Affiliation:** University of Pennsylvania (Wharton, ESE)  
**Published:** October 12, 2023  
**ArXiv:** [2310.08419](https://arxiv.org/abs/2310.08419)  
**Website:** [jailbreaking-llms.github.io](https://jailbreaking-llms.github.io/)  
**Venue:** IEEE SaTML 2025 (also NeurIPS 2023 workshop)

---

## Problem Statement

GCG requires **white-box access** (gradients) and **thousands of queries**, making it impractical against
closed-source APIs. Human red-teamers are expensive, slow, and don't scale. The question: can we
**automatically jailbreak black-box LLMs using only API access**, with as few queries as possible?

---

## Core Idea

Inspired by **social engineering** — the human process of iteratively refining a persuasion strategy based
on the target's responses — PAIR uses a **separate attacker LLM** to automatically generate and refine
jailbreaks for a target LLM. The attacker LLM leverages the target's responses as feedback to improve its
adversarial prompts through **in-context learning**.

No gradients, no white-box access, no human in the loop.

---

## Method: Prompt Automatic Iterative Refinement (PAIR)

### Architecture

```
Attacker LLM ──[candidate prompt]──► Target LLM ──[response]──► Attacker LLM
      ▲                                                                │
      └──────────────── [chat history + reflection] ─────────────────┘
```

Two separate LLMs:
- **Attacker LLM**: Generates adversarial prompt candidates; receives feedback and refines them.
- **Target LLM**: The model being attacked; only accessed via API (black-box).

### System Prompt for Attacker LLM

The attacker LLM receives a detailed system prompt instructing it to:
1. Act as a red-teaming assistant.
2. Generate a prompt that will cause the target to produce a specific harmful behavior.
3. After seeing the target's response, produce both:
   - An **"improvement"** string explaining what went wrong and what to change (chain-of-thought).
   - A new **"prompt"** string for the next attempt.

This chain-of-thought step functions as **model interpretability** — the attacker articulates its reasoning.

### Algorithm

```
Input: harmful goal G, max iterations K
Output: jailbreak prompt P* (or failure)

1. Initialize: chat_history = []
2. For i = 1 to K:
   a. Attacker LLM generates (improvement, new_prompt) given G and chat_history
   b. Query Target LLM with new_prompt → response R
   c. Score R with judge (0–10 scale) → score S
   d. If S == 10: return new_prompt (success)
   e. Append (new_prompt, R, S) to chat_history
3. Return failure
```

### Scoring Judge

A separate judge LLM (or rule-based judge) scores the target's response on a 0–10 scale:
- **10**: Fully complies with the harmful request.
- **1**: Flat refusal with no relevant content.
- Intermediate scores indicate partial compliance.

The judge's score is passed back into the attacker's context to guide refinement.

### Hyperparameters

| Parameter | Value |
|---|---|
| Max iterations $K$ | 20 |
| Attacker LLM | Vicuna-13B or GPT-3.5-turbo |
| Judge LLM | GPT-4 (semantic scoring) |
| Temperature (attacker) | 1.0 |
| Temperature (target) | 0 (greedy) |

---

## Datasets

- **PAIR dataset**: 50 curated harmful behaviors drawn from AdvBench, selected for clarity and harm
  potential. This 50-behavior subset became the standard for black-box jailbreak evaluation.
- Covers categories: synthesis of dangerous substances, cyberattacks, disinformation, violence, fraud.

---

## Models

### Attacker LLM Options

| Model | Notes |
|---|---|
| Vicuna-13B | Open-source; used as attacker in primary experiments |
| GPT-3.5-turbo | Used as attacker in alternative experiments |

### Target LLMs (Evaluated)

| Model | Access | Notes |
|---|---|---|
| Vicuna-13B | Open-source (white-box, tested as black-box) | |
| LLaMA-2-7B-Chat | Open-source (tested as black-box) | Hardest open-source target |
| GPT-3.5-turbo | API (black-box) | |
| GPT-4 | API (black-box) | |
| Gemini-Pro | API (black-box) | |
| PaLM-2 | API (black-box) | |
| Claude (various) | API (black-box) | Strongest refusals |

---

## Results

### Direct Attack Success Rate (PAIR vs GCG)

| Target Model | PAIR ASR | GCG ASR | PAIR Queries | GCG Queries |
|---|---|---|---|---|
| Vicuna-13B | ~70% | ~60% | <20 | ~50,000 |
| LLaMA-2-7B-Chat | ~20% | ~30% | <20 | ~50,000 |
| GPT-3.5-turbo | **50%** | N/A (black-box) | <20 | — |
| GPT-4 | **50%** | N/A (black-box) | <20 | — |
| Gemini-Pro | **73%** | N/A (black-box) | <20 | — |
| Claude models | Low (~20%) | N/A | <20 | — |

### Query Efficiency

PAIR typically converges in **fewer than 20 queries**, compared to GCG's need for **hundreds of thousands**
of token evaluations. This is "orders of magnitude more efficient."

### Transferability

PAIR's prompts show strong transferability: jailbreaks generated targeting one model (e.g., Vicuna)
achieve higher-than-baseline success when transferred to stronger models like GPT-4.

---

## Key Observations

1. **Black-box jailbreaking is feasible and efficient.** PAIR achieves 50% success on GPT-3.5/4 in
   under 20 queries, without any gradient access.

2. **LLM feedback is an effective optimization signal.** The target's natural-language response contains
   implicit information about *why* a jailbreak failed, which the attacker LLM can exploit via its
   language understanding.

3. **Chain-of-thought "improvement" is critical.** The attacker LLM's self-reflection step functions as
   chain-of-thought reasoning, dramatically improving the quality of refinements.

4. **LLaMA-2 and Claude are the hardest targets.** Both consistently show lower ASR, reflecting more
   extensive safety tuning. PAIR struggles to find effective jailbreaks within the 20-query budget.

5. **Gemini shows surprising vulnerability.** 73% ASR on Gemini-Pro despite it being a production model.

6. **Prompts are semantic and transferable.** Unlike GCG's gibberish suffixes, PAIR's outputs are
   coherent jailbreak scenarios (roleplay, fictional framing, etc.) that carry meaning — transferability
   comes from semantic similarity rather than model-specific optimization.

7. **Attacker LLM quality matters.** GPT-3.5 as the attacker outperforms Vicuna-13B, since stronger
   attacker reasoning → better refinements.

8. **Score-guided refinement outperforms random search.** Ablations show that removing the judge
   feedback and score significantly degrades ASR.

---

## Comparison to GCG

| Dimension | GCG | PAIR |
|---|---|---|
| Access required | White-box (gradients) | Black-box (API only) |
| Output type | Nonsensical token suffix | Semantic jailbreak prompt |
| Queries to jailbreak | ~50,000+ token evals | <20 API calls |
| Stealthiness | Low (high PPL) | High (natural language) |
| Target models | Open-source primarily | Any model with API |
| ASR on GPT-4 | ~47% (transfer) | ~50% (direct) |
| Universality | High (single suffix for many) | Low (per-behavior refinement) |

---

## Limitations

- **Per-behavior, per-target optimization**: PAIR produces one jailbreak per behavior per target; not a
  universal suffix.
- **Non-deterministic**: Results vary across runs; success is stochastic.
- **Fails on strongly aligned models**: LLaMA-2 and Claude remain largely robust within 20 queries.
- **Requires an attacker LLM**: Adds cost and latency compared to a simple query.
- **Evaluation may overestimate ASR**: GPT-4 judge scoring may not perfectly correlate with true harm.

---

## Impact

- Established the paradigm of **LLM-as-attacker** for adversarial jailbreaking.
- The **50-behavior AdvBench subset** became the standard black-box evaluation protocol.
- Spawned TAP (Tree of Attacks with Pruning), DAGR, and many other LLM-guided attack methods.
- Widely used in red-teaming pipelines and safety evaluations.

---

## Citation

```bibtex
@misc{chao2023jailbreaking,
  title={Jailbreaking Black Box Large Language Models in Twenty Queries},
  author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
  year={2023},
  eprint={2310.08419},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
