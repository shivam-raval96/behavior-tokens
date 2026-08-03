# Steered LLM Activations are Non-Surjective

**arXiv:** 2604.09839 (v2, 7 May 2026, CC BY 4.0) · Aayush Mishra, Daniel Khashabi\*, Anqi Liu\* (Johns Hopkins University; \*equal advising)
**Type:** Theory + small-scale empirical validation. A **negative / non-existence** result about activation steering.

---

## 1. Core Claim

Activation steering pushes the residual stream **off the manifold of states reachable from discrete prompts**. Framed as a **surjectivity** question: does every steered activation admit a preimage under the model's natural forward pass? **Almost surely, no.**

Consequence: **white-box steerability does not imply black-box (prompt-side) exploitability.** Demonstrating that a behavior can be induced by steering is *not* evidence that a prompt exists to induce it.

Analogy used by the authors: steering is like a brain–computer interface altering muscle movement by external stimulation, rather than through natural motor control.

---

## 2. Contributions

1. **Non-surjectivity of steering** — formalize prompt-reachability as surjectivity; prove steered states almost surely have no exact prompt preimage.
2. **Empirical evidence** across three open-weight models comparing white-box steering trajectories to black-box prompt-only replication attempts.
3. **Threat-model implication** — motivates evaluation protocols that explicitly decouple internal controllability from prompt-side exploitability.

---

## 3. Background & Notation

- Vocabulary `V`; prompt set `S = V^{≤K}` (K = context window) — **countable and practically finite**.
- L-layer transformer, params `Θ ∈ R^P`. Model treated as `F: R^K × V × R^P → R`, computing `r_i = F(r_{<i}, s_i; Θ)` at position `i`, layer `j` (fixed w.l.o.g.).
- **Theorem 3.1 (Transformers are real-analytic)** — from Nikolaou et al., "Language models are injective and hence invertible." If MLP activations are real-analytic (tanh, GELU), then `r_i = F(r_{<i}, s_i; Θ)` is real-analytic in `Θ`.
- **Injectivity** (Nikolaou et al.): for random init from practical distributions (Gaussian, Xavier), distinct prompts almost surely never collide: `P(r_i = r'_i) = 0`. Proof uses Mityagin — the zero set of a non-identically-zero real-analytic function has measure zero, applied to `h(Θ) = ‖r_i − r'_i‖²`. Preserved under a finite number of GD steps → applies to real trained LLMs.

---

## 4. Theory

**Definition 4.1 (Steering mechanism):** `r̃_i = F(r̃_{<i}, s̃_i; Θ) + λ·v`, steering vector `v ∈ R^d`, coefficient `λ`, applied at all token positions in the context window.

**Definition 4.3 (Difference-of-Means / DOM steering vector):** fixed layer ℓ and position index (e.g. −1, the last non-padded token):
`v(Θ, D) := (1/|D₊|) Σ_{x∈D₊} F_{−1ℓ}(x;Θ) − (1/|D₋|) Σ_{y∈D₋} F_{−1ℓ}(y;Θ)`
Since `F` is real-analytic in `Θ` and `v` is a finite linear combination of such maps, `v(·, D)` is itself real-analytic in the same `Θ`.

### Three results (built in escalating strength)

| Theorem | Statement | Interpretation |
|---|---|---|
| **4.2** | For `Θ ~ μ`, `v ~ γ` with non-zero densities: `P(r̃_i = r'_k) = 0` for any prompts `s, s'` and positions `i, k`. | Random steering vectors move activations off the natural manifold. `Im(F)` is a *countable* set of points; everything else is a "hole" in `R^d`. |
| **4.4 (Almost sure non-intersection)** | With DOM vector `v(Θ)`, `\|D₊\|,\|D₋\| ≥ 2`, distinct prompts `s, s'`: `P_{Θ~μ}(r̃_i = r'_k) = 0`. | Realistically extracted (difference-of-means) steering vectors inherit the same non-surjectivity. |
| **4.5 (Almost sure sequence divergence)** | Let `v*` be an *adversarial* steering vector that forces `r̃_i = r'_k`. Then `P_{Θ~μ}(r̃_{i+1} = r'_{k+1}) = 0`. | Even a deliberately engineered collision diverges at the very next position. Matching a whole sequence requires a probability-zero intersection at every step. |

**Proof technique (all three):** define a *steering collision function*, e.g.
`g(Θ, v) = ‖F(r'_{<k}, s'_k; Θ) − (F(r̃_{<i}, s̃_i; Θ) + v)‖²` (and `g_next` for Thm 4.5),
show it is real-analytic in the joint space and **not identically zero** (via explicit witness constructions `Θ*` in Appendix A), then apply Mityagin's measure-zero zero-set result.

**Caveat the authors flag:** theoretically there could exist models with non-zero collision probability, but they would need adversarial initialization from a zero-density distribution, maintenance of the collision property throughout training, *and* standard NL capability. No such model is known.

---

## 5. Experimental Setup

**Models (three families, deliberately small to make exhaustive token search tractable):**
- Llama-3.2-1B-Instruct
- Qwen-2.5-0.5B-Instruct
- gemma-3-1b-it

Non-thinking chat models only (standard setup for the steering methods being replicated).

**Steering vectors (two):**
1. **refusal** — Arditi et al. refusal direction, applied with `λ` **negative** (λ = −1) to break safety alignment.
2. **persona** — Chen et al. persona vectors, applied with `λ` **positive** (λ = +1); they test the "evil" persona.

**Prompts:** 10 harmful prompts sampled from Arditi et al.'s set (refusal); 10 prompts from Chen et al.'s evil-persona eval set (persona).

**Decoding:** greedy, for consistency.

**Procedure:** run prompts `s` normally → collect natural activations `r` and generations `g`. Run with steering → collect steered activations `r̃` and steered generations `g̃`. Goal: find prompts `s'` whose *natural* activations `r'` match `r̃`.

---

## 6. Empirical Methods & Findings

### 6.1 SipIt inversion (§5.1)
- **SipIt** (Nikolaou et al.) is an `O(N·|V|)` algorithm that inverts natural activations back to the producing prompt. It requires knowing prompt length and activation positions in advance. It tests all tokens at position 1 until one matches the given activation, fixes it as prefix, then repeats.
- **Baseline works:** original prompts were successfully recovered from **natural** activations across all models (L2 ≈ 0 for the top token).
- **Steered activations fail:** SipIt fails at the **very first token**, for all models and all prompts. Steered activations sit far from the activations of *any* natural token (shown via top-2 nearest-token L2 distances). This is the core empirical evidence of non-surjectivity.
- **Surprising secondary finding:** projecting `r̃` to nearest tokens (`s' = proj(r̃_s)`) recovers *the original test prompt* almost exactly (`s' ≈ s`), even at high `λ`. Generating from `s'` always yields standard, **non-steered** behavior. So steering induces an unnatural shift that no other prompt imitates — it does not move you toward some other prompt's activations.

Example (Llama-3.2-1B-Instruct, Table 1):
- refusal steering: natural output refuses; steered output complies. Reconstructed `s'` differs from `s` only in a trailing special `<eot>` token, and its response refuses again. `r` invertible = Yes; `r̃` invertible = **No**.
- evil-persona steering: natural output is ordinary career advice; steered output is malicious. Reconstructed `s'` == `s` exactly; response is the ordinary one. `r̃` invertible = **No**.

### 6.2 Many-shot ICL prefixes (§5.2)
- Motivation: many-shot jailbreaking (Anil et al.) achieves behaviorally similar results to negative refusal steering, and ICL relaxes SipIt's equal-length assumption by allowing prefixes.
- **Finding:** ICL demonstrations *can* elicit similar behavior (bypassing refusal), but the **internal mechanisms and generated outputs diverge** — no activation alignment at either the prompt position (`‖r̃_s − r'_s‖ ≫ 0`) or the response position (`‖r̃_g̃ − r'_g̃‖ ≫ 0`).
- Interpretation: behavioral similarity ≠ activation equivalence. This undermines many-shot prompting as a search avenue for steering-matching prompts, without proving absolute non-existence.

### 6.3 Other prompt-search methods (Appendix F)
Prefix tuning, **GEPA**, and copying-via-instructions were also tried. **None succeeded** in replicating steered activations.

---

## 7. Implications & Discussion

- **The headline non-implication:** a recurring conflation in the literature treats "easy to induce via steering" as "the model is exploitable via prompts." This paper shows that inference is invalid.
- **Open-weight vs. closed-weight:** steering can genuinely bypass safety in open-weight or developer-controlled settings (Arditi et al.; Wang & Shu's Trojan Activation Attack). But those demonstrations do not automatically imply risk in closed-weight deployments where users only have black-box access.
- Motivates evaluation protocols that explicitly **decouple white-box controllability from black-box exploitability** (contrast with Casper et al., "Black-box access is insufficient for rigorous AI audits," and Che et al. on model-tampering attacks as more rigorous evaluation).
- Cited data point: Anthropic reports Claude 4.5 produced near-zero unsafe responses in standard safety tests, yet activation steering suppressing evaluation-awareness increased unsafe behavior, with one trial at an 8% misalignment rate under a particular steering vector.

---

## 8. Limitations (authors' own)

- The primary contribution is a **theoretical non-existence result**. Empirically *proving* non-existence of a matching prompt is intractable — the prompt space is exponentially large. The experiments "provide a peek into the complicated landscape" rather than a proof.
- Experiments restricted to **small models (0.5B–1B)** purely to manage the cost of exhaustive token search.
- Thinking/reasoning models excluded; steering-vector extraction for them is more complex (though application is similar).

---

## 9. Quick Reference

| Item | Value |
|---|---|
| Models | Llama-3.2-1B-Instruct, Qwen-2.5-0.5B-Instruct, gemma-3-1b-it |
| Steering directions | refusal (Arditi et al.), evil persona (Chen et al.) |
| λ | −1 (refusal removal), +1 (persona addition); also swept |
| Prompts | 10 per direction |
| Decoding | greedy |
| Inversion algorithm | SipIt, `O(N·\|V\|)` |
| Failed prompt-search methods | nearest-token projection, many-shot ICL, prefix tuning, GEPA, copying-via-instruction |
| Key theorems | 4.2 (random `v`), 4.4 (DOM `v`), 4.5 (adversarial `v*`, sequence divergence) |
| Key prior results relied on | Nikolaou et al. (injectivity/invertibility), Mityagin (zero set of real-analytic functions) |
