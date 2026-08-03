# RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs

**arXiv:** 2510.13901 · Nguyen, Le, Vu, Cooper, Susilo (VNPT AI / Univ. of Wollongong)
**Type:** White-box, single-instance adversarial suffix attack

---

## 1. Core Idea

Refusal responses form **dense clusters** in embedding space. Existing suffix attacks (GCG, COLD-Attack, ASETF) collapse into these clusters, which lowers ASR and raises detectability. RAID adds an explicit **refusal-aware geometric regularizer** to suffix optimization, plus a **coherence term** for fluency, then decodes continuous embeddings back to tokens with a **critic-guided beam search**.

Three components:
1. Embedding relaxation (continuous suffix `Z`)
2. Refusal-aware triplet regularizer
3. MMD coherence constraint + critic-guided decoding

---

## 2. Method

### 2.1 Problem setup & relaxation
- Target model `f_θ`, harmful instruction `x`, discrete suffix `s = (s_1..s_n) ∈ V^n`, input `x ⊕ s`.
- Objective: `s* = argmax P(Y_harm | x ⊕ s)`, i.e. minimize `L_aff(s) = −log P(Y_harm | x ⊕ s)`.
- **Relaxation:** replace discrete `s` with learnable continuous vectors `Z = (z_1..z_n) ∈ R^{n×d}`. Input becomes `E(x) ⊕ Z`, where `E(x)` is frozen (no gradient) and only `Z` is optimized.
- `L_aff(Z) = −log P(Y_harm | E(x) ⊕ Z)`.

### 2.2 Refusal-aware triplet regularization
Operates at a **fixed intermediate (middle) transformer layer ℓ**; `h^(ℓ)(·)` returns the hidden state of the **last suffix token** (pooling over suffix positions gives comparable results).

- **Refusal mean `r`:** collect activations for sampled suffixes whose generations match a refusal template set `T` (template-matching predicate `Match(y,T) ∈ {0,1}`); average them. Updated online with EMA: `r_i ← (2/3)·r_{i−1} + (1/3)·mean(h_refusal(x ⊕ s_m))`.
- **Refusal direction `d`:** difference of means over harmful vs. harmless instruction sets — `d = μ − ν`, where `μ = mean h^(ℓ)(t), t ∈ D_harmful` and `ν = mean h^(ℓ)(t), t ∈ D_harmless`. (Follows Arditi et al. 2024 single-direction hypothesis; cites Wollschläger et al. 2025 "concept cone" as the competing view.)
- **Anchor `a`:** `a = h^(ℓ)(x ⊕ s)` for the current suffix.
- **Directional ablation / positive `p`:** `p = a − (dᵀa)d` — the refusal-ablated counterpart of the anchor.
- **Triplet loss (margin m > 0):**
  `L_refusal = max{0, ‖a − p‖₂ − ‖a − r‖₂ + m}`
  Pulls the anchor toward the ablated point `p`, pushes it away from the refusal mean `r`.

### 2.3 Coherence via MMD
Makes relaxed suffix embeddings `Z` resemble a benign reference distribution `B = {b_1..b_M}` (either token embeddings sampled from `W`, or hidden activations of benign suffixes at layer ℓ).

- Unbiased squared MMD between `Z` and `B` with **Gaussian RBF kernel**; bandwidth `σ` set by median-distance heuristic or a small multi-kernel mixture.
- `L_MMD(Z; B) = MMD²(Z, B)`.
- Effects: (i) penalizes clustered/repetitive suffix embeddings, (ii) keeps them near benign text statistics, (iii) fully differentiable.
- **Stability note:** using the *same* representation space for `B` and `Z` (both token-embedding space, or both layer-ℓ activation space) trains most stably.

### 2.4 Critic-guided decoding (Algorithm 1)
Nearest-neighbour projection per position ignores sequence coherence, so RAID uses beam search scoring each candidate token by a convex combination:

`S_i(v | s_{1:i−1}) = λ·sim(z_i, W_v) + (1−λ)·log p_{f_θ}(v | x ⊕ s_{1:i−1})`, λ ∈ [0,1]

- `sim` = cosine similarity; `W_v` = embedding of token `v`.
- **Shortlisting:** candidate set `C_i = TopK_v sim(z_i, W_v)` — score only shortlisted tokens.
- Fallback when LM probs unavailable/expensive: n-gram log-probability. (Paper found **LM-based coherence most effective**; n-gram is a fast fallback.)
- Length normalization when comparing completed beams; optional final reranking by full-sequence LM log-prob + average affinity `(1/n)Σ sim(z_i, W_{t_i})`.
- **Complexity:** `O(n·k·K)` scoring calls.

### 2.5 Full objective
`L_RAID(Z) = L_aff(Z) + λ_refusal · L_refusal(Z; d, r) + λ_MMD · L_MMD(Z; B)`

Optimized w.r.t. `Z` only (gradient descent / Adam), `f_θ` and `E(x)` frozen: `Z ← Z − η ∇_Z L_RAID(Z)`.

### 2.6 Auxiliary algorithm — DecodeSuffix (Alg. 2, Top-K stochastic projection)
Used during optimization to sample discrete suffixes for refusal-mean estimation: per position compute `C_j = TopK sim(z_j, W_v)`, then sample `N_seed` sequences by uniform (or softmax-over-similarity) sampling from `C_j`.

---

## 3. Hyperparameters

| Parameter | Value / range |
|---|---|
| Beam width `k` | 4–16 |
| Shortlist size `K` (TopK) | 32–128 |
| Triplet margin `m` | > 0 (value not specified) |
| `λ` (decoding affinity vs. LM coherence) | ∈ [0,1] |
| `λ_refusal`, `λ_MMD` | ≥ 0 (trade-off weights) |
| Layer `ℓ` | intermediate / middle layer |
| Kernel | Gaussian RBF, σ by median heuristic |
| Optimizer | GD / Adam, lr `η` |
| Generation (test cases) | temperature 0.2, max_tokens 512 |

---

## 4. Experimental Setup

- **Models:** Llama-2-7B(-chat), Mistral-7B-v0.2, Guanaco-7B, Vicuna-7B-v1.5.
- **Dataset:** AdvBench — 1,000 harmful queries (500 harmful strings + 500 harmful behaviors).
- **Baselines:** PEZ, GCG, COLD-Attack, ASETF (ASETF numbers taken from its paper — no code released).
- **Metric:** ASR = successful jailbreaks / total attempts × 100%; plus average computation time per attack.
- **Two scenarios:** (1) no system prompt; (2) with default system prompts (basic and complex variants).

---

## 5. Results

### Scenario 1 — ASR (%), no system prompt

| Method | Llama-2-7B | Mistral-7B-v0.2 | Guanaco-7B | Vicuna-7B-v1.5 |
|---|---|---|---|---|
| PEZ | 18.00 | 16.00 | 52.00 | 48.00 |
| GCG | 88.00 | 100.00 | 100.00 | 97.69 |
| COLD-Attack | 88.85 | 94.81 | 98.65 | 97.12 |
| ASETF | 91.00 | 95.00 | N/A | 94.00 |
| **RAID** | **92.35** | **100.00** | **100.00** | **100.00** |

> Note: the paper's prose cites 80.00% for GCG on Llama-2 while its table reports 88.00% — an internal inconsistency.

### Scenario 2 — ASR (%) with system prompts (Llama-2-7B)

| Method | Basic sys prompt | Complex sys prompt |
|---|---|---|
| PEZ | 0 | 0 |
| GCG | 20 | 0 |
| COLD-Attack | 50 | 20 |
| ASETF | N/A | N/A |
| **RAID** | **60** | **20** |

System prompts sharply degrade all methods; RAID retains the largest margin under the basic prompt and ties COLD-Attack under the complex one.

### Compute time per attack (Llama-2-7B)
RAID **93 s** < ASETF 104 s < PEZ 254 s < COLD-Attack 325 s < GCG 1146 s.
RAID is simultaneously the fastest and the highest-ASR method — it avoids GCG's repeated forward/backward token-level search.

---

## 6. Key Insights / Takeaways

1. **Refusal clustering is the failure mode.** Prior suffix attacks collapse into refusal regions of embedding space; explicitly repelling from the refusal mean while pulling toward the directional-ablation target avoids that collapse.
2. **Embedding-space regularization > token-level search** for both effectiveness and cost. Continuous relaxation + one decoding pass beats discrete combinatorial search by ~12× wall clock.
3. **Distributional (MMD) coherence** is what keeps the suffix low-perplexity and evades perplexity-based filters, without needing a separate fluency model.
4. **Decoding matters.** Per-position nearest-neighbour projection loses sequence coherence; a critic that mixes embedding affinity with LM likelihood recovers it.
5. **Defense implication:** defenses relying only on refusal clustering or perplexity detection remain vulnerable to geometry-aware adversaries. Representation-level and decoding-time defenses are needed.
6. **Limitation:** RAID is explicitly **single-instance** (per prompt–model pair), not universal/transferable. Universal and cross-model extensions are left to future work.

---

## 7. Related-Work Positioning (Table 1 in paper)

| Method | Access | Key idea |
|---|---|---|
| GCG | white-box | gradient-guided discrete token search; strong ASR, high perplexity |
| COLD-Attack | white-box | energy-based constrained decoding; controllable, costly, no refusal-awareness |
| ASETF | white-box | continuous embedding optimization + NN decoding; efficient but projection artifacts |
| DiffusionAttacker | white-box | seq2seq diffusion rewriting; fluent, expensive |
| JBFuzz | black-box | fuzzing-style mutation + lightweight evaluator |
| JOOD | black-box | out-of-distribution transformation of harmful inputs |
| **RAID** | white-box | refusal-aware embedding optimization + critic-guided decoding |

---

## 8. Ethics / Reproducibility Notes
- Appendix A lists 7 representative test cases (TC-01…TC-07) across code exploitation, incitement, identity theft, financial fraud, counterfeiting, IED construction, and cyberbullying. The authors state that verbatim actionable content has been redacted and full transcripts are gated behind a responsible-disclosure process. (Specific suffixes and model outputs are deliberately not reproduced in these notes.)
- No public code release is mentioned.
