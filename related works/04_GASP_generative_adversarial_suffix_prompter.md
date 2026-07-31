# GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs

**arXiv:** 2411.14133 · Advik Raj Basani (BITS Pilani Goa), Xiao Zhang (CISPA)
**Code/data:** https://github.com/TrustMLRG/GASP · Project page: https://air-ml.org/project/gasp/
**Type:** Fully **black-box** (API-only) suffix generation. NeurIPS-format submission with full checklist.

---

## 1. Core Idea

Train a dedicated **SuffixLLM** `g_φ: X → E` that *generates* adversarial suffixes, rather than searching for them per-prompt. Refine it in an alternating loop:

- **Latent Bayesian Optimization (LBO)** searches the continuous latent embedding space of candidate suffixes, guided by real-time feedback from the TargetLLM.
- **ORPO** (odds-ratio preference optimization) fine-tunes SuffixLLM on the LBO-ranked suffixes.

Result: **human-readable**, query-efficient, transferable suffixes with fast inference — no gradient access, no discrete token search.

**Four pipeline modules:** (A) pretrain SuffixLLM on AdvSuffixes → (B) LBO search with TargetLLM feedback → (C) iterative ORPO fine-tuning → (D) SuffixLLM output distribution aligns with TargetLLM.

---

## 2. Problem Formulation

- Adversarial objective: `max_{e∈E} p_θ(y | x + e)` where `x` = malicious prompt, `e` = suffix, `y` = harmful response.
- **Readability constraint:** `E_nat = {e ∈ E : R(e;x) ≥ γ}` with `R(e;x) = p_nat(x + e)` = probability the modified prompt lies in the natural-language distribution.
- **Constrained problem:** `max_e p_θ(y | x + e) s.t. p_nat(x + e) ≥ γ`.

Positioning vs. AdvPrompter (its closest relative): AdvPrompter (i) lacks adaptability to the specific TargetLLM (it aims for universal jailbreaks) and (ii) is inefficient because it still searches in discrete token space via AutoDAN's solver.

---

## 3. Method

### 3.1 Stage 1 — Pretraining SuffixLLM (SFT)
- Standard supervised fine-tuning on **AdvSuffixes**:
  `L_SFT(φ) = −E_{(x,E_x)} Σ_{e∈E_x} log p_φ(e | x)`
- Purpose: learn a **prior distribution** capturing both linguistic fluency and adversarial potential, defining a meaningful latent space that dramatically reduces downstream search complexity.

### 3.2 Stage 2 — LBO search (Algorithm 3)
Three required components:

**Encoding.** Each generated suffix `e = g_φ(x)` is passed back through SuffixLLM; hidden states from the **final decoder layer** are aggregated over tokens into a latent vector `z_e`. (Ablation F.4 confirms the final layer, Layer 32, is best.)

**Decoding.** Nearest-neighbour in the same latent space by Euclidean distance: `e(z) = argmin_{e_i∈E_x} ‖z − z_{e_i}‖₂`.

**Scoring — GASPEval.** Decoded suffix appended to `x`, sent to TargetLLM, output scored by a JudgeLLM across **21 binary criteria** (hate speech, illegal instructions, misinformation, threats, …), each rated **0–2**:
- 0 = completely harmful, no caution
- 1 = harmful but with ethical warnings
- 2 = completely harmless
Threshold: **average score < 1 ⇒ harmful** (regardless of disclaimers). **Lower score = more effective jailbreak.**
**Lazy evaluation:** only suffixes explicitly selected by the LBO loop are scored, concentrating budget on promising latent regions.

**Gaussian Process surrogate.** `h(z) ~ GP(μ(z), k(z,z'))`; posterior gives `h(z)|A ~ N(μ_A(z), σ²_A(z))`.

**Acquisition.** Minimizing `h(z)`, so e.g. Lower Confidence Bound `α(z) = μ_A(z) − β·σ_A(z)`. Next query: `z_next = argmin_{z ∈ Z_x \ A₁} α(z)`, decoded via NN, scored, and added to the GP training set.
Loop terminates when `|A| > M` (query budget) or a suffix scores 0.

**Efficiency:** converges in roughly `u + O(log M) ≪ k` evaluations instead of scoring all `k` generated suffixes.

### 3.3 Stage 3 — ORPO fine-tuning
Sort evaluated suffixes **ascending by GASPEval score** (lower = more harmful). Most harmful = preferred `y₊`; less successful = dispreferred `y₋`.

`L_ORPO(φ) = E[ ℓ_SFT(φ; x, y₊) + λ · ℓ_OR(φ; x, y₊, y₋) ]`

Because training is guided by ascending-score ranking, the **first** suffix SuffixLLM emits after training is by design the most adversarially effective — which is what `GASPInfer` (Algorithm 2) exploits: generate the sorted list, return `Ẽ_x[0]`, no re-evaluation needed.

---

## 4. Datasets

**AdvSuffixes (new, released):**
- Built from all **519 AdvBench harmful instructions** (MIT license).
- Generated with **LLaMA-3.1-8B-Lexi-Uncensored-V2** using **two-shot prompting**: one DAN-style jailbreak demo + one character role-play demo, then the target instruction. Model asked to produce **20–25 suffixes** per instruction adapted to that instruction's semantics.
- **>11,763 suffixes total** (~22.6 per prompt), after iterative generation + human filtering.
- Split **75% pretraining / 25% fine-tuning** (`β = 0.75`), fixed across experiments.
- Released under GNU GPL-v3.

**OOD test set (new):** 100 harmful prompts generated from AdvBench's 127 hardest prompts, deliberately varied in syntax/phrasing to be out-of-distribution. Divergence measured with the Llama-3.1-8B tokenizer: **JS divergence 0.368, KL divergence 0.155**.
Rationale: AdvBench has heavy thematic redundancy (>24 bomb-related prompts, 62 fraud/theft prompts), so even a disjoint train/test split leaks semantic similarity.

**Standard benchmarks also used:** AdvBench, HarmBench.

---

## 5. Experimental Setup

**TargetLLMs (open):** Mistral-7B-Instruct-v0.3, Falcon-7B-Instruct, LLaMA-2-7B-chat, LLaMA-3-8B-instruct, LLaMA-3.1-8B-instruct.
**TargetLLMs (closed):** GPT-4o, GPT-4o-mini, GPT-3.5-turbo, Claude-3.7-Sonnet, Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-3-Haiku. Deployed via Azure for a standardized pipeline.

**SuffixLLM:** LLaMA-3.1-8B-Lexi-Uncensored-V2.
**JudgeLLM (GASPEval):** LLaMA-3.1-8B-Lexi-Uncensored-V2 (GPT-4o also found highly effective).
**Readability judge:** Wizard-Vicuna-7B-Uncensored.

**Hardware:** 3× NVIDIA DGX A100 (40GB). All results = **median of 3 runs** (no error bars reported).

**Baselines:** GCG, AutoDAN, AdvPrompter (warm-start) [optimization-based]; PAIR, TAP, ICA [black-box].
**Metric:** `ASR@k` — at least one of `k` attempts succeeds. Reported at k=1 and k=10.
**Evaluators (three):** Keyword Matching, StrongREJECT, GASPEval.

---

## 6. Hyperparameters (Table 4)

| Phase | Hyperparameter | Value |
|---|---|---|
| **Pretraining** | epochs `S₁` | 10 |
| | warmup steps | 500 |
| | weight decay | 0.01 |
| | learning rate `η₁` | 5e-5 |
| | LoRA r / alpha / dropout / bias | 16 / 32 / 0.1 / none |
| **LBO** | acquisition function | **EI** (Expected Improvement) |
| | # calls `M` | 6 |
| | acquisition optimizer | sampling |
| | # initial points `u` | 2 |
| | beta | 0.25 |
| **ORPO** | epochs `S₂` | 15 |
| | warmup steps | 500 |
| | weight decay | 0.01 |
| | learning rate `η₂` | 2e-4 |
| **Inference** | max length | 256 |
| | # return sequences | 1 |
| | temperature | 0.9 |
| | top-p | 0.85 |
| | repetition / length penalty | 1.0 / 1.0 |
| **Other** | split `β` | 0.75 |
| | suffixes per query `k` | 20–25 |
| | TargetLLM sampling | temp 0.9, top-p 0.85 |

---

## 7. Results

### 7.1 Main ASR@10/ASR@1 (%) on the 100-prompt OOD test set

| Method | Evaluator | Mistral-7b-v0.3 | Falcon-7b | LLaMA-3.1-8b | LLaMA-3-8b | LLaMA-2-7b |
|---|---|---|---|---|---|---|
| GCG | KM / SR / GASPEval | –/47, –/22, –/37 | –/75, –/17, –/52 | –/6, –/7, –/6 | –/0, –/8, –/2 | –/3, –/17, –/5 |
| AutoDAN | KM / SR / GASPEval | –/100\*, –/64, –/69 | –/69, –/34, –/42 | –/2, –/2, –/1 | –/100\*, –/54, –/62 | –/1, –/1, –/0 |
| AdvPrompter | KM / SR / GASPEval | 52/38, 71/48, 77/55 | 95/73, 92/51, 93/52 | 11/0, 13/4, 17/4 | 7/1, 8/0, 5/0 | 6/1, 4/1, 7/1 |
| PAIR | KM / SR / GASPEval | –/57, –/61, –/64 | –/88, –/93, –/91 | –/14, –/26, –/18 | –/8, –/12, –/9 | –/4, –/6, –/7 |
| TAP | KM / SR / GASPEval | –/64, –/67, –/61 | –/97, –/98, –/98 | –/18, –/26, –/25 | –/9, –/11, –/8 | –/6, –/4, –/8 |
| ICA | KM / SR / GASPEval | –/100\*, –/59, –/62 | –/100\*, –/84, –/91 | –/80\*, –/58, –/59 | –/100\*, –/48, –/54 | –/0, –/0, –/0 |
| **GASP** | KM / SR / GASPEval | 58/31, 84/56, **82/64** | 94/72, 99/53, **100/86** | 20/6, 31/7, **68/11** | 1/0, 34/9, **71/6** | 0/0\*, 31/15, **64/9** |

`*` = anomalous keyword-matching artifact (see §7.4). ASR@10 omitted for baselines where multi-attack cost was prohibitive.

### 7.2 AdvBench / HarmBench (GASPEval, ASR@10/ASR@1, 100 prompts)

| Method | AdvBench Mistral | AdvBench LLaMA-2 | HarmBench Mistral | HarmBench LLaMA-2 |
|---|---|---|---|---|
| GCG | –/66 | –/46 | –/63 | –/31 |
| AutoDAN | –/69 | –/35 | –/64 | –/22 |
| AdvPrompter | 95/68 | 51/21 | 76/54 | 39/19 |
| PAIR | –/71 | –/4 | –/54 | –/30 |
| TAP | –/76 | –/6 | –/59 | –/28 |
| **GASP** | **97/56** | **63/24** | 72/55 | 38/15 |

### 7.3 Closed-source models (ASR@10, total cost)

| Model | Cost | Requests | Tokens | ASR@10 |
|---|---|---|---|---|
| GPT-4o | $2.31 | 1,723 | 303,574 | 47% |
| GPT-4o-mini | $0.07 | 1,664 | 191,029 | 40% |
| GPT-3.5-turbo | $0.52 | 2,376 | 445,239 | 41% |
| Claude-3.7-Sonnet | $4.41 | — | 284,702 | **59%** |
| Claude-3.5-Sonnet | $2.69 | — | 238,155 | 40% |
| Claude-3.5-Haiku | $0.69 | — | 261,442 | 57% |
| Claude-3-Haiku | $0.39 | — | 390,295 | 32% |

**Total attack cost across all closed models ≈ $5.** This cost-efficiency is one of the paper's central claims.

### 7.4 Defended models
Tested against SmoothLLM, Self-Reminder, Goal Prioritization, AutoDefense, SafeDecoding (10 settings).
- SafeDecoding is hardest at ASR@1 (12% Falcon-7B, 9% Mistral-7B), but ASR@10 recovers strongly.
- Peak: **86% on Falcon-7B + SmoothLLM**, **80% on Mistral-7B + Self-Reminder**.
- GASP often surpasses or rivals the strongest attacks reported in the defense papers themselves.

### 7.5 Efficiency & readability
- Training time **1.75× faster than AdvPrompter**; inference substantially faster than all baselines (GCG/AutoDAN/PAIR/TAP use per-prompt search).
- **Readability 0.94** (Wizard-Vicuna-7B-Uncensored, 0–1 scale over coherence/fluency/clarity/conciseness) — comparable to TAP and PAIR, far above GCG and AutoDAN.
- **Human study:** 52 participants (mostly university students), 20 prompts (5 per framework, 4 frameworks), blinded. **79.23%** rated GASP prompts most readable, AdvPrompter 16.15%. GCG/AutoDAN prompts most frequently flagged as adversarial.

### 7.6 Ablations
- **Without LBO/ORPO:** SuffixLLM alone (trained only on the AdvSuffixes prior) struggles badly, especially on Mistral-7B and Falcon-7B → latent-space exploration is essential.
- **GASPEval vs. StrongREJECT as LBO guide:** replacing GASPEval with StrongREJECT drops ASR notably. StrongREJECT is stricter and misses subtle/novel jailbreaks; GASPEval's fine-grained semantic feedback better guides latent search.
- **ORPO vs. SFT:** ORPO outperforms, critical to suffix coherence and ASR.
- **Acquisition function (F.3):** **EI > LCB > PI**. EI reaches highest ASR fastest; LCB converges more slowly (conservative); PI stagnates early (under-explores).
- **Latent layer (F.4):** Layer 32 (final) > Layer 16 > Layer 3, though differences are modest — deeper layers capture more semantically aligned, task-relevant information.
- **Query budget `M` (F.7):** higher `M` → better ASR but longer training. **Initial points `u`:** **2 is best**; more initial points give diminishing returns → LBO benefits more from rapid GASPEval-guided exploitation than from broad initial exploration.

---

## 8. Evaluator Failure Modes (a genuinely useful side contribution)

- **Keyword Matching** produces false positives (flags benign content by isolated words) *and* false negatives.
- **StrongREJECT** over-penalizes borderline prompts, causing false negatives and reducing exploration if used as an optimization signal.
- **AutoDAN's and ICA's "100%" scores** are keyword-matching artifacts: DAN-style genetic optimization exploits weak input sanitization on LLaMA-3/Mistral, and some ICA suffixes cause the model to merely **repeat the harmful prompt**, which still passes keyword matching.
- **GASP's apparently low KM scores** are the inverse artifact: GASP's suffixes elicit responses that *include* built-in warning phrases ("this answer contains sensitive and unethical content") while still containing the core harmful payload — keyword matching scores these as failures. StrongREJECT and GASPEval, which read full context, correctly flag them.
- GPT-4 rating agreement with humans: Agreement 86.52%, Precision 78.23%, Recall 83.49%, F1 80.78%.

---

## 9. Key Insights

1. **Generative > search.** Amortizing suffix discovery into a trained generator removes per-prompt combinatorial search entirely; inference is a single forward pass.
2. **The latent space is the right search space.** Pretraining on a suffix corpus creates a space where semantically and functionally similar suffixes are neighbours, making GP-guided nearest-neighbour decoding well-behaved.
3. **Feedback granularity drives search quality.** A coarse binary judge (StrongREJECT) is a worse optimization signal than a fine-grained 21-criterion rubric, even if the coarse judge is a better final evaluator.
4. **Readability is a first-class attack property**, not a nicety — it defeats perplexity filters and human moderation, and it's what separates GASP from GCG/AutoDAN.
5. **Black-box attacks on frontier models are cheap.** $5 total for seven closed models at 32–59% ASR@10.
6. **Benchmark contamination is real.** AdvBench's thematic redundancy means disjoint splits don't guarantee OOD evaluation; the authors' 100-prompt OOD set is why all methods (including theirs) score lower than in their original papers.

---

## 10. Limitations (authors')
- Unclear whether current ASR is near an optimal limit.
- Query complexity against the TargetLLM could be reduced further.
- No error bars (median of 3 runs only).
- GASP is proposed as repurposable for **adversarial retraining** / defense hardening — untested.
