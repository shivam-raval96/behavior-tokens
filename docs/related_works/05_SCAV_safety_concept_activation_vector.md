# Uncovering Safety Risks of LLMs through Concept Activation Vector (SCAV)

**arXiv:** 2404.12038 (v5, 30 Nov 2024) · Zhihao Xu\*, Ruixuan Huang\*, Changyu Chen, Xiting Wang (Renmin University of China; HKUST)
**Venue:** NeurIPS 2024 · **Code:** https://github.com/SproutNan/AI-Safety_SCAV
**Type:** White-box **embedding-level** attack + **prompt-level** attack, both guided by a learned safety classifier.

---

## 1. Core Idea

Prior embedding-space attacks (RepE, JRE) choose perturbation directions by **heuristics** (random subtraction of malicious/safe activations + PCA or dimension selection), giving no principled magnitude and requiring grid search. SCAV instead:

1. Learns a **linear classifier** `P_m(e)` = probability the LLM considers embedding `e` malicious.
2. Uses that classifier to derive a **closed-form optimal perturbation** (direction *and* magnitude), plus **automatic layer selection**.
3. Reuses the same classifier as the objective for a **prompt-level** attack (transferable to black-box APIs).

The framework doubles as an **interpretability probe** of how LLMs encode safety.

---

## 2. SCAV Framework

### 2.1 The classifier
`P_m(e) = sigmoid(wᵀe + b)`, `w ∈ R^d`, `b ∈ R`.

Trained with cross-entropy + regularization:
`argmin_{w,b} −(1/|D|) Σ [y log P_m(e) + (1−y) log(1−P_m(e))] + λ₁‖w‖² + λ₂b²`
`y = 1` if instruction malicious, 0 if safe.

**Implementation:** `sklearn.linear_model.LogisticRegression` defaults; **λ₁ = λ₂ = 0.5**.
**Critical detail:** the **L2 penalty is essential**. Replacing it with L1, or removing it, destroys the perturbation effect (ASR-keyword drops to 0). Varying the L2 coefficient over a wide range (0.5→3) has no meaningful effect (100/100/98/100).

### 2.2 Verifying the linear interpretability assumption
- **Aligned models** (Vicuna-v1.5-7B, LLaMA-2-7B/13B-Chat): test accuracy of `P_m` exceeds **95% from around layer 10–11**, rising to **>98%** at the last layers.
- **Unaligned model** (Alpaca-7B): accuracy much lower; t-SNE shows malicious/safe instructions are **completely inseparable**.
- Replicated on LLaMA-3-8B-Instruct, Qwen-1.5-7B-Instruct, Mistral-7B-Instruct-v0.2, Deepseek-v2-Lite-Chat, ChatGLM-4-9B-Chat — same pattern (low early, sharp jump to ≥90%, held to last layer).
- **Why it's easy:** distance statistics on LLaMA-2-7B-Chat show a large margin between classes and smaller within-class distances: `d_m` mean 56.21, `d_s` mean 84.89, **`d_{m/s}` mean 113.88** with the *lowest* variance (32.36). One malicious/safe pair suffices to train a ≥92% accurate classifier at layers 15+.

---

## 3. Embedding-Level Attack

### 3.1 Single layer — constrained optimization
Perturb `e → ẽ = e + ε·v`, and solve:

`argmin_{ε,v} |ε|  s.t.  P_m(e + ε·v) ≤ P₀,  ‖v‖ = 1`

- Minimizing `|ε|` ⇒ small performance loss (avoids repetitive/irrelevant output).
- Constraint `P_m(ẽ) ≤ P₀` ⇒ attack success (model no longer regards input as malicious).
- **`P₀ = 0.01%`**. Being a *probability* rather than a magnitude, `P₀` adapts `ε` automatically across layers and models.

**Closed-form solution (proof in Appendix C):**
```
ε = I(P_m(e) > P₀) · (sigmoid⁻¹(P₀) − b − wᵀe) / ‖w‖
v = w / ‖w‖
```
`v` is **perpendicular to the separating hyperplane** — the shortest path moving malicious embeddings into the safe subspace.

**Why baselines fail (Fig. 2):** RepE and JRE compute a *global* difference vector between malicious and safe embedding populations. This depends heavily on global data distribution, needs more data, and may not align with the separating hyperplane — JRE's vector can be *perpendicular to the correct direction* (Case 3), and RepE can produce *opposite* perturbations across random runs.

### 3.2 Multi-layer (Algorithm 1)
```
for l = 1..L:
    if TestAcc(P_m) > P1:                  # skip layers where safety concept not yet formed
        e ← embedding of x at layer l AFTER attacking previous layers
        if P_m(e) > P0:
            e ← e + ε·v
```
`P₁ = 90%`, `P₀ = 0.01%`. Perturbations are computed **sequentially** on the already-perturbed embedding, and applied at **every token step of generation**.

**Sensitivity (Table 13):** `P₀` matters more than `P₁`. `P₀ = 1e-4` or `1e-5` → 100% ASR-keyword at all `P₁ ∈ {0.85, 0.90, 0.95}`; `P₀ = 1e-3` degrades slightly (96–98%). Since separable layers typically exceed 85% test accuracy and non-separable ones fall below 70%, `P₁ ≈ 90%` is safe.

---

## 4. Prompt-Level Attack

GCG/AutoDAN maximize the probability of a *heuristic* target response `T` ("Sure, here is how to…"), which may differ from the real successful response. SCAV replaces this with the classifier:

`argmin_S  P_m(e^L_S) · ‖e^L_S − e^L‖`

- `e^L` = last-layer embedding of user instruction `x`; `e^L_S` = last-layer embedding when attack prompt `S` is prepended/combined.
- First term = attack effectiveness; second = minimal modification (preserves response quality).
- **Product form** chosen over a Lagrangian `‖·‖ + λ·P_m` to avoid an extra hyperparameter `λ` and to treat *percentage* changes in each term as equally important despite scale differences.
- Solved with **AutoDAN's hierarchical genetic algorithm** (paragraph-level + sentence-level populations), otherwise unmodified.
- Constrained form (Eq. 3) not used here because constraints don't fit the genetic algorithm and `P₀` can't be controlled directly.

---

## 5. Experimental Setup

**Victim models:** LLaMA-2-7B-Chat, LLaMA-2-13B-Chat (white-box); GPT-4-turbo-2024-04-09 (black-box).

**Training data (embedding attacks):** 140 malicious instructions from AdvBench + HarmfulQA, 140 safe instructions generated by GPT-4.
**Test data:** subset versions of **AdvBench** and **StrongREJECT**, 50 malicious instructions each, no overlap with training. Also HarmBench (80 cases) as a fully held-out check.

**Baselines:** DeepInception (manual prompts), AutoDAN, GCG (learned prompts), RepE, JRE (embedding perturbation), Soft prompt.
- AutoDAN: `num_steps = 100`, `batch_size = 256`.
- JRE reproduced (no code released): retain 35% of dimensions for 7B, 25% for 13B, perturb all layers.
- RepE: authors' released randomized dataset used to avoid the sign-flip problem.

**Compute:** 8× NVIDIA 32GB V100, `max_new_tokens = 1500`.

**Metrics:**
- **ASR-keyword** — refusal-keyword matching (keyword table in Appendix B.1). Known flaw: garbled output counts as success.
- **ASR-answer** — does the response give relevant information toward the goal?
- **ASR-useful** — is it actionable enough to act on immediately? (stricter)
- **Language flaws** = Repetition ∧ Inconsistence ∧ Unspecific.
GPT-4 used as judge (temperature 0, 5 repeated evaluations + voting; self-agreement 89.28%). GPT-4 vs. human: **Agreement 86.52%, Precision 78.23%, Recall 83.49%, F1 80.78%**. Human eval: 6 annotators, 3 per item, agreement 89.14%, IRB-equivalent approval obtained, <4h per annotator, above-minimum-wage pay.

---

## 6. Results

### 6.1 Embedding-level, automatic eval (AdvBench / StrongREJECT, %)

| Model | Method | ASR-keyword ↑ | ASR-answer ↑ | ASR-useful ↑ | Language flaws ↓ |
|---|---|---|---|---|---|
| LLaMA-2-7B | JRE | 80 / 90 | 76 / 72 | 68 / 70 | 70 / 70 |
| | RepE | 70 / 94 | 90 / 98 | 86 / 92 | 44 / 24 |
| | Soft prompt | 56 / 64 | 50 / 44 | 40 / 38 | 62 / 66 |
| | **SCAV** | **100 / 100** | **96 / 98** | **92 / 96** | **2 / 10** |
| | Δ vs best baseline | +20 / +4 | +6 / 0 | +6 / +4 | −42 / −14 |
| LLaMA-2-13B | JRE | 84 / 94 | 68 / 78 | 68 / 70 | 36 / 44 |
| | RepE | 86 / 92 | 88 / 98 | 84 / 94 | 20 / 18 |
| | Soft prompt | 80 / 74 | 66 / 28 | 50 / 28 | 44 / 68 |
| | **SCAV** | **100 / 100** | **98 / 100** | **96 / 98** | **0 / 2** |

Human evaluation (7B) confirms: SCAV ASR-answer 100/96, ASR-useful 92/90, language flaws 12/8.

### 6.2 Data efficiency
SCAV reaches ~100% ASR-keyword with only **5 malicious/safe instruction pairs**, with much lower variance than baselines. RepE scores **0** at dataset size 1.

### 6.3 Prompt-level attack (AdvBench / StrongREJECT, %)

| Model | Method | ASR-keyword | ASR-answer | ASR-useful | Lang. flaws |
|---|---|---|---|---|---|
| LLaMA-2-7B | DeepInception | 42/46 | 28/22 | 10/8 | 60/76 |
| | AutoDAN | 24/30 | 22/26 | 14/10 | 60/62 |
| | GCG | 28/26 | 32/26 | 10/16 | 76/72 |
| | **SCAV** | **54/60** | **60/46** | **44/40** | **52/44** |
| LLaMA-2-13B | GCG | 40/34 | 24/18 | 10/16 | 58/80 |
| | **SCAV** | **72/54** | **46/48** | **28/46** | 58/42 |

Improvements of **+12 to +42** on ASR criteria, up to **−18** language flaws.

### 6.4 Transfer of attack prompts to GPT-4
| Source | Method | ASR-keyword | ASR-answer | ASR-useful | Lang. flaws |
|---|---|---|---|---|---|
| LLaMA-2-7B | AutoDAN | 36/32 | 28/22 | 26/18 | 68/82 |
| | GCG | 4/8 | 4/16 | 2/16 | 92/90 |
| | **SCAV** | **70/30** | **66/20** | **52/20** | 68/72 |
| LLaMA-2-13B | **SCAV** | **82/40** | **48/26** | **60/22** | **54/72** |

Combining attack prompts from both LLaMA sizes (**SCAV-Both**): 96/52, 78/30, 80/36, 42/58.
Combining **all** prompt-level methods: GPT-4 gives **useful** responses to **84%** (AdvBench) / **54%** (StrongREJECT) of malicious instructions, most without language flaws.

### 6.5 Attacking 7 open-source models (embedding level)
LLaMA-2-7B/13B-Chat, LLaMA-3-8B-Instruct, Mistral-7B, Qwen-1.5-7B-Chat, Vicuna-v1.5-7B, WizardLM-2.
**Averages: ASR-keyword 99.71 / 98.57, ASR-answer 91.71 / 92.86, ASR-useful 84.29 / 89.71, language flaws 11.71 / 13.14.** Every model answers >85% of malicious instructions (one at 78% on AdvBench).
Additional models (Appendix F.2): ChatGLM4-9B 94/86/82/18; Deepseek-v2-lite-Chat 100/96/86/6; Gemma-1.1-7B-it 100/90/86/14. HarmBench (80 cases): LLaMA-2-7B 99.5/97.5/90/20; 13B 98.75/95/87.5/13.75.

### 6.6 Against defenses
- **Unlearning (Eraser):** other attacks are largely neutralized (AIM 0.5%, GCG 8.26%, AutoDAN 2.88% ASR-keyword; harmfulness ≈1/5). **SCAV: 97.34% ASR-keyword, harmfulness 4.72/5** on AdvBench and 98.79% / 4.86 on AdvExtent. ⇒ unlearning did **not** truly erase harmful knowledge.
- **Prompt-level defenses** (Self-reminder, ICD, Paraphrasing) on LLaMA-2-7B: ASR-keyword still 92/98/98, ASR-useful 88/90/90.
- **Adversarial training / circuit breakers:** LLaMA-3-8B-Instruct-RR 98 ASR-keyword / 74 ASR-useful; Mistral-7B-Instruct-RR 94 / 70.

---

## 7. Insights into LLM Safety Mechanisms (§4.3)

1. **Linear separability ≈ the safety mechanism.** Attacking a linearly separable layer (10, 20, 30) raises ASR-keyword; attacking layer 0 does not. Separability at a layer appears to indicate not only that the model *understands* the safety concept there, but that it will *use* it in subsequent layers.
2. **Different layers model safety from related but different perspectives.** Attacking only layer 10 lowers `P_m` at layer 10, but `P_m` **climbs back up** at later layers — the model self-corrects using remaining information. Perturbing layers 10–13 prevents that correction. ⇒ a *limited number* of layers jointly determine overall safety.
3. **Perturbation across layers is coupled.** Single-layer coefficients don't simply add: if layer `n` is selected, the coefficient for layer `n+1` shrinks. Intermediate layers are selected most often (layers 13–23 with probability >0.6; layers 24+ with probability 0–0.3), and single-layer perturbation rarely exceeds 90% ASR ⇒ multi-layer perturbation is necessary.
4. **Cross-model commonality.** Embedding-level attacks transfer between white-box LLMs (Vicuna / LLaMA-2 / Mistral / LLaMA-3) with sometimes-large ASR-keyword — e.g. Vicuna→LLaMA-2 100, Vicuna→Mistral 100, but Vicuna→LLaMA-3 only 4. Safety mechanisms share structure, but *when and why* transfer occurs is not understood.
5. **`P_m` predicts ASR.** Average post-attack `P_m` tracks ASR across methods: SCAV 0.01/0.01 (ASR-keyword 100/100) vs. DeepInception 72/69 (42/46), AutoDAN 70/67 (24/30), RepE 10/7, JRE 0.04/0.03.
6. **Prompt-level attacks affect intermediate layers too.** Eq. 5 uses only the last layer, yet optimizing on middle/late layers gives comparable performance — the attack prompt propagates through the whole forward pass during optimization.

---

## 8. Ablations
- **Automatic hyperparameter selection vs. human grid search:** automation improves ASR-useful by 2–10% and cuts language flaws by up to 20% (vs. humans manually perturbing layers 9–13 with unified `ε = −1.5`).
- **Perturbation direction:** at matched layers *and* at matched L2 perturbation magnitude, SCAV beats RepE and JRE on ASR-keyword — the advantage is the direction, not just the size.
- **Single-layer only:** insufficient; layer 0 perturbation produces entirely garbled output (ASR-keyword misjudged as high; true value 0).

---

## 9. Key Takeaways

1. **Principled beats heuristic.** A hyperplane-normal direction with a closed-form magnitude eliminates grid search, needs ~5 training pairs, and simultaneously improves ASR *and* output quality — the two usually trade off.
2. **Reparameterize magnitude as probability.** Setting a target `P_m ≤ P₀` instead of a raw `ε` makes the attack transfer across layers and models without retuning.
3. **Safety is linearly encoded from ~layer 10 onward in aligned models, and not at all in unaligned ones** — a clean operational signature of alignment.
4. **Safety is redundantly distributed across a small set of layers** with a self-correction mechanism; single-site interventions get repaired downstream.
5. **Unlearning defenses may be superficial.** Eraser blocks GCG/AutoDAN/AIM but is fully defeated by SCAV (97.34%), implying the harmful knowledge remains.
6. **ASR-keyword is unreliable** — it rewards garbled output. The paper's ASR-answer / ASR-useful / language-flaw decomposition is the more informative evaluation.

## 10. Limitations (authors')
No in-depth explanation of *why* perturbation vectors and attack prompts transfer between models.
