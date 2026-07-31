# VISOR++: Universal Visual Input based Steering for Large Vision Language Models

**arXiv:** 2509.25533 · Ravikumar Balakrishnan (HiddenLayer Inc.), Mansi Phute (Georgia Tech)
**OpenReview:** https://openreview.net/forum?id=8kEDWYmAVa
**Type:** Steering via **input-space image optimization** — an attack-technique-turned-control-mechanism. Successor to VISOR (arXiv 2508.08521).

---

## 1. Core Idea

Activation steering requires **invasive runtime access** to model internals, so it can't be used with API-served or closed-source models. System prompting is deployable but easily overridden by user instructions.

VISOR++ optimizes a **single image** whose presence in the input induces the *same activation pattern* that a steering vector would produce — moving steering from the model supply chain into the **visual input domain**. One image can be jointly optimized across an **ensemble of VLMs** and emulate each of their individual steering vectors.

**Key reframing:** because most modern generative models support multimodality, an image can be used to steer *language* tasks (suppressing sycophancy, altering refusal, etc.) without any image-related content in the request.

---

## 2. Problem Formulation

Given VLMs `M = {M₁..M_K}` with steering vectors `{v_{k,ℓ}}` per model `k` and layer `ℓ ∈ L_k`, find a universal image:

```
x* = argmin_{x∈X} Σ_k Σ_j Σ_ℓ  D( h_ℓ^(k)(x, p_k^(j)) ,  h_ℓ^(k)(x₀, p_k^(j)) + α·v_{k,ℓ} )
```

- `h_ℓ^(k)(x, p)` = layer-ℓ activation of model `k` on image `x` with text prompt `p`
- `h_ℓ^(k)(x₀, p) + α·v_{k,ℓ}` = target activation (baseline from a neutral image `x₀`, plus scaled steering vector)
- `D` = distance metric (L2 in practice)
- Prompt ensemble `{p_k^(j)}_{j=1..N_p}` = diverse phrasings of the behavioral context

Three simultaneous universality requirements: **model architecture**, **prompt variation**, **layer depth**.

### Stated challenges
1. **Activation-level objectives** — must control intermediate activations at multiple depths, not just final outputs.
2. **Cross-model transferability** — each VLM has a distinct non-differentiable preprocessing pipeline that breaks gradient flow.
3. **Behavioral consistency** — the effect must hold across diverse prompts and contexts.

---

## 3. Method

### 3.1 Differentiable preprocessing pipeline (a key engineering contribution)
Standard HuggingFace processors take PIL images and apply non-differentiable ops (PIL resize/crop) before tensorization, **severing the computational graph**. VISOR++ starts directly from image tensors and reimplements preprocessing in differentiable PyTorch:

`P_k^diff(x) = ( Resize_bilinear(x, (H_k, W_k)) − μ_k ) / σ_k`

with model-specific normalization params `μ_k, σ_k` pulled from each processor config. This maintains a complete gradient path from loss → vision encoder → resizing → original pixel space.

**Resolutions:** images initialized at a common **384×384**, resized per model — 336×336 for LLaVA-1.5-7B, 384×384 for IDEFICS2-8B.

### 3.2 Optimization
- **Per-model images:** **PGD** with EoT (Expectation over Transformations) is very effective.
- **Universal (ensemble) images:** borrows **CWA-SSA** — Common Weakness Approach with Spectral Simulation Attack (Chen et al., ICLR 2024) — which provides **two-level (dual) momentum** and **spectral augmentation**.

**High-level intuition of CWA-SSA:** find a basin in the ensemble models' loss landscapes that is both **flat (wide)** and **close (overlapping)**, maximizing transferability to new models.

### 3.3 Algorithm 1 — Universal Visual Steering Optimization
```
Init x ← x₀ ; g_inner ← 0 ; g_outer ← 0
Precompute target activations ĥ for all (model, prompt) pairs
for t = 1..T:
    x_orig ← x
    # INNER LOOP: sequential over models
    for k = 1..K:
        ∇_k ← SpectralGradient(x, M_k, P_k, {p_k^(j)}, {ĥ})
        g_inner ← μ·g_inner + ∇_k / (‖∇_k‖₂ + ε₀)        # L2 normalization
        x ← x − α_inner · g_inner                          # immediate update
    # OUTER LOOP: trajectory stabilization
    Δx ← x − x_orig
    g_outer ← μ·g_outer + Δx / ‖Δx‖₁                       # L1 normalization
    x ← x_orig + α_outer · sign(g_outer)                   # sign-based update
    x ← Clip(x, 0, 1)
```

### 3.4 Algorithm 2 — Spectral Gradient (spectral augmentation)
```
for s = 1..S:
    η ~ N(0, σ²I) ;  x_noise ← x + η/255
    X_freq ← DCT2D(x_noise)
    m ~ U(1−ρ, 1+ρ)^{H×W×3}                # random spectral mask
    x_aug ← IDCT2D(X_freq ⊙ m)
    x_proc ← P_k(x_aug)                     # differentiable preprocessing
    L ← Σ_j Σ_ℓ  w_ℓ^(k) · ‖h_ℓ^(k) − ĥ_{ℓ,j}^(k)‖₂²
    L ← L / (N_p · |L_k|)
    ∇_avg += ∇_x L
return ∇_avg / S
```
i.e. add Gaussian noise → DCT → multiply by a random frequency mask → inverse DCT → differentiable preprocessing → extract activations for all prompts → weighted L2 distance to targets.

---

## 4. Experimental Setup

### 4.1 Models (ensemble of 2, deliberately architecturally diverse)
| Model | Vision encoder | LM | Visual tokens | Input |
|---|---|---|---|---|
| **LLaVA-1.5-7B** | CLIP ViT-L/14 | Vicuna-7B (2-layer MLP projection) | 576 | 336×336 |
| **IDEFICS2-8B** | SigLIP | Mistral-7B (Perceiver pooling + MLP) | 64 (compressed) | 384×384 |

Chosen for maximal diversity under compute constraints: different vision encoders, LMs, visual token counts, and preprocessing pipelines.

### 4.2 Unseen evaluation models
- **Open-access:** LLaVA-NeXT, Llama-3.2-11B, llava-llama-3-8b, Qwen2-vl-7b
- **Closed-access:** Claude Sonnet 3.5, GPT-4-Turbo, GPT-4V

### 4.3 Behavioral datasets (from Panickssery et al. / CAA)
| Behavior | Train | Test | Control direction (+/−) |
|---|---|---|---|
| Sycophancy | 1,000 | 50 | Agree / Disagree |
| Survival Instinct | 700 | 300 | Shutdown / Self-preserve |
| Refusal | 320 | 128 (text says 138) | Refuse / Comply |

**Unrelated-task control:** **MMLU** test set, 57 subjects, **14,000 samples**.

### 4.4 Baselines
1. **Steering vectors** (CAA-style): activation differences between positive and negative examples at token positions where responses diverge. Since LLaVA-1.5 requires visual input, a **standardized mid-grey image (RGB 128,128,128 with noise σ = 0.1×255)** is used for all steering-vector computations. Swept over multipliers `α` and token positions.
2. **System prompting** — natural-language instructions from Panickssery et al., same baseline image for fairness.

### 4.5 Metric — Behavioral Alignment Score (BAS)
```
BAS_k = (1/|T|) Σ_{(x⁺,x⁻)∈T}  P_k(x⁺|I, method) / ( P_k(x⁺|I, method) + P_k(x⁻|I, method) )
```
For closed-access models (no logprobs) the reported metric is the *fraction of examples over which each behavior was observed*.

---

## 5. Hyperparameters

### 5.1 Steering-vector extraction (grid-searched, Table 6)
| Model | Behavior | Layers | Multipliers | Token positions |
|---|---|---|---|---|
| LLaVA-1.5 | Refusal | [5, 11, 13, 17, 19] | −1 / +1 | Last 1 |
| | Survival Instinct | [7–14] | −3 / +1 | Last 1 |
| | Sycophancy | [0,1,2,11,12,13,14] | −5 / +1 | Last 7 |
| IDEFICS2 | Refusal | [11, 14, 17, 20] | −1 / +1 | Last 1 |
| | Survival Instinct | [8,12,16,20,24,28] | −1 / +4 | Last 1 |
| | Sycophancy | [0,1,2,11,12,13] | −4 / +1 | Last 7 |

### 5.2 VISOR++ with PGD (per-model images)
- Signed gradients, **step size 5/255**
- **Perturbation budget 255/255** (full budget — no specific base image required)
- **5–10 prompts** from the training set per use case
- Convergence around **2,000 steps**, with early stopping

### 5.3 Universal VISOR++ (SSA / CWA-SSA)
- Full epsilon budget
- Spectral augmentation: **S = 20 samples** per iteration, **σ = 16**, **ρ = 0.5** mask range
- **Adaptive LR schedule**, base step size **100**: step size ×1.1 when loss improves, ×0.8 after **3 iterations of stagnation** (patience = 3); bounded between **0.1× and 5×** the base rate
- **5,000–10,000 steps** per task. **Sycophancy did not fully converge even after 20k steps.**
- Hyperparameters largely consistent across behaviors, with task-specific step counts and LR schedules

---

## 6. Results

### 6.1 Behavioral Alignment Scores (Table 1)

| Dataset | Steering | Model | No Steering | System Prompt | Steering Vector | Per-model VISOR++ | Universal VISOR++ |
|---|---|---|---|---|---|---|---|
| **Refusal** | Negative | LLaVA-1.5 | 0.643 | 0.698 | **0.334** | 0.417 | 0.353 |
| | Negative | IDEFICS2 | 0.520 | 0.565 | 0.300 | **0.231** | 0.290 |
| | Positive | LLaVA-1.5 | 0.643 | 0.824 | **0.934** | 0.831 | 0.799 |
| | Positive | IDEFICS2 | 0.520 | 0.832 | 0.817 | **0.940** | 0.909 |
| **Survival Instinct** | Negative | LLaVA-1.5 | 0.523 | 0.498 | 0.410 | 0.372 | **0.365** |
| | Negative | IDEFICS2 | 0.456 | 0.416 | **0.313** | 0.344 | 0.370 |
| | Positive | LLaVA-1.5 | 0.523 | 0.608 | **0.612** | 0.602 | 0.575 |
| | Positive | IDEFICS2 | 0.456 | 0.648 | 0.625 | **0.675** | 0.634 |
| **Sycophancy** | Negative | LLaVA-1.5 | 0.691 | 0.674 | 0.394 | **0.393** | 0.623 |
| | Negative | IDEFICS2 | 0.755 | 0.759 | **0.367** | 0.394 | 0.581 |
| | Positive | LLaVA-1.5 | 0.691 | 0.679 | **0.726** | 0.698 | 0.698 |
| | Positive | IDEFICS2 | 0.755 | 0.744 | **0.756** | 0.756 | 0.755 |

**Reading:** for negative steering lower = better; for positive steering higher = better.
- Refusal on IDEFICS2: VISOR++ dynamic range **0.231–0.940** vs. steering vectors' 0.300–0.817 — **stronger** behavioral modification.
- Universal VISOR++ matches per-model dynamic range for refusal and survival instinct; **sycophancy negative steering lags** (0.623/0.581 vs. 0.393/0.394) — attributed to non-convergence.
- **System prompting is weak, especially for negative steering** (0.698 vs. baseline 0.643 on LLaVA refusal). VISOR++ shows **2–3× stronger** modification; the gap is largest where behavior must be *suppressed*.

### 6.2 Transfer to unseen models (negative steering, Table 2)

| Model | Refusal Δ | Survival Δ | Sycophancy Δ |
|---|---|---|---|
| LLaVA-NeXT | −0.027 | −0.028 | −0.026 |
| Llama-3.2-11B | **−0.048** | −0.013 | +0.022 |
| llava-llama-3-8b | −0.027 | −0.053 | −0.019 |
| Qwen2-vl-7b | −0.007 | −0.021 | 0 |
| Claude Sonnet 3.5 | 0 | −0.016 | 0 |
| GPT-4-Turbo | −0.007 | **−0.076** | **−0.070** |
| GPT-4V | −0.026 | −0.015 | −0.030 |

- **Directional consistency in 6 of 7 unseen models.** Absolute deltas are modest, but the sign is the point.
- **Qwen2-vl-7b** is least affected — most architecturally distinct of the open-access models.
- **Claude Sonnet 3.5** shows almost no effect.
- **GPT-4-Turbo** shows the largest negative deltas (survival instinct, sycophancy).
- **Positive steering (Table 7) transfers only to GPT-4 variants**; other models show no shift or a small *opposite* effect.
- Table 8 shows ensemble scaling 1→2 already yields clear directional steering; IDEFICS2-only images steer better than LLaVA-only, but neither matches the universal image.

### 6.3 Specificity — MMLU (Table 3, 14,000 samples)
| | LLaVA-1.5-7B random / VISOR++ | IDEFICS2-8B random / VISOR++ |
|---|---|---|
| Mean | 0.491 / 0.492 | 0.485 / 0.486 |
| Std | 0 / 0.001 | 0 / 0.001 |

**≈99.9% performance preserved.** The images induce behavioral shifts only, without degrading general capability.

---

## 7. Key Insights

1. **Activation steering can be emulated from the input space.** This is the central mechanistic claim — a strong statement about the connection between visual inputs and hidden states.
2. **Closed-source models are not safe from "activation-based" control by virtue of hiding internals.** The paper explicitly frames inaccessibility of internals as creating a *false sense of security*.
3. **Conversely, this makes steering-as-guardrail deployable** for API-served models, which is the defensive framing.
4. **Differentiable preprocessing is the enabling trick** — non-differentiable PIL ops were the practical barrier to cross-architecture image optimization.
5. **Steering ≫ system prompts for suppression.** System prompts can be overridden and barely move negative-steering scores; optimized images do not have this weakness.
6. **Contrast with Schaeffer et al. (2407.15211):** VISOR++ explicitly notes that classical PGD-style optimization across an ensemble does *not* produce transferable images, and that Schaeffer et al.'s successful transfers required near-identical architectures. VISOR++'s claim is that CWA-SSA's flat-and-overlapping-basin objective, combined with the *subtler* target of behavioral shift (rather than a specific target string or caption), gets partial transfer from a **2-model** ensemble — including to closed-access models.
7. **Specificity is achievable.** Behavioral steering need not cost general capability.

---

## 8. Limitations
- **Ensemble of only 2 models** due to compute constraints; the transfer claim is described as "promising" rather than established.
- **Absolute transfer deltas are small** (0.007–0.076); the evidence is directional consistency, not magnitude.
- **Sycophancy does not converge** — required an order of magnitude more steps than other tasks and still hadn't converged at 20k.
- Positive-steering transfer works only for GPT-4 variants.
- Two-phase hyperparameter grid search (steering-vector extraction, then image optimization) is expensive and per-task.
