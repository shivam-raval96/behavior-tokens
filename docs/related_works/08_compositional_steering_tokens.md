# Compositional Steering of Large Language Models with Steering Tokens

**arXiv:** 2601.05062 (v1 8 Jan 2026, v2 19 Apr 2026) · Gorjan Radevski (Independent), Kiril Gashteovski (NEC Labs Europe / CAIR Skopje), Giwon Hong (Edinburgh), Carolin Lawrence (NEC Labs Europe), Goran Glavaš (Würzburg)
**Venue:** ACL 2026 · **Code:** https://github.com/nec-research/SteeringTokens
**Type:** Controllability / steering method. Not an attack paper.

---

## 1. Problem

Single-behavior steering is well studied; **compositional steering** — steering toward multiple behaviors simultaneously — is not.

Existing options and their failure modes:
- **Fine-tuning:** computationally intensive, risks negative interference and catastrophic forgetting; `N` arbitrarily composable behaviors imply `2^N` independent finetuning runs.
- **Natural-language prompting:** brittle; inconsistent behavior across semantically equivalent phrasings.
- **Prompt compression / activation steering / gist tokens / persona vectors:** compress individual behaviors well, but largely fail to generalize compositionally.

**The gap the paper names:** solutions for compressing *individual* behaviors are abundant; what's missing is **an effective representation for the concept of composition itself** that generalizes over arbitrary behavior combinations *and* over the arbitrary *number* of composed behaviors.

---

## 2. Method

### 2.1 Steering tokens live in the **input** space
For each behavior `b ∈ B`, introduce a trainable embedding `e_b ∈ R^d` (`d` = hidden size). Add one trainable **composition token** `<and>` with embedding `e_<and> ∈ R^d`.

For a prompt `x` with two behaviors, feed:
`[ E_x , e_{b_i} , e_{<and>} , e_{b_j} ]`
where `E_x` = frozen subword embeddings of `x`.

**Three claimed advantages:**
1. Prevents model collapse — all LLM parameters stay frozen.
2. Facilitates composition — behaviors combine through *learned interactions in the input space*, not by arithmetic on activations.
3. Computationally efficient — only `|B| + 1` d-dimensional vectors need learning.

**This is the key differentiator from prior work**, which operates in activation space. Directly composing independently trained modules on top of the same parameters/activations is known to be destructive.

### 2.2 Two-stage compositional self-distillation

**Stage 1 — individual behavior tokens.** Frozen LLM acts as both teacher and student.
- Teacher input: `x ⊕ I_b` (prompt + natural-language instruction)
- Student input: `x ⊕ <b>` (prompt + steering token)

`L_dist = KL( P_teacher(y | x, I_b) ‖ P_student(y | x, <b>) )`

Both distributions temperature-scaled: `P(y) = softmax(logits(y)/T)`; final loss scaled by `T²`. **T = 10.0** (high temperature, so the student matches the *full* distribution rather than just the mode).

Anti-overfitting: **10 instruction paraphrases per behavior** (e.g. "{Answer, Respond, Reply} in Spanish"), one sampled per training example.

**Stage 2 — composition token `<and>`.** LLM **and** behavior tokens both frozen; only `e_<and>` trains.
- Teacher: `x ⊕ I_{b_i} ⊕ I_{b_j}`
- Student: `x ⊕ <b_i><and><b_j>`

> **Freezing the behavior tokens in stage 2 is the crux.** It forces `<and>` to learn the *behavior-independent concept of composition* rather than just modifying individual behavior representations.

### 2.3 Initialization
- **Behavior tokens:** "semantic" init — mean of the LLM's frozen embeddings of the tokens in the behavior's instruction.
- **`<and>` token:** **zero init** — avoids biasing toward any behavior, letting it learn composition purely from data.

### 2.4 Orthogonality regularization
Prevents `<and>` from collapsing into representations similar to existing behavior tokens:

`L_orth = Σ_{b ∈ B_seen} ( e_<and> · e_b / (‖e_<and>‖ · ‖e_b‖) )²`

Final loss: `L = L_dist + λ · L_orth`, with **λ = 0.5**.

---

## 3. Experimental Setup

### 3.1 Models (7 instruction-tuned LLMs, 4 families)
Qwen3-4B, Qwen3-8B, Qwen3-14B, Llama-3.2-3B, Llama-3.2-8B, SmolLM3-3B, OLMo-7B.
- **Qwen-8B** = primary model for baseline comparison
- **Qwen family** = scaling analysis
- **Qwen-4B** = ablations (efficiency)

### 3.2 Behaviors — 15, in 4 categories, **verifiable** constraints only
| Category | Behaviors |
|---|---|
| Languages | Spanish, French, Italian, Portuguese, **German**\* |
| Length | 10–50, 50–70, **70–90**\*, 90–120 words |
| Formatting | lowercase, uppercase, **title case**\* |
| Structure | 1, 2, **3**\*, 4, 5 sentences |

`*` = **held-out / unseen** (4 total; 11 seen during `<and>` training). Every category has a held-out behavior, so zero-shot composition can be validated per category.

Rationale for verifiable constraints (following Stolfo et al.): satisfaction is automatically checkable, enabling large-scale compositional evaluation without human annotation or costly/error-prone LLM judges.

### 3.3 Composition types
- **Seen compositions:** both behaviors from the seen set (their combination was in `<and>` training).
- **Unseen compositions:** one or both behaviors from the held-out set.
- **3-behavior compositions are all unseen** by default, since `<and>` is trained only on 2-behavior combinations — a distinct generalization test over *number* of behaviors.
- For Qwen, a **2+3-token** variant is also trained (on both 2- and 3-behavior compositions) to test whether explicit 3-behavior supervision helps or overfits.

### 3.4 Data
- Prompts from **SmolTalk** (Apache 2.0).
- **Qwen3-30B-A3B-Instruct** used to separate the core question from embedded constraints.
- **50k prompts sampled per behavior**; answers generated by Qwen3-30B-A3B-Instruct (random paraphrase per example).
- For `<and>` training: responses generated for all **cross-category** 2-behavior combinations.
- Unseen behaviors excluded from all training-data creation.
- **Testing: 1,000 held-out prompts per combination**, all orderings → **>1M evaluations per model**.

### 3.5 Metrics
1. **Mean accuracy** — % of generations satisfying **all** k behaviors, averaged over all `k!` token orders (removes ordering bias). *Best accuracy* (max over orderings) also reported as an upper bound.
2. **Order variance** — `Δ_max = max_{i,j} |a_i − a_j|` across orderings of the same token set, averaged over combinations. Chosen over standard deviation because it captures **worst-case** ordering sensitivity — the practical risk that an unlucky arrangement tanks performance.
3. **Response quality** — Likert 1–5 by LLM judge (Qwen3-30B-A3B-Instruct), evaluated only on *accurate* responses. The judge prompt explicitly instructs it **not** to evaluate instruction-following (assume it was followed) but to use the instruction to set realistic expectations (don't penalize brevity if 10–50 words was required; don't judge English fluency if Spanish was required).

### 3.6 Baselines
1. **Instruction steering** (Stolfo et al.) — append behavior instructions to the prompt; random paraphrase sampling for fairness. The authors stress this is the **default real-world paradigm** and that prior compositional-steering work (Cao et al., Han et al., Nguyen et al.) mostly **fails to evaluate it**.
2. **LoRA DARE** — behavior-specific low-rank adapters trained with the same self-distillation objective, then merged via Yu et al.'s interference-reducing method. A parameter-space alternative to input-space steering.
3. **LM-Steer** (Han et al.) — steering vectors as linear projections of word embeddings; essentially token-based steering; claims (unquantified) compositional ability.
4. **Concatenation** — behavior tokens with *no* `<and>` token (does implicit attention-mediated interaction suffice?).
5. **Hybrid** — steering tokens + instructions together.

Standard CAA activation steering (Rimsky et al.) is **omitted** because Stolfo et al. show it substantially trails instruction-based steering for verifiable behaviors on its own.

---

## 4. Hyperparameters (Appendix A)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-3 |
| Batch size | as large as fits on one GPU |
| Epochs | 2, over 50,000 training examples |
| LR schedule | linear warmup (10% of steps) → linear decay to 0 |
| Gradient clipping | max norm 1.0 |
| Distillation temperature `T` | 10.0 (loss scaled by `T²`) |
| Orthogonality `λ_orth` | 0.5 |
| Precision | bfloat16 mixed |
| Data preprocessing | drop top 0.1% longest examples |
| Instruction paraphrases | 10 per behavior |

---

## 5. Results

### 5.1 Main comparison — Qwen-8B (Table 1)

| Method | 2-Beh Seen ↑ | 2-Beh Unseen ↑ | 2-Beh Ord.Var ↓ | 2-Beh Qual ↑ | 3-Beh Seen ↑ | 3-Beh Unseen ↑ | 3-Beh Ord.Var ↓ | 3-Beh Qual ↑ |
|---|---|---|---|---|---|---|---|---|
| LoRA DARE | 81.5 | 44.8 | — | 4.7 | 58.4 | 17.6 | — | 4.6 |
| LM-Steer | 18.1 | 13.4 | — | 1.3 | 2.2 | 2.1 | — | 1.2 |
| Instruction Steering | 90.7 | 71.8 | 7.8 | 4.9 | 83.7 | 54.0 | 18.1 | 4.9 |
| **Concatenation** (no `<and>`) | 81.3 | 62.1 | 25.3 | 4.8 | 59.6 | 33.2 | 55.8 | 4.8 |
| **Composition (`<and>`)** | 90.9 | **76.9** | 5.3 | 4.9 | 83.1 | 59.5 | 25.5 | 4.9 |
| **Hybrid (`<and>` + Instr.)** | **92.2** | 76.3 | **4.4** | 4.9 | **87.9** | **62.9** | **15.2** | 4.9 |

- Composition beats instructions on **unseen** compositions by **+5.1%** (2-behavior) and **+5.5%** (3-behavior); comparable on seen.
- **Concatenation collapses** on unseen 3-behavior (33.2% vs 59.6% seen) with huge order variance (55.8) → **an explicit learned operator is required**; implicit attention interaction is not enough.
- **LM-Steer fails completely** (2.2% on seen 3-behavior, response quality 1.2) — activation/embedding-projection steering does not compose.
- **LoRA DARE** generalizes poorly (44.8 → 17.6 on unseen).
- Response quality is uniformly ~4.7–4.9 except LM-Steer — steering accuracy does not cost content quality.

### 5.2 Cross-architecture (Table 2, unseen accuracy 2|3, order variance 2|3)

| Model | Instruction | Steering | Hybrid (Δ vs Instr.) |
|---|---|---|---|
| Qwen-4B | 68.9 \| 55.6 | 69.1 \| 60.7 | 69.2 (+0.3) \| 58.0 (+2.4) |
| Qwen-8B | 71.8 \| 54.0 | **76.9** \| 59.5 | 76.3 (+4.5) \| **62.9** (+8.9) |
| Llama-3B | 66.7 \| 33.8 | 69.3 \| 33.9 | **74.9** (+8.2) \| **43.4** (+9.6) |
| Llama-8B | 67.8 \| 40.2 | 67.0 \| 39.5 | **76.3** (+8.5) \| **52.9** (+12.7) |
| Smol-3B | 53.2 \| 32.5 | 53.2 \| 35.5 | **53.5** (+0.3) \| **37.2** (+4.7) |
| OLMo-7B | 56.8 \| 30.9 | 56.9 \| 28.4 | **60.9** (+4.1) \| **37.5** (+6.6) |

- **Hybrid wins on every architecture.**
- Family-dependent: **Qwen is more steerable** and favors tokens over instructions; **Llama is much less steerable** regardless of method — but Hybrid "rescues" weak models (+12.7% for Llama-8B on 3-behavior).

### 5.3 Scaling (Table 3, Qwen family)
- 3-behavior unseen with steering tokens (trained on 2-behavior only): **59.5% at 8B → 68.0% at 14B (+8.5%)**; instructions **52.1% → 61.4% (+9.3%)**. Both benefit from scale.
- Hybrid at 14B: **69.2%** with order variance down to **6.2%** (vs 18.6% at 4B).
- **Training on 2+3-behavior compositions *degrades* performance at 14B**: 63.9% vs 68.0% for 2-only, and higher order variance (17.3 vs 13.9). At 8B it gives marginal variance reduction (21.2 → 15.5) at comparable accuracy.
  → Evidence that a **general composition operator is already learned from 2-behavior data**, and that larger models learn compositional patterns from simpler examples. Training-data efficiency is scale-dependent.

### 5.4 Ablation — `<and>` init and orthogonality (Table 4, Qwen-4B)

| `<and>` init | `L_orth` | Seen ↑ | Unseen ↑ | Ord.Var ↓ |
|---|---|---|---|---|
| No `<and>` token | — | 73.6 | 49.7 | 27.0 |
| Zero vector | ✗ | **94.5** | 66.9 | 11.2 |
| Zero vector | ✓ | 93.7 (−0.8) | **69.1** (+2.2) | **5.3** (−5.9) |
| "and" embedding | ✗ | 94.2 | 55.2 | 9.4 |
| "and" embedding | ✓ | 94.2 (+0.0) | **70.8 (+15.6)** | 7.1 (−2.3) |
| Avg. steering tokens | ✗ | 94.3 | 58.4 | 8.6 |
| Avg. steering tokens | ✓ | 93.5 (−0.8) | 64.4 (+6.0) | 6.2 (−2.4) |

- **`<and>` is essential:** without it, seen drops to 73.6 (vs 93–95), unseen to 49.7 (vs 64–71), order variance jumps to 27.0 (vs 5–11).
- **All inits give comparable *seen* performance (93–95%)** but *unseen* varies 55–71% → compositional generalization is the true differentiator, and it does not depend on how behaviors are learned.
- **Orthogonality is critical**, especially for semantic init: "and"-embedding init gains **+15.6%** unseen with it. For zero init it mainly reduces variance (11.2 → 5.3).
- **Token-average init benefits least** — averaging behavior embeddings is a poor starting point for composition.

### 5.5 Per-behavior breakdown (Fig. 2, Qwen-14B / Llama-8B)
- Gains for token steering are driven by **unseen cross-category** combinations, especially **title_case paired with language or length**.
- **Model-dependent failure modes:** Qwen-14B improves on most unseen combinations; Llama-8B excels on formatting compositions but **fails on the unseen length behavior (words_70_90)**, where instructions are superior.
- **Hybrid eliminates these failure modes** — the strongest practical argument in the paper.

### 5.6 Single-behavior performance (Table 7)
Individual steering tokens are, **alone, as effective as instruction steering** (e.g. Llama-3B all-tokens: text 92.0 vs. steering 93.4 vs. hybrid 95.5). This is explicitly contrasted with Stolfo et al.'s activation steers, which alone are *dramatically worse* than prompt-based steering.

---

## 6. Key Insights

1. **Composition is a learnable object.** A single `d`-dimensional `<and>` vector, trained only on 2-behavior pairs, generalizes to unseen behaviors, unseen combinations, *and* an unseen number of behaviors.
2. **Input space > activation space for composition.** Activation-space methods (LM-Steer, LoRA merging) are destructive when composed; input tokens compose through the model's own attention mechanism.
3. **Freezing behavior tokens in stage 2 is what makes it a composition operator** rather than a behavior-modifier.
4. **Orthogonality regularization is what prevents collapse** — without it, `<and>` drifts toward behavior representations and unseen performance craters (55.2% vs 70.8%).
5. **Implicit composition is not enough** — concatenation without `<and>` collapses on hard cases with extreme order sensitivity.
6. **Tokens and instructions are complementary, not competing.** Hybrid consistently wins, most dramatically on weaker models. This is the deployment recommendation.
7. **Evaluate the obvious baseline.** The paper's methodological critique — that most compositional-steering work skips the simple, very competitive "just put both instructions in the prompt" baseline — is worth carrying forward.
8. **More supervision can hurt.** Explicit 3-behavior training degrades 14B performance, indicating the 2-behavior operator already generalizes.

---

## 7. Limitations (authors')
- **Verifiable constraints only** (length, format, language, structure). Broader semantic constraints (tone, style) would need human annotation or LLM judges.
- **Up to 3 behaviors only.** Real applications may need many more; whether accuracy degrades gracefully or rapidly with property count is open.
- **3B–14B models only.** Frontier models exceed this; unclear whether gains continue or hit diminishing returns, and whether Hybrid remains necessary at scale.
