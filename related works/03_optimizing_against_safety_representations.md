# Optimizing Against Safety Representations: Activation-Guided Adversarial Suffixes and the Geometry of Refusal

**arXiv:** 2607.08883 (v1, 9 Jul 2026, cs.LG) · Ege Çakar, Hannah Guan, Kayden Kehe
**Venue:** AAAI 2026 Summer Symposium Series (Proc. AAAI Symposium Series 9(1), 27–34, DOI 10.1609/aaaiss.v9i1.42902). Earlier presented at the ICLR Re-Align workshop as *"Accelerating Adversarial Suffix Optimization via Continuous Relaxation and Activation-Guided Objectives."*
**Code:** https://github.com/Ege-Cakar/ImprovingGCG

> **Sourcing note:** the arXiv full text (HTML/PDF) was not retrievable at time of writing due to repeated fetch rate limits. This note is compiled from the abstract, the AAAI record, retrieved excerpts of the HTML, and the authors' code repository README — which documents the objectives, models, and evaluation stack in detail. Exact per-objective ASR tables and the full hyperparameter grid should be verified against the PDF.

---

## 1. Framing

Behavioral alignment **masks fragile internal safety representations**. If refusal is mediated by low-dimensional directions in activation space (Arditi et al.), then two questions follow: how are those representations *structured and localized*, and can *optimization* access them directly?

The paper uses adversarial suffix attacks as a **probe of representational alignment**, not merely as an attack. Two contributions:

1. **Activation-Guided GCG (AG-GCG)** — keeps GCG's greedy discrete suffix search but replaces the output log-likelihood objective with losses that directly suppress projections onto the learned refusal direction in the residual stream.
2. **Soft-GCG (SGCG)** — a continuous relaxation of discrete suffix optimization via Gumbel-Softmax, ~**33× faster** than standard GCG (the repo README claims 43× in its own experiments) with *improved* ASR.

---

## 2. Method A — Activation-Guided GCG

### 2.1 Setup
- Discrete greedy coordinate search over suffix tokens is **unchanged** from GCG.
- The **refusal direction** comes from the "Refusal Direction" pipeline (Arditi et al.), representing a one-dimensional safety subspace in the residual stream. It ships with `{layer, pos}` metadata identifying where the direction was extracted.
- The loss is swapped from "maximize P(affirmative target string)" to "minimize alignment with the refusal subspace."

A closely related formulation (Directional-GCG, seen in the follow-on literature) makes the shape explicit: with `r_l` the refusal direction at layer `l` and `h_l` the hidden state at the final token position for query + suffix,
`L = −cos(r_l, h_l)`
i.e. push the hidden representation away from the refusal direction. This is **less restrictive than target-string optimization** — it does not require matching a fixed affirmative response, just broad suppression of refusal-related alignment.

### 2.2 The five activation objectives
Exposed in code as `--activation-obj`:

| Objective | Scope |
|---|---|
| `negative` | negate the projection onto the refusal direction |
| `zero` (single-layer) | zero the projection at a single layer–position pair |
| `layer_zero_all` | zero across all layers (layer-wide) |
| `token_all_layers` | zero across all token positions and layers (token-wide) |
| `global_zero` | **global** — all layers and all positions |

Candidate suffixes are scored via `--activation-score-mode` ∈ {`global`, `local`, `token_all_layers`}.

### 2.3 Headline structural finding
**Suppressing refusal globally across all layers and positions beats targeting a single layer–position pair.**

→ Interpretation: **safety representations are distributed across the forward pass rather than causally localized to a single site.** This is a direct empirical qualification of the strong "single direction at a single site" reading of the refusal-direction literature — the direction may be one-dimensional, but its *causal footprint* is spread over the network.

Secondary finding: activation-based objectives achieve **higher attack success per optimization step** than standard GCG, i.e. mechanistic representation-level targets make discrete suffix search more **sample-efficient**.

---

## 3. Method B — Soft-GCG

### 3.1 Mechanism
- Optimize a **"soft" suffix** as a distribution over the vocabulary using **Gumbel-Softmax** (Jang et al. 2017), then **project back to discrete tokens via argmax** ("snapping back to hard tokens").
- Gumbel-Softmax: `y_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ)`, `g_i ~ Gumbel(0,1)`; τ → 0 approaches a true categorical draw.
- **Temperature annealing schedules**, including a three-phase **"slushy" schedule**.
- **Two loss options:** cross-entropy, or a **Carlini–Wagner-style** loss.
- **Variant 2** (`--2` / `--variant2` flag) = 3-phase temperature schedule + CW-style loss instead of pure CE.

### 3.2 Hybrid schedules
A sweep over the ratio of **Soft steps vs. discrete GCG steps**: pure GCG, pure Soft, 50/50, and heavy-Soft-warmup configurations. Results reported as mean ASR ± SEM per configuration, plus an efficiency curve trading off soft vs. discrete steps.

### 3.3 Speed
**33× speedup** over standard GCG (paper abstract) while *increasing* ASR. Repo states 43× with no performance decrease. Practical framing from the paper's impact statement: the attack moves from a resource-intensive operation to one **executable within minutes on consumer hardware**, and permits training **longer suffixes**.

---

## 4. Experimental Setup

**Models**
- Activation-Guided GCG: **LLaMA-2-7B-Chat** (`meta-llama/Llama-2-7b-chat-hf`), with the bundled Llama-2 refusal direction.
- Soft-GCG: the **Gemma 3 family** — 270M, 1B, 4B, 12B (27B skipped, see below). Chosen because (a) Ollama identifies Gemma 3 as the strongest family runnable on a single GPU, and (b) it comes from a major lab with wide adoption, making it a realistic local-deployment target.

**Data:** harmful/harmless prompt splits from the refusal-direction pipeline; AdvBench-style prompt lists for the Soft-GCG sweeps (`advbench_prompts.txt`).

**Evaluation stack (four channels):**
1. **Substring/keyword refusal matching** (refusal-substring list)
2. **LlamaGuard2** (via Together API)
3. **HarmBench** classifier
4. **Internal activation metrics** — cosine similarity and **Frobenius distance** between suffix-induced and refusal-direction-ablated activations

**Baselines run in-pipeline:** standard GCG suffix (`--run-gcg-baseline`), and a **refusal-direction ablation baseline** (hooks only, no suffix) (`--run-ablation-baseline`).

**GCG-side hyperparameters exposed:** `--n-train` (number of harmful train prompts for the universal suffix), `--n-steps` (discrete optimization steps), `--batch-size`, `--topk`, `--temp` (candidate sampling/evaluation), `--allow-non-ascii`, `--seed`, FastChat `--conversation-template`.

---

## 5. Results

### 5.1 Scale-dependent robustness (Gemma 3 family, Table 2 in paper)
- **Up to 34% ASR on Gemma 3 4B.**
- ASR **drops monotonically with model size**; the 12B variant showed negligible ASR, so **27B was not evaluated**.
- The 270M result is asterisked because the model **became incoherent** after optimization — a reminder that low-capability models can be "broken" rather than jailbroken, and that substring-matching ASR is inflated by incoherent output.
- Smaller models remain vulnerable; larger, better-safety-trained models resist **both** activation- and suffix-based attacks at the paper's compute-constrained settings.

### 5.2 Authors' interpretation of the scaling trend
> The drop in attack success with increasing model size suggests a **scaling-dependent change in how refusal is represented**. Larger models may encode safety constraints in **higher-rank or more distributed subspaces**, reducing the effectiveness of both localized activation suppression and smooth input-space optimization.

This dovetails with the global-vs-single-site finding in §2.3: if refusal is distributed, localized interventions weaken as the distribution widens with scale.

---

## 6. Key Insights

1. **Refusal is distributed, not point-localized.** Global (all layers, all positions) suppression outperforms single layer–position targeting. Treat the "single refusal direction" result as a statement about *dimensionality*, not about *localization*.
2. **Activation objectives are more sample-efficient than output objectives.** Targeting the representation directly beats forcing a fixed affirmative prefix, and does not require guessing a target string.
3. **Continuous relaxation is nearly free.** Gumbel-Softmax + argmax projection retains (or improves) ASR at ~1/33 the cost. Discrete combinatorial search over the vocabulary was apparently not buying much.
4. **Capability correlates with robustness** within a model family — but this is measured at compute-constrained settings, so it is a statement about attack budget as much as about the models.
5. **Evaluator choice matters.** The paper deliberately cross-checks substring matching against LlamaGuard2 and HarmBench, and adds internal activation metrics; the 270M incoherence case is exactly the failure mode substring matching cannot catch.
6. **Defensive framing:** the results argue for **representation-aware alignment** — spreading or raising the rank of safety encodings rather than relying on behavioral alignment over a low-dimensional refusal subspace.

---

## 7. Reproducibility Pointers (from repo)

```
# Activation-Guided GCG
python activation_pipeline.py \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --direction-path .../direction.pt --direction-meta .../direction_metadata.json \
  --activation-obj layer_zero_all \
  --run-gcg-baseline --run-ablation-baseline

# Re-score completions
python scripts/eval_safety.py --variants baseline,activation_gcg,gcg,ablation \
  --methods substring_matching,llamaguard2,harmbench

# Soft-GCG on Gemma 3
python run_sgcg_gemma.py --model gemma3:1b          # add --2 for Variant 2
python sweep_script.py                              # Soft vs GCG step-ratio sweep
python eval_sweep_bars.py --results_csv ... --prompts_file advbench_prompts.txt
```

Activation-distance analysis: `final_analysis/compute_activation_frobenius.py` (cosine + Frobenius stats between suffix-induced and ablated activations).
