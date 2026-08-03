# Failures to Find Transferable Image Jailbreaks Between Vision-Language Models

**arXiv:** 2407.15211 (v1 Jul 2024, v2 Dec 2024) · Schaeffer, Valentine, Bailey, Chua, Eyzaguirre, Durante, Benton, Miranda, Sleight, T. Wang, Hughes, Agrawal, Sharma, Emmons, Koyejo, Perez (Stanford / Anthropic / Constellation et al.)
**Venue:** ICLR 2025; NeurIPS 2024 workshops — RBFM (**Best Paper**), Frontiers in AdvML (Oral), Red Teaming GenAI (Oral), SoLaR (Spotlight)
**Code:** https://github.com/RylanSchaeffer/AstraFellowship-When-Do-VLM-Image-Jailbreaks-Transfer
**Type:** Large-scale **negative result** on VLM adversarial transfer.

---

## 1. Headline Result

Gradient-based **universal image jailbreaks against VLMs are extremely difficult to transfer.** Optimizing an image against a single VLM — or an ensemble of 8 — always jailbreaks the attacked VLM(s) but shows **little-to-no transfer** to any other VLM. This holds regardless of shared vision backbones, shared language models, instruction-tuning, or safety alignment.

Transfer succeeds **only** when attacking large ensembles of VLMs that are "highly similar" to the target.

This contrasts sharply with (a) transferable *text* jailbreaks against LLMs (GCG/Zou et al.) and (b) transferable adversarial attacks against image classifiers — suggesting VLMs may be *more* robust to gradient-based transfer attacks, and that the added vision modality is a smaller attack-surface increase than expected.

---

## 2. Methodology

### 2.1 Target behavior
**"Harmful-yet-helpful"**: outputs that are both harmful *and* instrumentally useful toward the user's harmful goal. This is a deliberately stricter bar than toxicity or non-refusal.

### 2.2 Loss function
Given N prompt–response pairs, optimize a single image by minimizing negative log-likelihood that a set of **frozen** VLMs each produce the target response:

`L(Image) = −log Π_n Π_VLM p_VLM(nᵗʰ harmful-yet-helpful response | nᵗʰ harmful prompt, Image)`

(Standard in the VLM-robustness literature; some prior papers use different losses.)

### 2.3 Three text datasets (to vary the data distribution)
| Dataset | Size | Notes |
|---|---|---|
| **AdvBench** | 416 train / 104 test | Highly formulaic — target always "Sure, " + restatement of the prompt. Previously used for transferable *text* jailbreaks. |
| **Anthropic HHH** | 416 train / 104 test | Red-teaming subset, hand-modified into "unhinged" harmful-yet-helpful responses. Subsampled to match AdvBench. |
| **Generated** | **48k train / 12k test** | 51 curated harmful topics → 100 subtopics each (Claude 3 Opus) → 20 prompts per subtopic (Claude 3 Opus) → 20 responses per prompt (Llama-3-8B-Instruct) → Claude 3 Opus selects the most harmful → filtered. |

**Generated-dataset construction detail:** Claude refused to write harmful responses, so Llama-3-8B-Instruct was used (its safety training was "fairly easy to bypass with prompt engineering"). Automated filter: Claude 3 Opus rates harmfulness 1–5; discard anything not scoring 5 (discarding 20–70% per topic). Manual filter: search for caveat keywords, review every match, delete harm-mitigating text. The 51 topics span theft, drug crimes, fraud, violent crimes, cybercrime, terrorism, weapons offenses, deepfakes, CBRN-adjacent "sensitive information," etc., with per-topic item counts (≈51–1,643 each).

### 2.4 Attack hyperparameters
| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Momentum | 0.9 |
| Epsilon | 1e-4 |
| Weight decay | 1e-5 |
| Steps | **50,000** |
| Batch size | 2, ×4 gradient accumulation → **effective batch 8** |
| Image shape | (3, 512, 512) |
| Image init | uniform random noise in [0,1) **or** a natural image — **no difference found** |
| VLM parameters | all frozen |

### 2.5 Models — 40+ VLMs
Built on **Prismatic** (Karamcheti et al.), chosen because: (i) it provides dozens of VLMs varying vision backbone (CLIP ViT-L/14, SigLIP ViT-SO/14, DINOv2 ViT-L/14, DINOv2+SigLIP), language backbone, and finetuning data mixtures; (ii) it can easily be adapted to compute gradients w.r.t. input images (other repos require far more effort — note the repo README: Prismatic disables vision-backbone gradients by default and one line must be commented out); (iii) its training code is extensible.

The authors **built and publicly released 18 new VLMs** on HuggingFace: cross-product of **6 language backbones** (Gemma Instruct 2B, Gemma Instruct 8B, Llama-2 Chat 7B, Llama-3 Instruct 8B, Mistral Instruct v0.2 7B, Phi-3 Instruct 4B) × **3 vision backbones** (CLIP, SigLIP, DINOv2+SigLIP).

Other evaluated VLMs include DeepSeek-VL Chat 7B + SigLIP&SAM-B, LLaVA v1.5 7B/13B (Prismatic reproductions), Vicuna-based Prismatic models, Qwen-VL-Chat, and "Control" variants.

### 2.6 Four attack-success metrics — and why two were dropped
1. **Cross-Entropy Loss** on an eval split — fast, but only measures the *target* response, missing equally-harmful-but-different responses.
2. **LlamaGuard 2** (8B, Llama-3-based classifier)
3. **HarmBench Classifier** (13B, Llama-2-based)
4. **Claude 3 Opus** — prompted to describe helpfulness/harmfulness against a rubric, then give a Likert 1 (safe) – 5 (harmful-yet-helpful) rating, rescaled to [0,1].

**LlamaGuard 2 and HarmBench frequently disagreed with the authors' own judgments**, so results are reported using only **Cross-Entropy Loss and Claude 3 Opus**.

---

## 3. Results (six experimental settings)

| § | Setting | Transfer? |
|---|---|---|
| 3.1 | Optimized against a **single VLM** (30 attacked VLMs swept, × image init × text dataset) | **None.** Attacked VLM always jailbroken; zero transfer to any other VLM, regardless of shared vision backbone, shared LM, instruction-tuning, safety alignment, image init, or dataset. |
| 3.2 | Optimized against **ensembles of 8 VLMs** (3 different ensembles) | **None.** Every in-ensemble VLM jailbroken on held-out text data — and *as quickly* as when attacking single VLMs — but no VLM outside the ensemble. Ensembling did not help. |
| 3.3 | 4 **identically-initialized** VLMs (identical vision backbone, LM, MLP) trained on **overlapping data** (LLaVA v1.5 Instruct; +LVIS-Instruct-4V; +LRV-Instruct; +both) | **Partial / weak.** |
| 3.4 | Identically-initialized VLMs, **1-Stage vs 2-Stage** training | **None.** |
| 3.5 | **Different training checkpoints** of the same VLM (1 epoch attacked; tested on 1.25, 1.5, 2, 3 epochs) | **Partial**, decaying with more optimizer steps: 1.25 ≈ 1.5 > 2 > 3. |
| 3.6 | Ensembles of 1 → 2 → 8 **"highly similar"** VLMs (the 9 models from §3.3–3.5, each differing from `one-stage+7b` in exactly one detail) | **Yes.** For all targets but one, transfer improved monotonically with ensemble size; 8 achieved **strong** transfer. Still **no transfer to the 2-Stage VLM.** |

**Background on 1-Stage vs 2-Stage (Karamcheti et al.):** 2-Stage first finetunes the connector MLP alone, then the connector + language backbone together. 1-Stage finetunes both simultaneously and performs equally well.

---

## 4. Key Insight — the mechanism hypothesis

The 2-Stage failure is the most informative datapoint. Because 2-Stage holds the language model fixed during stage one, it can be viewed as **initializing the connecting MLP differently** from 1-Stage. That single difference kills transfer, while extra training data or extra epochs only degrade it partially.

> **The mechanism by which vision-backbone outputs are injected into the language model plays a critical role in transfer.**

Note the asymmetry: identical initialization + more data → partial transfer; identical initialization + different connector init → no transfer. Transfer appears to track connector geometry rather than backbone identity.

---

## 5. Critique of Prior Claimed Successes (Appendix B.1)

Niu et al. claim transferable image jailbreaks on AdvBench, contradicting this work and Bailey et al. The authors offer five conjectures:

1. **Different scoring.** This work requires positive evidence of harmful *and* helpful output; Niu et al. score success as "does not begin with a refusal string." Nonsense output counts as success under the latter.
2. **Different success criteria.** Niu et al. sum three success types, including "repetition or rephrasing of the harmful instruction, with less informative content." The classification procedure isn't stated and promised examples were not found.
3. **No baselines.** Niu et al. report no baseline refusal rates. The authors fed Niu et al.'s Figure-6 example prompt into LLaVA 1 **with no image at all** and got a nearly identical harmful response (Fig. 7) — the "jailbreak" image was doing nothing.
4. **Statistical significance.** Niu et al.'s ASR values differ by ±5%, while this work observes ±10% fluctuation across VLMs from sampling randomness alone, even without adversarial images. Niu et al.'s LLaVA-1 ASR (~25%) matches this work's *baseline* ASR.
5. **Image initialization.** Niu et al. initialize from a harmful image (e.g. a grenade) and prompt about a related topic; since their criteria count discussion of the *image* as success, this can artificially inflate ASR.

---

## 6. Broader Methodological Critique of the Field (§B)

The authors argue prior VLM-robustness work is patchwork along five dimensions:
- **Models:** overwhelmingly MiniGPT-4 / InstructBLIP / LLaVA, with overlapping, lower-performing, non-safety-aligned language backbones (FlanT5, OPT, Vicuna).
- **Methods:** inconsistent attacks, constraints, and text datasets; some papers misreport their own methodology in ways only discoverable by reading their code.
- **Behavior:** narrow harm types (usually toxicity); no check that harmful output is *instrumentally useful*.
- **Metrics:** missing baseline refusal rates; nebulously defined "ASR"; uncommon judges (e.g. Beaver-dam-7B).
- **Results:** direct contradictions between papers on whether transfer occurs.

They also note that safety alignment in the language backbone is unintentionally removed during VLM construction (finetuning safety-aligned LMs on benign data compromises safety — Qi et al.; also Bailey et al., Zong et al., Li et al.).

---

## 7. Future Directions (authors')
1. **Mechanistic understanding of VLM transfer resistance** — study activations/circuits, especially how visual and textual features are integrated. Do image-based and text-based attacks induce the same output distributions / exploit the same circuits?
2. **More transferable attacks** — compute limits prevented exploring more sophisticated optimizers; a different optimization process might change the findings.
3. **Detection of image jailbreaks** — given white-box access every studied VLM was easily jailbroken, so a robust defense needs runtime detection.
4. **More robust VLMs** — visual vulnerabilities persist regardless of language-backbone safety alignment.

---

## 8. Caveats
- **"Absence of evidence of successful transfer is not evidence of an absence of successful transfer."** The authors are explicit about this.
- Compute-constrained: single optimization recipe (Adam + NLL), not a search over attack algorithms.
- No mathematical definition of "highly similar" VLM is offered.
- Social-impact framing: the paper concludes multimodality poses a **surprisingly small** increase in attack surface via gradient transfer.
