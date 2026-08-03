# AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes

**Authors:** Zeyi Liao, Huan Sun  
**Affiliation:** Ohio State University (OSU NLP Group)  
**Published:** April 11, 2024  
**ArXiv:** [2404.07921](https://arxiv.org/abs/2404.07921)  
**Code:** [github.com/OSU-NLP-Group/AmpleGCG](https://github.com/OSU-NLP-Group/AmpleGCG)  
**Follow-up:** AmpleGCG-Plus ([2410.22143](https://arxiv.org/abs/2410.22143))

> ⚠️ The authors have **not publicly released** the trained AmpleGCG model due to safety/misuse concerns.
> The codebase and training data (millions of GCG-generated suffixes) are available.

---

## Problem Statement

GCG is powerful but fundamentally **wasteful**: during optimization, it generates hundreds of candidate
suffix sequences at each step and keeps only the one with the lowest loss. Many intermediate suffixes that
successfully jailbreak the target are **discarded**. Furthermore, each new query requires running GCG
from scratch — there is no **amortization** across queries.

The paper identifies two core problems with vanilla GCG:
1. **Missed successes**: Intermediate optimization steps often produce successful jailbreaks that are
   thrown away in favor of the lowest-loss candidate.
2. **No generalization**: Each harmful query requires an independent expensive optimization run.

---

## Core Idea

**Treat GCG optimization as a data collection process.** Run GCG many times across many harmful queries,
collect all *successful* intermediate suffixes, and use them as training data to fine-tune a **generative
model** that learns the distribution of effective adversarial suffixes conditioned on any harmful query.

At inference time: instead of running GCG, simply prompt AmpleGCG to instantly produce hundreds of
candidate suffixes for a new query.

---

## Method

### Phase 1: Data Collection via GCG Mining

1. Run GCG on a large set of harmful queries from AdvBench.
2. At each optimization step, **record every suffix** that successfully jailbreaks the target model
   (not just the final result).
3. Collect (harmful query, successful suffix) pairs as training data.

Key insight: GCG's intermediate successful steps are far more numerous than the final suffix it returns —
vanilla GCG's "lowest loss only" policy is unnecessarily restrictive.

### Phase 2: Training the Generative Model

- Fine-tune an **open-source seq2seq or decoder-only LLM** on the collected (query, suffix) pairs.
- Training objective: given a harmful query, generate a suffix that, when appended, induces compliance.
- The model learns the **distribution** of effective suffixes — not just one per query.

### Phase 3: Inference

At attack time:
1. Feed a new harmful query to AmpleGCG.
2. Sample **200 candidate suffixes** in ~4 seconds (batch generation, no gradient computation).
3. Append each suffix to the query and test against the target model.
4. Report success if any suffix succeeds (pass@k evaluation).

This converts the attack from an expensive optimization to a **cheap sampling problem**.

### Architecture

| Component | Details |
|---|---|
| Base model for AmpleGCG | Open-source LLM (LLaMA-2-based, exact size varies by experiment) |
| Training data | Millions of (query, successful suffix) pairs from GCG runs |
| Training approach | Supervised fine-tuning (SFT) on query→suffix pairs |
| Inference | Beam search + sampling to generate diverse suffixes |
| Generation speed | 200 suffixes per query in ~4 seconds |

---

## Datasets

- **AdvBench Harmful Behaviors**: 520 behaviors used for GCG data collection and evaluation.
- **Training data**: Millions of GCG-generated adversarial suffixes mined from optimization trajectories
  on Llama-2-7B-Chat and Vicuna-7B.
- **Evaluation**: 50-behavior subset (Chao et al. protocol) for cross-model testing.

---

## Models

### GCG Data Collection Targets (White-Box)

| Model | Notes |
|---|---|
| Vicuna-7B | Primary source of training data |
| LLaMA-2-7B-Chat | Primary source of training data |

### Transfer Evaluation Targets

| Model | Type |
|---|---|
| Vicuna-7B | Open-source |
| LLaMA-2-7B-Chat | Open-source |
| GPT-3.5-turbo | Closed-source |
| GPT-4 | Closed-source |

---

## Results

### Attack Success Rate (ASR) — Open-Source Models

| Method | Vicuna-7B ASR | LLaMA-2-7B-Chat ASR |
|---|---|---|
| GCG (vanilla) | ~97% | ~57% |
| **AmpleGCG** | **~100%** | **~100%** |

AmpleGCG achieves **near-100% ASR** on both open-source models, surpassing vanilla GCG.

### Transfer to Closed-Source Models

| Target Model | Method | ASR |
|---|---|---|
| GPT-3.5-turbo | GCG (transfer) | ~87% |
| **GPT-3.5-turbo** | **AmpleGCG** | **~99%** |
| GPT-4 | GCG (transfer) | ~47% |
| GPT-4 | AmpleGCG | Significant improvement |

### Speed Comparison

| Method | Time per query |
|---|---|
| GCG (500 steps) | 20–60 minutes (GPU) |
| **AmpleGCG (200 suffixes)** | **~4 seconds** |

Roughly **300–900x speedup** at inference time.

### AmpleGCG-Plus (Follow-up, 2024)

The follow-up paper introduces an enhanced version achieving:
- Better ASR in **fewer attempts**.
- Effective attacks on **GPT-4o** series at similar rates to GPT-4.
- Breaks **circuit breakers** defense — a robustness-focused safety mechanism.

---

## Key Observations

1. **GCG discards most of its successful work.** The intermediate optimization steps contain far more
   successful jailbreaks than vanilla GCG's single-suffix selection suggests. Exploiting these
   "missed" successes is the central contribution.

2. **Amortization dramatically reduces per-query cost.** The expensive work is done once at training
   time; inference is nearly free. This makes AmpleGCG dangerous at scale.

3. **Generalization is strong.** The trained model generalizes to harmful queries it has never seen
   during training, suggesting it has learned abstract properties of effective suffixes.

4. **Transfer to closed-source is excellent.** A 99% ASR on GPT-3.5 from a model trained on open-source
   data demonstrates that adversarial suffix properties are highly transferable across model families.

5. **Safety concern led to no public model release.** The authors determined that public release would
   pose too great a risk of misuse — a rare example of self-regulation in adversarial ML research.

6. **Pass@k evaluation reveals true capability.** By sampling 200 suffixes and reporting any success,
   AmpleGCG operates in a "pass@k" regime — more realistic than single-attempt evaluation, and harder
   to defend against.

7. **Circuit breakers defense is also vulnerable.** AmpleGCG-Plus broke the circuit breakers
   representation engineering defense, suggesting that learned generative attackers can adapt to
   diverse defenses.

---

## Comparison to GCG

| Dimension | GCG | AmpleGCG |
|---|---|---|
| Attack paradigm | Optimization (per query) | Sampling (amortized) |
| Inference time | 20–60 min/query | ~4 sec/query |
| White-box required at inference | Yes | No |
| Training required | No | Yes (heavy upfront) |
| ASR on LLaMA-2-7B-Chat | ~57–88% | ~100% |
| ASR on GPT-3.5 (transfer) | ~87% | ~99% |
| Suffix diversity | Low (single suffix) | High (200 diverse suffixes) |
| Scalability | Poor | Excellent |

---

## Limitations

- **Requires large-scale GCG training data**: Collecting millions of successful suffixes via GCG is
  expensive (requires many GPU-hours of GCG runs upfront).
- **Training dependency**: The model needs to be retrained if the target model family changes
  significantly.
- **Not generalizable to completely novel harmful categories** not represented in training data.
- **Model not publicly released**: Limits reproducibility and downstream research.
- **Pass@k evaluation is optimistic**: Real attackers may not always get 200 attempts.

---

## Citation

```bibtex
@article{liao2024amplegcg,
  title={AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs},
  author={Liao, Zeyi and Sun, Huan},
  journal={arXiv preprint arXiv:2404.07921},
  year={2024}
}
```
