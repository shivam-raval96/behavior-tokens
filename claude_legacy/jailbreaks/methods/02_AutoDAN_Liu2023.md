# AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models

**Authors:** Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao  
**Affiliation:** University of Wisconsin-Madison, USC, University of California Davis  
**Published:** October 3, 2023  
**ArXiv:** [2310.04451](https://arxiv.org/abs/2310.04451)  
**Venue:** ICLR 2024

---

## Problem Statement

GCG and similar token-level attacks produce **nonsensical, high-perplexity suffixes** that are trivially
detected by perplexity-based defenses. Manual jailbreak prompts (e.g., DAN) are stealthy and readable but
require significant human effort and are not automatically optimizable. The key challenge: can we
**automatically generate jailbreak prompts that are both effective and semantically meaningful** (i.e.,
low perplexity, human-readable)?

---

## Core Idea

Replace token-level gradient optimization with a **hierarchical genetic algorithm (HGA)** that operates on
human-readable prompt templates. Rather than optimizing over raw token sequences, AutoDAN evolves
complete prompt paragraphs, maintaining semantic coherence while searching for effective jailbreaks.

The harmful request is embedded inside a plausible-looking prompt structure.

---

## Method: Hierarchical Genetic Algorithm (HGA)

### Overview

The genetic algorithm treats prompt strings as "individuals" in a population, evolving them across
generations through selection, crossover, and mutation operations.

### Two-Level Hierarchy

AutoDAN operates at two levels of granularity:

**Paragraph Level (coarse):** Evolves full paragraph structures of the jailbreak prompt.  
**Sentence Level (fine):** Within each paragraph, evolves individual sentences.

This hierarchy helps maintain overall semantic integrity while allowing fine-grained variation.

### Genetic Operations

| Operation | Description |
|---|---|
| **Initialization** | Seed population with hand-crafted DAN-style prompts |
| **Fitness Scoring** | Use target LLM's loss (negative log-prob of affirmative response) as the fitness function — requires white-box access |
| **Selection** | Momentum-based scoring: $s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot \text{fitness}_t$ — maintains history to avoid noise |
| **Crossover** | Multi-point crossover at both paragraph and sentence boundaries |
| **Mutation** | LLM-assisted rephrasing: use a helper LLM to rephrase sentences while preserving meaning |
| **Replacement** | Replace worst-performing individuals with offspring |

### Key Innovations vs. Plain Genetic Algorithm

- **Momentum-based fitness scoring** stabilizes selection across noisy evaluations.
- **Hierarchical crossover** at multiple text granularities avoids semantic destruction.
- **LLM-assisted mutation** (using GPT-3.5) to rephrase rather than randomly permute tokens —
  preserves readability.

### Algorithm Flow

```
1. Initialize population from handcrafted DAN prompts
2. Repeat for G generations:
   a. Evaluate fitness of each individual against target LLM
   b. Select parents via momentum-weighted fitness
   c. Crossover: combine paragraph segments from two parents
   d. Mutate: rephrase sentences with helper LLM
   e. Replace weakest individuals with offspring
3. Return best-performing prompt
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Population size | 64 |
| Generations | 100 |
| Crossover rate | 0.5 |
| Mutation rate | 0.1 |
| Momentum $\beta$ | 0.9 |
| Helper LLM for mutation | GPT-3.5-turbo |

---

## Datasets

- **AdvBench Harmful Behaviors** (Zou et al., 2023): 520 harmful behaviors; subsets of varying sizes used.
- Evaluation focused on the 50-behavior subset standardized by Chao et al. (2023).

---

## Models

### White-Box Evaluation Targets

| Model | Notes |
|---|---|
| Vicuna-7B | Primary white-box target |
| Guanaco-7B | Secondary white-box target |
| LLaMA-2-7B-Chat | Hardest white-box target (most aligned) |

### Transfer / Black-Box Evaluation

| Model | Notes |
|---|---|
| GPT-3.5-turbo | Tested for transferability |
| GPT-4 | Tested for transferability |

---

## Results

### White-Box Attack Effectiveness and Stealthiness (AdvBench)

| Model | Method | ASR | Recheck ASR | Perplexity (PPL) |
|---|---|---|---|---|
| Vicuna-7B | Handcrafted DAN | 34.2% | 33.9% | 22.97 |
| Vicuna-7B | GCG | 97.1% | 87.5% | **1532.2** |
| Vicuna-7B | AutoDAN-GA | 97.3% | 95.0% | 37.5 |
| Vicuna-7B | **AutoDAN-HGA** | **97.7%** | 91.7% | 46.5 |
| Guanaco-7B | GCG | 98.1% | 97.5% | **458.6** |
| Guanaco-7B | **AutoDAN-HGA** | **98.5%** | 93.7% | 39.3 |
| LLaMA-2-7B-Chat | GCG | 45.4% | 43.1% | **1027.6** |
| LLaMA-2-7B-Chat | **AutoDAN-HGA** | **60.8%** | 65.6% | 54.4 |

Key finding: AutoDAN achieves **comparable or higher ASR** to GCG while maintaining **PPL ~40x lower**
than GCG, making it nearly invisible to perplexity-based defenses.

### Defense Bypass

AutoDAN explicitly defeats **perplexity-based filters** because its prompts read like natural text (PPL
comparable to human-written DAN prompts, ~40-55).

### Improvement over Handcrafted DAN

AutoDAN improved handcrafted DAN attack success by over **250%** on Vicuna-7B (34% → 98%).

---

## Key Observations

1. **The stealthiness-effectiveness trade-off can be broken.** Prior work assumed that effectiveness
   (high ASR) required nonsensical token sequences that were easily detected. AutoDAN shows both goals
   can be achieved simultaneously.

2. **Perplexity is a misleading defense metric.** Systems relying on perplexity filtering are vulnerable
   to semantically coherent adversarial prompts.

3. **Hierarchical structure matters.** AutoDAN-HGA outperforms plain genetic algorithm (AutoDAN-GA)
   especially on harder targets like LLaMA-2, because coarse-to-fine evolution preserves the overall
   prompt structure while allowing fine-grained variation.

4. **LLM-assisted mutation is critical.** Using a helper LLM to rephrase sentences maintains
   semantic integrity far better than random token mutation.

5. **LLaMA-2 is significantly harder.** Both GCG (45%) and AutoDAN (61%) struggle more on LLaMA-2
   than on Vicuna/Guanaco, but AutoDAN shows a larger *relative* improvement (+15 pp vs GCG's 45%).

6. **Momentum scoring reduces variance.** Without momentum, genetic algorithms have noisy fitness
   landscapes; momentum smooths selection and accelerates convergence.

7. **Cross-model transferability is strong.** AutoDAN prompts generated on open models transfer to
   GPT-3.5 with competitive success rates, owing to their semantic coherence.

---

## Comparison to GCG

| Dimension | GCG | AutoDAN |
|---|---|---|
| Optimization space | Token-level (discrete) | Sentence/paragraph-level (semantic) |
| Output readability | Gibberish (PPL ~1000+) | Human-readable (PPL ~40–55) |
| Perplexity defense bypass | ❌ Easily detected | ✅ Bypasses perplexity filters |
| White-box required | ✅ Yes | ✅ Yes (fitness evaluation) |
| Transferability | Moderate | Strong (semantic prompts generalize better) |
| Speed | Slow (500 gradient steps) | Slower (generational evolution + LLM calls) |
| ASR on LLaMA-2 | 45–88% | 61% |

---

## Limitations

- Requires **white-box access** for fitness evaluation (loss computation against target LLM).
- The **helper LLM mutation** (GPT-3.5) adds API costs and latency.
- Slower than GCG in terms of wall-clock time due to generational iteration + LLM mutation calls.
- The initialization depends on hand-crafted DAN prompts — performance may degrade if the starting
  population doesn't include strong seed prompts.
- **Not fully automated** end-to-end (initialization requires human-crafted seeds).

---

## Citation

```bibtex
@article{liu2023autodan,
  title={AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models},
  author={Xiaogeng Liu and Nan Xu and Muhao Chen and Chaowei Xiao},
  journal={arXiv preprint arXiv:2310.04451},
  year={2023}
}
```
