# Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?

**Authors:** Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, + 5 others  
**Affiliation:** Beihang University  
**Published:** September 8, 2025 (v1); revised January 27, 2026 (v2)  
**ArXiv:** [2509.06350](https://arxiv.org/abs/2509.06350)

---

## Problem Statement

All existing GCG variants (vanilla GCG, I-GCG, MAC, SM-GCG, etc.) use **fixed-length suffixes** and
treat all token positions uniformly — every position participates in gradient updates with equal weight.
But is this necessary? The paper asks: **do all tokens in an adversarial suffix actually contribute to the
jailbreak?** If not, can we identify and prune redundant positions?

---

## Core Finding

Not all suffix tokens are equally important. Empirical analysis shows:
- The **adversarial signal concentrates** on a **subset** of token positions.
- A **minority of low-impact positions** can be **aggressively pruned** without degrading attack success.
- Remaining shorter suffixes are **equally effective** and **less detectable**.

---

## Method: Mask-GCG

### Key Innovation: Learnable Token Masks

Mask-GCG introduces a **learnable binary mask** $m \in \{0, 1\}^l$ over the suffix positions of length $l$.

Each position $i$ has a mask value $m_i$ that determines:
- Whether position $i$ participates in gradient updates.
- Whether position $i$ is retained or pruned from the suffix.

### Mask Score Update

At each iteration, mask scores $\mathbf{s}$ are updated based on gradient magnitudes:

$$s_i \leftarrow \alpha \cdot s_i + (1-\alpha) \cdot \|\nabla_{\mathbf{e}_{s_i}} \mathcal{L}\|$$

Where:
- $\alpha$ is a momentum coefficient.
- $\|\nabla_{\mathbf{e}_{s_i}} \mathcal{L}\|$ is the gradient magnitude at position $i$ — a proxy for
  how much that token position "matters."

Positions with **high scores** (high gradient magnitude) are identified as high-impact;
positions with **low scores** are candidates for pruning.

### Pruning Strategy

- Pruning is applied at fixed intervals (e.g., every 10 iterations).
- The **lowest-scoring positions** are removed, shortening the suffix.
- A **rollback mechanism** restores the previous suffix if ASR or loss degrades after pruning.
- This prevents over-aggressive pruning from collapsing the attack.

### Algorithm Flow

```
Initialize: suffix s of length l, mask scores s_i = 0 for all i

For each iteration t:
  1. Compute gradient ∇L for all non-masked positions
  2. Update mask scores: s_i ← α·s_i + (1-α)·‖∇L_i‖
  3. Perform GCG token-swap using gradient from high-score positions only
     (low-score positions get lower sampling probability)
  4. If t % prune_interval == 0:
       a. Remove bottom-p% of positions by mask score → shorter suffix s'
       b. Evaluate loss/ASR with s'
       c. If no degradation: keep s' (commit prune)
          Else: rollback to s
  5. Update suffix tokens as in vanilla GCG

Return: optimized short suffix
```

### Key Properties

- **Plug-and-play**: Can be combined with any GCG variant (I-GCG, MAC, etc.) as a drop-in module.
- **Dynamic length**: Suffix length decreases over optimization (not fixed).
- **Gradient space reduction**: Pruning reduces the number of positions being optimized, which lowers
  computational cost per iteration.

### Hyperparameters

| Parameter | Value |
|---|---|
| Momentum $\alpha$ | 0.9 |
| Prune interval | Every 10 iterations |
| Prune fraction | 10–20% of remaining positions per prune step |
| Rollback threshold | Loss increase > ε |
| Initial suffix length $l$ | 20 (standard GCG default) |

---

## Datasets

- **AdvBench Harmful Behaviors** (Zou et al., 2023): Standard 520-behavior dataset.
- **HarmBench** (Mazeika et al., 2024): Broader standardized red-teaming benchmark used for
  cross-model and cross-attack evaluation.

---

## Models Evaluated

| Model | Type |
|---|---|
| Vicuna-7B-v1.5 | Open-source, white-box |
| LLaMA-2-7B-Chat | Open-source, white-box |
| LLaMA-3-8B-Instruct | Open-source, white-box |
| Mistral-7B-Instruct | Open-source, white-box |
| GPT-3.5-turbo | Closed-source, transfer only |
| GPT-4 | Closed-source, transfer only |

---

## Results

### Attack Success Rate vs. GCG

Mask-GCG maintains **comparable or higher ASR** to vanilla GCG while producing significantly shorter
suffixes.

| Model | GCG ASR | Mask-GCG ASR | Suffix Length Reduction |
|---|---|---|---|
| Vicuna-7B | ~97% | ~97% | ~40–50% |
| LLaMA-2-7B-Chat | ~57% | ~58% | ~40–50% |
| LLaMA-3-8B | ~80% | ~82% | ~35–45% |
| Mistral-7B | ~85% | ~86% | ~40% |

### Suffix Length Reduction

Mask-GCG consistently reduces suffix length by **40–50%** without sacrificing ASR, producing much more
compact adversarial sequences.

### Computational Efficiency

- Fewer gradient computations per iteration (pruned positions skipped).
- **Faster convergence** to first successful jailbreak.
- Total optimization time reduced by ~25–35% on typical setups.

### Stealthiness (Perplexity)

Shorter suffixes with fewer gibberish tokens → **lower perplexity** than full-length GCG suffixes.
Though still higher perplexity than natural text, the gap narrows substantially.

---

## Key Observations

1. **Token redundancy is real and significant.** A substantial fraction of suffix tokens contribute
   near-zero gradient signal — they are effectively noise. Pruning them does not hurt and often helps.

2. **Gradient magnitude is a reliable proxy for token importance.** Positions with consistently high
   gradient magnitudes across iterations are the load-bearing tokens for the jailbreak.

3. **Compact suffixes are more stealthy.** Shorter suffixes with fewer random-looking tokens are
   harder to detect by length-based or token-anomaly filters, even if perplexity remains elevated.

4. **Rollback prevents catastrophic pruning.** Without rollback, aggressive pruning can irreversibly
   collapse the attack. Rollback makes the algorithm safe to apply at any iteration.

5. **Pruning at fixed intervals (not continuous) is optimal.** Continuous masking destabilizes the
   optimization. Interval-based pruning gives the optimizer time to adapt before the next reduction.

6. **The finding generalizes across GCG variants.** Mask-GCG's insight (redundancy exists) applies
   to I-GCG, MAC, and others — it's a property of the optimization landscape, not of vanilla GCG
   specifically.

7. **Interpretability side-benefit.** By revealing which token positions matter most, Mask-GCG provides
   a tool for **understanding what makes adversarial suffixes work** — a step toward mechanistic
   interpretability of adversarial attacks.

---

## Implications for Defenses

- **Position-aware detection** may be more effective than perplexity filtering: if adversarial
  effects concentrate on specific positions, targeted monitoring of those positions could detect attacks.
- **Future alignment training** should focus on the specific positional and functional patterns that
  adversarial suffixes exploit.
- Shorter, more stealthy suffixes will **challenge simple length-based or perplexity-based filters**.

---

## Comparison to Other GCG Variants

| Variant | Core Change | Mask-GCG Relation |
|---|---|---|
| GCG (baseline) | Uniform token optimization, fixed length | Mask-GCG adds position weighting + pruning on top |
| I-GCG | Multi-coordinate updates, diverse targets | Mask-GCG compatible (can wrap I-GCG) |
| MAC | Temporal momentum on gradient estimates | Mask-GCG addresses spatial, not temporal, redundancy |
| SM-GCG | Multi-space gradient sampling | Orthogonal — addresses gradient noise, not redundancy |
| **Mask-GCG** | **Learnable position masking + pruning** | Addresses token redundancy; reduces suffix length |

---

## Limitations

- Still requires **white-box access** (gradients).
- The rollback mechanism introduces overhead and potential **non-monotonic suffix length trajectories**.
- Pruning heuristic (gradient magnitude threshold) may not be optimal for all model architectures.
- Stealthiness improvement is **relative**: suffixes remain detectable by perplexity filters, just
  somewhat less so than full-length GCG.

---

## Citation

```bibtex
@article{mu2025maskgcg,
  title={Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?},
  author={Junjie Mu and Zonghao Ying and Zhekui Fan and Zonglei Jing and others},
  journal={arXiv preprint arXiv:2509.06350},
  year={2025}
}
```
