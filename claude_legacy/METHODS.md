# METHODS.md

Model: `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of gated
`meta-llama/Llama-3.2-1B-Instruct`; 16 layers, hidden 2048, vocab 128256).

## 1. Contrastive steering vector

Dataset rows: same `prompt`, a concept response (`label +1`) and a neutral one
(`label −1`). Collect the residual-stream activation at `cfg.layer` (index into
`hidden_states[layer+1]`), pooled over the **response** tokens:

- `pooling: last` (default) — last response-token activation
- `pooling: mean` — mean over response tokens
- `pooling: attention` — attention-mass-weighted mean

Response span located by comparing the full chat template against the prompt-only
prefix. Steering vector (CAA / diff-of-means):

```
v = mean(A_pos) − mean(A_neg)          # A_* : [N, hidden]
v ← v / ‖v‖   if cfg.normalize          # unit vector; raw_norm reported separately
```

## 2. Concept probe

Logistic regression on the same pooled activations (`clf_C`, `clf_test_frac`,
`seed`). `test_acc` is the linear-decodability check. Reused later to score
generations and inside GCG eval.

## 3. Steering curve

For each `α` in `[curve_min, curve_max]` step `curve_step`: register a forward hook
adding `α·v` to the layer output at **every** position, generate (greedy,
`max_new_tokens`) on `eval_n_prompts` held-out prompts, re-encode each generation
and classify with the probe. Reports `concept_rate` (fraction classified positive)
and `mean_prob`. Resumable per scale; `_free_memory()` (mps/cuda cache clear) runs
between scales to avoid swap.

## 4. GCG suffix optimization (token_optimization)

Goal: a discrete suffix `s` (length `gcg_suffix_len`), inserted at the end of the
**user turn** (immediately before its `<|eot_id|>` — a system turn adds an earlier
eot, so use the **last** one), whose effect reproduces `α·v` (`α = gcg_target_scale`).

Each step (Greedy Coordinate Gradient):
1. gradient of the loss w.r.t. the suffix one-hot (`one_hot @ W_embed`, model frozen);
2. per position take the top-`gcg_topk` tokens by −grad;
3. sample `gcg_search_batch` random single-token substitutions, evaluate their true
   loss in one batched forward, keep the best.
Repeat `gcg_steps`. Resumable per step. `gcg_n_prompts` optimizes a shared suffix
over N prompts (transfer). `gcg_init_token` repeated = initial suffix.

### Objectives (`gcg_objective`)

- **`project`** (default for legacy): drive the shift's component along v to α:
  `loss = (⟨h_suffix − h_clean, v⟩ − α)²` at the last input token. Ignores off-v dims.
- **`match`**: full activation MSE `‖h_suffix − (h_clean + α·v)‖²`. Strict, weak.
- **`kl`**: match the **steered model's output distribution**. Teacher = model with
  `+α·v` added, its greedy continuation (`gcg_kl_tokens` steps) + per-step logits
  (computed once per prompt). Student = suffixed model teacher-forced on that
  continuation. `loss = mean_k KL(teacher_k ‖ student_k)`. Targets behavior directly.

### Behavioral eval (`evaluate_suffix`)

On held-out prompts, compute `concept_rate` for three conditions — **clean**,
**activation steering @α**, **GCG suffix** — by generating and classifying with the
probe. Saves per-prompt `transcripts.jsonl` (clean/steering/suffix side by side).
`proj` (mean ⟨Δ,v⟩) reported for reference (NOT the objective under `kl`).

## Key empirical facts (see EXPERIMENTS.md for numbers)

- Input suffix reproduces steering well **on-manifold** (moderate α, coherent text):
  sadness/rude ~0.95 of the steering effect.
- **proj caps ~1.4** regardless of suffix length/steps/target, and **anti-correlates**
  with behavior — the activation-match objectives game the proxy.
- **`kl` scales with compute; `project` degrades** (project raises proj with garbage
  tokens while behavior drops). At equal budget kl ≈ 2× project on power_seeking L10.
- **Short suffixes win / non-monotonic in length**: 1 token is robustly high because
  it's forced to pick one high-impact semantic token; long suffixes hide the shift in
  filler. High seed variance — report seed means.
- GCG is non-deterministic even at fixed seed (bf16 / hardware differences).
