# SUCCESS_CRITERIA.md

How to judge each stage. Numbers are guidelines from Llama-3.2-1B, layer 8-10.

## Steering vector + probe

- **Probe `test_acc` > 0.95** → concept is linearly decodable; proceed. 0.85-0.95 =
  usable but weaker vector. <0.85 → wrong layer / weak concept / bad labels; try
  another layer or check the dataset.
- **`raw_norm`** ~1-2 typical. Much larger norm ⇒ needs larger α to move behavior.

## Steering curve

- **Monotone in α**, low at α≤0, rising for α>0. Clean concepts saturate near 1.0.
- **Clean baseline (α=0) near 0** (e.g. 0.00-0.05) for free-text concepts. A baseline
  near **0.5 = the eval is broken** (usually an MC-format dataset scored on free text
  — the probe is near-chance). Fix the dataset/eval before trusting GCG numbers.
- If behavior only appears at very high α (knee at α≥7), that regime is likely
  **off-manifold/degenerate** (broken repetition). Don't target it.

## GCG suffix (the core claim)

Primary metric = **reproduction fraction** on held-out prompts:
```
frac = (suffix_rate − clean_rate) / (steering_rate − clean_rate)
```
- **Strong result:** suffix_rate within ~0.1 of steering_rate at a **coherent** α
  (e.g. sadness/rude: suffix 0.95 vs steering 0.82-0.99, frac ~0.8-1.0). Confirms the
  input channel reproduces steering.
- **Partial/negative:** suffix_rate ≈ clean_rate → the direction isn't token-reachable
  at that α (often because α is off-manifold, not because the method failed).
- **Sanity:** always read `transcripts.jsonl`. suffix_rate can be a probe artifact
  (curt refusals scored "rude"; broken repetition scored "power-seeking"). Coherent,
  on-topic generations in the target persona = a real result.

## Objective / method comparison

- A good objective **improves behavioral rate with more `gcg_steps`**. `kl` does;
  `project`/`match` plateau or degrade (proxy-gaming: proj ↑ while rate ↓). Prefer
  the objective whose *behavioral* number scales with compute.
- Report **seed means** (≥5-10 seeds), not single runs — GCG is high-variance.

## Red flags that invalidate a result

- Baseline (clean or α=0) ≈ 0.5 → MC-eval mismatch.
- Target α in the degenerate regime → suffix "reproduces" gibberish.
- Single-seed conclusions → variance can be 0.07-0.98 at fixed length.
- proj went up but rate went down → you optimized the proxy, not behavior.
