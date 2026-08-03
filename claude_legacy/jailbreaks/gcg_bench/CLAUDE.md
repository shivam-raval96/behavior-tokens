# gcg_bench/

Faithful **Zou et al. 2023 GCG** (affirmative-target adversarial suffix) benchmarked
on Llama-3.2-1B across 3 HuggingFace jailbreak datasets. Metric: **ASR** by
refusal-string matching. Standalone (loads the model directly, does not import the
steering pipeline). Method note: [../methods/01_gcg.md](../methods/01_gcg.md).

## Modules

- [gcg_zou.py](gcg_zou.py) — `GCG.optimize(goal, target)`: suffix optimization to
  maximize `P(target | goal + suffix)`. one-hot grad → top-k swaps → retokenize
  filter → batched CE eval → keep argmin. Prompt surgery splits the rendered chat
  string on a sentinel so it is chat-template-agnostic. `GCG.generate` for scoring.
- [datasets_hf.py](datasets_hf.py) — `load(name, n)` → `[{goal, target}]`. Datasets:
  `advbench` (mlabonne mirror), `jbb` (JailbreakBench, real targets), `forbidden`
  (TrustAIRLab). Missing targets synthesized AdvBench-style ("Sure, here is …").
- [asr.py](asr.py) — refusal-prefix list + `asr()`. Success = generation is not a refusal.
- [benchmark.py](benchmark.py) — driver: per dataset × behavior → optimize, generate
  attacked + baseline, score → `outputs/gcg_zou/{results.json, asr.png}`.

## Run

```
# quick smoke (MPS/CPU, advbench, 2 behaviors, 25 steps) — validates plumbing
.venv/bin/python -m jailbreaks.gcg_bench.benchmark --quick --dtype float32 --device mps

# full run — prefer a CUDA GPU (Modal A10G). bfloat16 on cuda.
.venv/bin/python -m jailbreaks.gcg_bench.benchmark \
    --datasets advbench jbb forbidden --n_behaviors 25 --steps 500 --dtype bfloat16
```

## Notes

- GCG is O(steps × search_batch) forwards **per behavior** — heavy. Scale
  `n_behaviors`/`steps` to the compute. Local MPS is a smoke tool, not for full runs.
- walledai/{AdvBench,HarmBench} are gated; advbench uses the ungated mlabonne mirror
  (identical goals). Swap `path` + pass an HF token for the gated originals.
- ASR here is refusal-match (over-counts off-topic non-refusals). Pair with an
  LLM judge for headline numbers; refusal-match is the reproducible attack-mechanism baseline.
- Parent: [../CLAUDE.md](../CLAUDE.md) · results → [../RESULTS.md](../RESULTS.md).
