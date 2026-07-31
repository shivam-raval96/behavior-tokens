# token_optimization/

GCG (Greedy Coordinate Gradient) to find a discrete **input suffix** whose
activations reproduce a steering vector's effect — hijacking behavior through the
input channel alone, no activation edit. Depends on `steering_vectors` (imports
config/model/data/evaluate + loads the vector & classifier). See [../CLAUDE.md](../CLAUDE.md).

## Modules

- [gcg.py](gcg.py) — `GCG.optimize`: insert an optimizable suffix at the end of the
  user turn (before the **last** `<|eot_id|>`); loss `‖⟨Δ,v⟩ − α‖²` (`project`,
  default) or full activation MSE (`match`). One-hot grad → top-k candidates →
  batched candidate eval → keep best. Model frozen; grad only on the suffix. Also
  `evaluate_suffix` (clean vs steering vs suffix) + `generate_with_suffix`.
- [checkpoint.py](checkpoint.py) — per-run folder `outputs/token_optimization/<label>/`,
  resume state, artifact, transcripts, and `write_results_md` (auto per-run summary).
  Re-exports vector/classifier loaders from `steering_vectors.checkpoint`.
- [plot.py](plot.py) — `plot_gcg` (loss + concept-rate bars), `plot_length_sweep`.
- [run.py](run.py) — one GCG run → its own labelled folder. `python -m token_optimization.run configs/<c>.yaml`.
- [sweep_len.py](sweep_len.py) — sweep suffix length → one `*_lensweep/` folder + length_sweep.png + results.md.
- [seed_sweep.py](seed_sweep.py) — variance across seeds → `*_seedsweep/` folder, one json + results.md.

Needs a completed `steering_vectors.run` first (loads its vector + classifier).

## Output layout — one folder per run

`outputs/token_optimization/<label>/` where label = `<concept>_L<layer>_a<scale>_len<suffix_len>[_seedN]`
(single run) or `<concept>_L<layer>_lensweep` / `_seedsweep`. Each holds
`artifact.json`, `transcripts.jsonl`, plot png(s), `results.md`, `run.log`,
`gcg_state.json` (resume). `results.md` is auto-written: config + suffix +
clean/steering/suffix rate table + interpretation.

## Findings (sadness, layer 8)

- Input suffix reproduces most of the steering effect via input tokens alone.
- **Short suffixes win / non-monotonic:** peak ~8 tok (single lucky run 0.95), but
  across 10 seeds L=1 (mean 0.70, std 0.17) beats L=8 (mean 0.44, std 0.28).
- **proj never reaches α (caps ~1.4) and anti-correlates with behavior** — the
  activation-match objective is a loose proxy; token *coherence* drives behavior.
- Probe fires on broad negative-affect, not the exact concept (L=1 `sarcast` → 0.82).
- GCG is non-deterministic even at fixed seed (bf16 / hardware) → report seed means.

(Historical pre-restructure runs archived under `outputs/token_optimization/<concept>_legacy/`.)
