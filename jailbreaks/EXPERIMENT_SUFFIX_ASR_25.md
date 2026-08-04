# Stage D — Single-suffix ASR on 25 unseen AdvBench behaviors

## Purpose

Measure the ASR of the completed one-behavior suffix on a larger, deterministic
sample before making claims about transfer. This is an evaluation-only run: it
does not update the suffix.

## Controls

The editable controls are in
[`configs/stage_d_suffix_asr_25_advbench.yaml`](configs/stage_d_suffix_asr_25_advbench.yaml).

- Source suffix: the completed Stage C one-behavior result.
- Dataset: 25 AdvBench behaviors selected with seed 0 after skipping the two
  rows used by the source train/test run.
- Generation: greedy decoding, 128 new tokens.
- Metric: refusal-matching ASR for baseline and suffix-attacked completions.
- Persistence: every five behaviors saves progress, full response records, and
  an aggregate ASR checkpoint; `run_mode: resume` continues from that point.

## Expected artifacts

`jailbreaks/runs/YYYY-MM-DD_stage-d-single-suffix-asr-25-advbench/`

- `config.yaml` — immutable run configuration
- `checkpoint.json`, `progress.json` — resumable progress and metrics
- `results.json` — suffix, ASR aggregates, and full baseline/attacked responses
- `RESULTS.md` — concise evaluation summary

Launch durably with:

```sh
modal run --detach modal_app.py --task stage_d_suffix_asr
```
