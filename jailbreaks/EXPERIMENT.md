# Stage A — benign GCG validation

The editable experiment control file is
[`configs/stage_a_benign_llama2_7b_chat.yaml`](configs/stage_a_benign_llama2_7b_chat.yaml).
It contains the model, harmless prompts, optimizer parameters, run mode, and
output naming. Keep every prompt and target prefix harmless.

For each run, the runner creates:

`jailbreaks/runs/YYYY-MM-DD_stage-a-benign-llama2-7b-chat/`

That directory holds `results.json`, `RESULTS.md`, `checkpoint.json`, and
`progress.json`. Set `run_mode: resume` in the YAML to continue a stopped run;
use `fresh` to restart and overwrite that dated run. Set `run.date` to an
explicit `YYYY-MM-DD` when resuming a run from a prior date.
