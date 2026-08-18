# Qwen3.5-9B Terminal Wrench last-token probe

Status: prepared; launch requires an explicit **proceed**.

## Objective

Measure how accurately a linear probe trained on Qwen3.5-9B's final
non-padding-token residual-stream activation can distinguish rewarded Terminal
Wrench exploit trajectories from legitimate baseline trajectories on held-out
tasks.

## Model and source data

- Model: `Qwen/Qwen3.5-9B`, pinned to the Hugging Face revision resolved at
  launch and recorded in the run artifacts; bfloat16 inference, no model
  parameter updates.
- Dataset: `few-sh/terminal-wrench`, pinned to commit
  `d8a29613235a0ef56a8b70b3142626a533da28c2`.
- Population: all 3,632 rewarded hack trajectories (serious and non-serious)
  and all 2,352 successful legitimate baseline trajectories: 5,984 examples
  before integrity filtering. Attacker legitimate solves and unsuccessful
  attempts are excluded.
- Representation: the complete original trajectory serialized in chronological
  role/content order. This intentionally measures the full-information
  condition; the attack-elicitation prompt remains present and is identified as
  a possible shortcut in the report.
- Tokenization: model-native chat template when compatible, otherwise a
  deterministic role-delimited serialization recorded in the run. Maximum
  length 16,384 tokens with left truncation so the terminal outcome remains
  visible. The activation is taken at the final attended token.

## Split and probe

- Split unit: `task_id`, preventing trajectories or model variants of the same
  task from crossing splits.
- Deterministic assignment: SHA-256 of `seed || task_id`, 70% train, 15%
  validation, 15% test, seed 42. Exact task/example/class counts are persisted
  before activation extraction; empty-class splits are a hard failure.
- Activation sweep: embedding output plus every transformer block output.
- Classifier: per-layer `StandardScaler` plus L2 logistic regression,
  `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, seed 42. The probe is
  fitted on train only.
- Layer selection: highest validation accuracy; ties choose the shallower
  layer. The test split is evaluated exactly once with that frozen choice.

## Evaluation

- Primary metric requested: ordinary test accuracy at threshold 0.5.
- Diagnostics: validation accuracy, balanced accuracy, ROC AUC, precision,
  recall, F1, confusion matrix, majority-class accuracy, per-source-model and
  per-category accuracy, plus bootstrap 95% confidence intervals grouped by
  task (10,000 resamples, seed 42).
- Success criterion: report the test accuracy with its grouped bootstrap
  interval and majority baseline. This is descriptive; there is no minimum
  accuracy gate.

## Execution and artifacts

- Trial type: new trial.
- Compute: one Modal A100-80GB, detached submission, 12-hour timeout, one
  retry only after a fingerprint-validated checkpoint.
- Batch size: 1 initially, increased only after a recorded memory-safe dry
  batch; inference uses no gradients. Probe fitting is CPU-side after each
  layer's activations are complete.
- Checkpoints: after dataset resolution, split construction, each 100 extracted
  examples, every fitted layer, and final evaluation. Progress payloads include
  phase, completed/total, elapsed time, throughput, ETA, fingerprint/run ID,
  latest and best metrics/layer, class counts, token lengths/truncations,
  activation norms, errors, and retries.
- Expected duration/cost: approximately 1.5-4 hours, dominated by long-context
  forward passes; approximately 1.5-4 A100-80GB-hours plus negligible CPU and
  storage. No paid judge API is used.
- Run path: `rough/runs/YYYY-MM-DD_HHMMSSZ_qwen35-9b-terminal-wrench-last-token-probe/`.
- Required files: source/config manifests, resolved config and fingerprint,
  split manifest, preprocessing audit, checkpoint/progress/history,
  `dashboard.html`, `dashboard_history.jsonl`, per-layer metrics, selected
  probe, aggregate and subgroup results, plots, `results.json`, and
  `RESULTS.md`.
- Live monitoring: a read-only Modal HTTP dashboard backed by `bt-outputs` and
  a continuously mirrored localhost dashboard are both verified after launch.

## Interpretation guardrail

Because the full original hack trajectories include the red-team elicitation
prompt while baselines do not, high accuracy may partly reflect prompt-source
recognition rather than behavioral reward-hacking detection. The final report
will state this prominently. A sanitized or prompt-stripped comparison would be
a separate experiment requiring its own card and explicit **proceed**.
