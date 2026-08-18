# Qwen3.5-9B Terminal Wrench last-token probe

Status: prepared; launch requires an explicit **proceed**.

## Objective

Measure how accurately a linear probe trained on Qwen3.5-9B's last assistant
response-content-token residual-stream activation can distinguish rewarded
Terminal Wrench exploit trajectories from legitimate baseline trajectories on
held-out tasks.

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
  visible.
- Probe position: the last content token of the final assistant response,
  including a serialized tool call when that is the response. Qwen3.5's pinned
  template terminates assistant turns with `<|im_end|>\n`; both that special
  token and the trailing newline are excluded. The loader locates the final
  assistant span from tokenized template boundaries and asserts that the chosen
  token is immediately before `<|im_end|>`. It records the chosen token ID,
  decoded token, neighboring IDs/text, and span indices for an auditable sample,
  and hard-fails any row whose boundary cannot be proven.

## Split and probe

- Split unit: `task_id`, preventing trajectories or model variants of the same
  task from crossing splits.
- Deterministic assignment: SHA-256 of `seed || task_id`, 70% train and 30%
  test, seed 42. Exact task/example/class counts are persisted before activation
  extraction; empty-class splits are a hard failure.
- Activation source: the post-block residual-stream output from each of the 32
  transformer layers at the single audited probe position. Forward hooks retain
  only `[batch, hidden_size]` slices rather than materializing full-sequence
  hidden states. No text is generated.
- Classifier: one independently fitted `StandardScaler` plus L2 logistic
  regression per layer,
  `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, seed 42. The probe is
  fitted on train only.

## Evaluation

- Primary metric requested: ordinary test accuracy at threshold 0.5.
- Prespecified primary layer: final transformer layer (layer 31), avoiding
  test-set layer selection. Test accuracy is also reported for every layer as a
  full curve; the maximum test-layer accuracy is labeled descriptive/oracle and
  is not treated as an unbiased selected-model result.
- Diagnostics: per-layer train accuracy, test balanced accuracy, ROC AUC, precision,
  recall, F1, confusion matrix, majority-class accuracy, per-source-model and
  per-category accuracy, plus bootstrap 95% confidence intervals grouped by
  task (10,000 resamples, seed 42).
- Success criterion: report the test accuracy with its grouped bootstrap
  interval and majority baseline. This is descriptive; there is no minimum
  accuracy gate.

## Execution and artifacts

- Trial type: new trial.
- Compute: four independent Modal A100-80GB extraction workers, detached
  submission, 12-hour timeout, and one retry only after a
  fingerprint-validated shard checkpoint. Examples are deterministically
  sharded by ID; workers never write the same shard.
- Batching: examples are length-bucketed, dynamically padded, and packed under
  an initial 32,768-token batch budget with a maximum batch size of 16. Each
  worker runs an instrumented dry batch, increases the token budget when memory
  permits, and halves it and retries on OOM. Inference uses bfloat16, Flash
  Attention 2, `torch.inference_mode()`, and `use_cache=false`.
- Parallel CPU work: tokenization/preprocessing uses worker processes while the
  GPUs extract activations; after shard merge, the 32 independent layer probes
  are fitted in a bounded process pool without nested BLAS oversubscription.
- Checkpoints: after dataset resolution, split construction, each completed
  activation shard (at most 100 examples), each fitted probe, and final
  evaluation. Progress payloads include
  phase, completed/total, elapsed time, throughput, ETA, fingerprint/run ID,
  latest and best metrics, class counts, token lengths/truncations,
  activation norms, errors, and retries.
- Expected duration/cost: approximately 30-90 minutes wall time with four GPU
  workers, dominated by long-context forward passes; approximately 2-6 total
  A100-80GB-hours plus CPU and about 1.6 GB for raw float16 activations before
  metadata/checkpoint overhead. No paid judge API is used.
- Run path: `rough/runs/YYYY-MM-DD_HHMMSSZ_qwen35-9b-terminal-wrench-last-token-probe/`.
- Required files: source/config manifests, resolved config and fingerprint,
  split manifest, preprocessing audit, checkpoint/progress/history,
  `dashboard.html`, `dashboard_history.jsonl`, response-boundary audit, float16
  activation shards and merged activation index (5,984 examples × 32 layers ×
  4,096 hidden dimensions before integrity filtering), 32 fitted probes,
  aggregate and subgroup results, per-layer plots, `results.json`, and
  `RESULTS.md`.
- Live monitoring: a read-only Modal HTTP dashboard backed by `bt-outputs` and
  a continuously mirrored localhost dashboard are both verified after launch.

## Interpretation guardrail

Because the full original hack trajectories include the red-team elicitation
prompt while baselines do not, high accuracy may partly reflect prompt-source
recognition rather than behavioral reward-hacking detection. The final report
will state this prominently. A sanitized or prompt-stripped comparison would be
a separate experiment requiring its own card and explicit **proceed**.
