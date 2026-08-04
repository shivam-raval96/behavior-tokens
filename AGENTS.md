# Codex workspace

This is the active Codex-oriented workspace. New work belongs in:

- `steering_vectors/` for activation-steering experiments and supporting code.
- `jailbreaks/` for jailbreak experiments, methods, and results.

The previous Claude-era implementation has been preserved intact under
`claude_legacy/`:

- `claude_legacy/steering_vectors/`
- `claude_legacy/token_optimization/`
- `claude_legacy/jailbreaks/`
- `claude_legacy/configs/`
- `claude_legacy/outputs/`
- `claude_legacy/AGENT.md`, `CLAUDE.md`, and the historical experiment and
  methods documents

Do not modify the legacy directories unless a task explicitly targets historical
code or results.

## Version control

After completing a requested workspace change, stage the relevant non-ignored
files, create a concise conventional commit, and push it to `origin/main`.
Preserve tracked file moves as renames. Do not force-add ignored generated
artifacts (such as checkpoints and logs) unless the user explicitly requests it.

## Experiment artifacts

Every experiment run is a repository artifact. After a run completes or stops,
pull its dated output folder from the Modal `bt-outputs` Volume into
`jailbreaks/runs/`, including its configuration, checkpoints, progress metrics,
JSON results, Markdown summary, and any configured suffixes or full responses.
Name each run folder `YYYY-MM-DD_HHMMSS_description`, using the experiment's
start time and a concise lowercase-hyphenated description.
Stage the complete run folder, commit it, and push it to `origin/main`; do not
leave a completed or stopped run's only record on the remote Volume.

For every ASR evaluation that stores raw generations, save both the normal
no-suffix baseline response and the suffix-attacked response for each behavior,
with their respective success flags. Never report suffix ASR without retaining
the paired baseline needed to interpret the change.

## Experiment progress

## Experiment preflight

Before launching any experiment, first provide a concise experiment card in the
task. It must state: the objective; model and dataset/split; the exact source
artifact or configuration; optimizer/evaluation hyperparameters; success
metric; checkpoint/progress cadence; expected duration/cost; and whether it is
a new trial, a continuation, or a comparison. For a comparison, explicitly
name every material difference from the prior trial (including target prefixes,
random seed/data ordering, evaluation rows, and metric). Do not launch until
this card has been stated, unless the user explicitly asks to skip it.

The card must be complete enough to approve: include sample size/selection,
generation settings, every steering/optimizer setting, required source
artifacts, output files/plots, and the exact evaluation metric. After presenting
it, wait for an explicit user green signal containing **"proceed"** before any
submission. Do not treat “create,” “prepare,” “plan,” or “start” said before
the completed card as launch authorization.

Planning, implementing, committing, configuring, and presenting an experiment
card never authorize a launch. After the plan/card is complete, wait for the
user's explicit instruction to **proceed** before submitting any new local or
remote experiment. Treat a request to change code, add a configuration, or
estimate a run as preparation only. When a prior run already supplies the
needed model artifacts (for example a steering vector and selected layer),
reuse them for follow-up evaluation instead of re-running extraction or
optimization unless the user explicitly requests a fresh run.

Once an experiment is running, new information or a changed follow-up request
does not cancel it. Keep the active run going and plan any follow-up separately
unless the user explicitly says to stop, cancel, or pause that run.

Every experiment must expose live progress: use `tqdm` for bounded loops, print
structured metrics at a configurable interval, and write a partial result
checkpoint. When an experiment runs remotely, monitor those metrics and report
meaningful updates in this task before completion.

Progress records must be useful for diagnosis, not merely a percentage. At each
checkpoint, print and persist structured fields for: phase; completed and total
work; elapsed time and throughput; configuration fingerprint/run ID; latest
objective/metric; current best metric and its associated state (for example,
layer, suffix, or steering scale); and any error/retry count. Include
method-specific diagnostics such as loss components, class counts, activation
position, vector norm, or baseline-versus-intervention metrics as applicable.
Use the same fields in the human-readable progress update where they are
available.

For an active remote run, continue reporting progress proactively: inspect the
remote app and checkpoint at regular intervals, then share the current step,
loss/ASR where available, active-goal state, and a revised completion estimate.
Do not wait for the user to ask for every status update, and never substitute a
local-terminal message for remote evidence.

Create or maintain a completion monitor for every remote experiment. When a run
reaches a terminal state, it must notify this task, pull the complete run folder
from the Modal Volume into `jailbreaks/runs/`, commit and push it, and report a
concise final status (including result summary, found suffix where applicable,
and any failure or retry information). Do this without waiting for the user to
ask for completion status.

After launching a remote experiment, do not report it as running merely because
the launch command was accepted. Check its Modal application logs and, when
applicable, its output Volume until there is evidence that the worker started
and made progress (for example: model loading completed, a progress metric was
emitted, or a checkpoint was written). If it failed during provisioning or
startup, report the failure plainly, fix it when in scope, and verify the
replacement launch the same way.

For long-running remote experiments, treat the local client as expendable. Use
a detached Modal submission, persist checkpoint state to the output Volume at
each completed safe boundary, and verify the remote app remains active after
the local submission process has exited. After any cancellation, timeout, or
connection interruption, inspect the persisted checkpoint and app state before
resuming; never infer progress from the local terminal alone.
