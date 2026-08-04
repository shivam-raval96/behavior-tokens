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

## Experiment progress

Every experiment must expose live progress: use `tqdm` for bounded loops, print
structured metrics at a configurable interval, and write a partial result
checkpoint. When an experiment runs remotely, monitor those metrics and report
meaningful updates in this task before completion.
