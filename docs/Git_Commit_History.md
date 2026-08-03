# Git commit history

This document preserves the useful repository-history guidance from the former
`GIT_COMMITS.md`. It is a historical reference; use `git log` for the current
source of truth.

## Repository and workflow

- Remote: `https://github.com/shivam-raval96/behavior-tokens.git` (`origin/main`).
- The project historically used `main` for its small, single-owner workflow.
- Commit code, configurations, documentation, and result summaries such as
  `results.md`, `*.json`, and `*.png`.
- Do not commit `.venv/`, `*.pt`, `*.log`, `*_state.json`, or `.modal.toml`;
  they are intentionally ignored as local or regenerable artifacts.
- Use `git add -A` so Git records directory moves as renames, then inspect the
  staged diff before committing.
- Push the main branch with `git push origin main` and confirm the reported
  commit range.

## Historical commits

| Commit | Summary |
| --- | --- |
| `ad7e999` | Initial commit (README stub). |
| `4d297fd` | Steering-vector and GCG behavior-token pipeline, including the original `steering/` package, Modal runner, and rude/sadness results. |
| `78bc9fc` | Split the implementation into `steering_vectors/` and `token_optimization/`; separated output layouts and added per-run `results.md`. |
| `e4d6f10` | Migrated the workspace to Codex, archiving the previous implementation and creating the active workspace/docs layout. |

## Historical uncommitted-work note

The prior record noted pending KL-objective work, power-seeking configurations
and outputs, and Claude-era guidance files. Those items were included in commit
`e4d6f10`; the original note is therefore no longer current.
