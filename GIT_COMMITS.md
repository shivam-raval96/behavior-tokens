# GIT_COMMITS.md

Remote: `https://github.com/shivam-raval96/behavior-tokens.git` (origin/main).

## Workflow

- Commit code + configs + docs + result **summaries** (`results.md`, `*.json`,
  `*.png`). Do NOT commit `.venv/`, `*.pt`, `*.log`, `*_state.json`, `.modal.toml`
  (all in `.gitignore`).
- Work on `main` for this project (small, single-owner). For risky/large changes,
  branch first.
- Stage with `git add -A` (git detects file moves as renames). Before committing,
  verify no junk staged:
  ```bash
  git diff --cached --name-only | grep -E "\.venv/|\.pt$|\.log$" && echo "FIX" || echo ok
  ```
- **Commit only when the user asks.** Report results first; commit on request.

## Commit message format

Short imperative subject; body explains what + why. **End every commit body with:**
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

## Push

`git push origin main`. Confirm the range printed (e.g. `4d297fd..78bc9fc`).

## History

| commit | summary |
|--------|---------|
| `ad7e999` | Initial commit (README stub) |
| `4d297fd` | Steering-vector + GCG behavior-token pipeline (original single `steering/` package + Modal runner + rude/sadness results) |
| `78bc9fc` | Restructure into `steering_vectors/` + `token_optimization/` packages; outputs split into `outputs/steering_vectors/` and `outputs/token_optimization/<label>/`; per-run `results.md` |

## Not yet committed (as of writing)

- KL objective (`gcg_objective: kl`) in `token_optimization/gcg.py` +
  `gcg_kl_tokens` in config; objective-aware GCG run label.
- power_seeking configs + `outputs/` (L8 MC, L10 free-text, KL-vs-project comparison).
- These agent docs (`AGENT.md`, `METHODS.md`, `SUCCESS_CRITERIA.md`, `SKILLS.md`,
  `EXPERIMENTS.md`, `UNCHANGEABLE_FILES.md`, `GIT_COMMITS.md`).

Suggested next commit subject: `Add KL-to-steered GCG objective + agent docs`.
