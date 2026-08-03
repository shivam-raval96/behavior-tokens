# Repository current state

_Updated: 2026-08-03_

## Active layout

| Path | Status | Purpose |
| --- | --- | --- |
| `steering_vectors/` | Active, fresh | New Codex-native steering-vector work. |
| `jailbreaks/` | Active, fresh | New Codex-native jailbreak work. |
| `claude_legacy/steering_vectors/` | Preserved | Former steering-vector package and its Claude instructions. |
| `claude_legacy/token_optimization/` | Preserved | Former GCG/token-optimization package. |
| `claude_legacy/jailbreaks/` | Preserved | Former jailbreak benchmarks, notes, and outputs. |
| `claude_legacy/configs/` | Preserved | Existing YAML experiment configurations. |
| `claude_legacy/outputs/comparisons/` | Preserved | Historical comparison plots. |
| `outputs/` | Historical supporting material | Existing artifacts, plots, and result summaries. |

## Documentation

`AGENTS.md` is the current agent entry point. `README.md` and `context.md`
describe the active workspace and research scope. The older `CLAUDE.md`,
`AGENT.md`, `METHODS.md`, and related top-level documents remain as historical
reference and may mention paths that are now under `claude_legacy/`.

## Important migration boundary

No implementation or configuration has yet been ported into the active Codex
workspaces. In particular, the root
`modal_app.py` and some historical documents still refer to the former package
paths. Treat those entry points as legacy until they are deliberately migrated.

## Source of experimental record

`EXPERIMENTS.md` remains the registry of prior findings. Its measurements and
links describe historical runs in `outputs/` and should not be interpreted as
results from the new active folders.
