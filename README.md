# behavior-tokens

Research workspace for studying whether discrete input tokens can reproduce
behavioral changes induced by activation steering vectors.

## Current status

This repository has been transitioned to a Codex-oriented layout. The active
workspaces are intentionally fresh:

- [`steering_vectors/`](steering_vectors/) — activation-steering experiments and
  supporting code.
- [`jailbreaks/`](jailbreaks/) — jailbreak experiments, method notes, and
  results.

The preceding implementation is preserved, unchanged, in
[`claude_legacy/`](claude_legacy/). It contains the former steering-vector,
token-optimization, and jailbreak packages. See
[`docs/Repo_Current_State.md`](docs/Repo_Current_State.md) before reusing or
migrating that work.

## Start here

- [`context.md`](context.md) — research question, terminology, and scope.
- [`claude_legacy/EXPERIMENTS.md`](claude_legacy/EXPERIMENTS.md) — historical
  experiment registry and findings.
- [`docs/Known_Issues_And_Followups.md`](docs/Known_Issues_And_Followups.md) —
  migration gaps and recommended next work.
- [`docs/related_works/`](docs/related_works/) — prior-work notes and paper
  references.
- [`AGENTS.md`](AGENTS.md) — workspace instructions for Codex agents.

## Layout

```text
steering_vectors/   active Codex workspace for steering-vector work
jailbreaks/         active Codex workspace for jailbreak work
claude_legacy/      preserved prior implementation, configs, outputs, and notes
docs/                current-state and follow-up documentation
docs/related_works/  prior-work notes and references
```

The legacy material is retained for reference; new work should be added to the
active folders unless a task explicitly calls for a historical change.
