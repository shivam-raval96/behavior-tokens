# Known issues and follow-ups

## Migration issues

1. **No active implementation yet.** `steering_vectors/` and `jailbreaks/`
   currently contain only Codex instructions. Port or rebuild code there before
   treating either as runnable.
2. **Legacy import paths remain.** `modal_app.py` and historical command docs
   refer to `steering_vectors` and `token_optimization` as importable root
   packages. Those commands will not work from the new layout without a planned
   migration or compatibility layer.
3. **Top-level historical documentation is stale.** `CLAUDE.md`, `AGENT.md`,
   `METHODS.md`, `SKILLS.md`, and `UNCHANGEABLE_FILES.md` describe the previous
   package layout. Use `AGENTS.md` and the docs in this directory for the active
   workspace.
4. **Most artifacts have not moved.** Historical comparison plots are archived
   under `claude_legacy/outputs/comparisons/`, while the remaining `outputs/`
   material is still at the repository root. A future migration should decide
   whether to retain, archive, or associate it with a new active experiment
   structure.

## Research follow-ups from the experiment registry

1. Add a coherence gate (for example, repetition or perplexity checks) when
   selecting a steering scale.
2. Compare KL and projection-style objectives at a coherent steering scale.
3. Report seed means and variance for short token suffixes.
4. Use free-text behavioral evaluations where possible; multiple-choice output
   can yield an unreliable baseline.
5. Validate any claimed behavioral shift by reading generations as well as
   using automated scores.

## Recommended migration sequence

1. Define the first active experiment and its intended module boundary.
2. Port only the legacy components needed for that experiment into the matching
   active directory.
3. Update the runner, config paths, and output contract together.
4. Add a small reproducible smoke test before running a full experiment.
5. Record the migration and resulting experiment in `EXPERIMENTS.md`.
