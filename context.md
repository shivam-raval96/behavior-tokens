# Project context

## Research question

Can a short sequence of discrete input tokens produce behavior similar to that
caused by an activation steering vector? This project investigates the
relationship between input-space interventions and internal representation-space
interventions in language models.

## Core terms

- **Steering vector**: a direction added to a model's hidden activations to
  shift a target behavior or persona.
- **Behavior token**: a discrete token sequence optimized or selected to induce
  a related behavioral shift through the model input.
- **Steering curve**: the measured behavior rate across a range of steering
  scales.
- **GCG**: a gradient-guided discrete-token optimization method used in the
  historical experiments.

## Scope

The work is intended to characterize model behavior, measurement quality, and
potential safety implications. Experimental claims need both quantitative
metrics and qualitative inspection: a high classifier score alone can reflect
degenerate, terse, or otherwise invalid output.

## Working convention

The active Codex workspaces are `steering_vectors/` and `jailbreaks/`. Earlier
code and notes are archived under `claude_legacy/`; consult them as reference,
but do not alter them unless a task specifically requests a legacy change.

For recorded observations and outstanding research directions, see
[`claude_legacy/EXPERIMENTS.md`](claude_legacy/EXPERIMENTS.md) and
[`docs/Known_Issues_And_Followups.md`](docs/Known_Issues_And_Followups.md).
