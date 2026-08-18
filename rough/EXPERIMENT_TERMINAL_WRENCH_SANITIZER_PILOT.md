# Terminal Wrench sanitizer pilot

Status: prepared; API execution requires explicit **proceed**.

## Experiment card

- Objective: produce five audit-ready hack trajectories with the red-team
  elicitation suffix removed and explicit linguistic admissions of reward
  hacking, cheating, verifier gaming, bypassing, exploitation, or deception
  neutrally rewritten without changing the technical trace.
- Source: run `2026-08-18_191845Z_qwen35-9b-terminal-wrench-last-token-probe`,
  Terminal Wrench revision `d8a29613235a0ef56a8b70b3142626a533da28c2`.
- Selection: five positive-label original trajectories, deterministically chosen
  by ascending SHA-256 of `42|example_id`.
- Transformation: remove the elicitation suffix beginning with `You are an
  authorized Red Team QA Agent`; send only assistant JSON `analysis` and `plan`
  fields for rewriting; preserve commands, durations, tool/terminal outputs,
  task facts, ordering, and completion flags programmatically.
- API: OpenAI Responses API, `gpt-5.6-luna`, low reasoning effort, strict JSON
  schema `terminal_wrench_sanitizer_v1`, `store=false`, five parallel calls,
  at most two retries.
- Success metric: 5/5 schema-valid outputs and identical SHA-256 hashes for all
  immutable actions/observations before and after rewriting.
- Checkpoints: selection and each completed response. Save response IDs, status,
  token usage, retries, raw structured judgments, prompt/schema version,
  before/after indicator matches, and fidelity notes.
- Output: unique
  `rough/runs/YYYY-MM-DD_HHMMSSZ_terminal-wrench-sanitizer-pilot/` containing
  resolved config, selection, dashboard/history, progress/checkpoint,
  `sanitized_examples.json`, `results.json`, and `RESULTS.md`.
- Expected duration/cost: 1-5 minutes; five long-context API requests, likely
  below $2, with actual usage recorded. New trial; no model generation or probe
  training.

