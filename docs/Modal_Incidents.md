# Modal incidents

## 2026-08-04 — official GCG runs cancelled after startup

**Impact:** Two detached upstream-reference GCG jobs stopped before their
requested step budgets. The native upstream log persisted partial controls,
losses, and tests, but did not contain sufficient optimizer state for an exact
resume.

**Observed behavior:** The Modal app transitioned to `stopped`; logs included a
cancellation signal while processing input. The local CLI’s successful detached
submission did not prove that the remote job would complete.

**Root cause:** The upstream runner logged results but lacked a checkpoint/resume
contract, explicit config fingerprint, signal-safe checkpoint handler, and
per-checkpoint Volume commit.

**Corrective actions:**

- use the `modal-experiments` skill for future launches;
- require checkpoint state and `fresh`/`resume` modes for long runs;
- verify a live metric/checkpoint after launch and after local client exit;
- run an intentional interruption/resume test before expensive experiments;
- pull and commit every completed or stopped run folder.

**Status:** Open until the five-behavior GCG runner passes the interruption test.
