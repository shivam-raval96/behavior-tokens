# Stopped run

The run stopped during paired generation at 23/25 behaviors. It generated and
persisted 23 complete baseline/suffix response pairs, made no judge calls, and
recorded zero model or API errors. ASR therefore cannot be reported.

## Diagnosis

The Modal logs record a cancellation signal at 2026-08-08 01:34:29 UTC while
processing the remote input, immediately after the local detached submitter was
interrupted. Although the CLI stated that the detached app would keep running,
the invocation used a local entrypoint whose outstanding `.remote()` call
remained the input being cancelled. This is a verified launch-lifecycle defect,
not an experimental or model failure.

The Volume checkpoint is internally consistent: phase `generation`, 23 rows,
configuration fingerprint
`c7b54c945ffd0ecc1d929f1fc83cbc66a5cd47f22fc87cb807e79b15bdfbdca9`, and
error count 0. `paired_generations.jsonl` also contains exactly 23 records.

Before any explicitly authorized resume, submit the remote function directly
instead of interrupting a local entrypoint, then verify that the checkpoint
advances beyond row 23 after the submitting process has exited. Resume must use
the identical configuration fingerprint and the existing run directory.
