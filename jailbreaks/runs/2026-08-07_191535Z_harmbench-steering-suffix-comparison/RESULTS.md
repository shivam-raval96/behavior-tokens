# HarmBench judge startup failure

Status: failed before judging.

The initial container lacked the SentencePiece tokenizer dependency. It wrote
only its resolved configuration and produced zero judgments. The corrected,
retry-safe replacement completed as
`2026-08-07_191629Z_harmbench-steering-suffix-comparison`.
