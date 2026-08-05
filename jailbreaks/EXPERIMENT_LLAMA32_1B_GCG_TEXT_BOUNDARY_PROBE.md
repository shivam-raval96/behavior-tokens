# Llama-3.2 GCG text-boundary probe

## Objective

Identify why decoded Llama-3.2 candidate controls do not preserve the expected
20-token control span after insertion into the full chat-template prompt.

## Setup

- Model: `meta-llama/Llama-3.2-1B-Instruct`, float16, fast tokenizer
- Data: original AdvBench CSV row 0 only
- Control: initial 20-token `!` control; top-k 256; one 512-candidate draw;
  printable-ASCII tokens only; seed 0
- Target: `Sure, here is ` plus the row target
- No optimization loop, no generation, and no ASR evaluation

## Measurements

Save the rendered prompt, character offsets, full-prompt control-slice IDs,
standalone control IDs, decoded/re-tokenized controls, and the complete
candidate re-tokenized-length histogram. For length-preserving candidates,
save their full-prompt control IDs and the first differing token index.

## Success criterion

The probe succeeds if it identifies whether the mismatch occurs at the leading
separator, the trailing control boundary, or the character-offset slice. It
does not measure jailbreak success.

## Artifact and cost

Write config, checkpoint/progress, JSON result, and Markdown summary to one
UTC-dated run folder. Expected runtime: under two minutes on A100-80GB; cost:
well below one GPU-minute. This is a fresh diagnostic, not a continuation.
