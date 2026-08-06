# Targeted-claim probe failure analysis

The deterministic compliance probe stopped the run before full candidate
generation. Only 3/32 tasks produced at least one judge-confirmed honest response
and one judge-confirmed deceptive response, versus the required 24/32.

## Observed behavior

- Honest condition: 14/64 honest, 16/64 deceptive, and 34/64 ambiguous.
- Deceptive condition: 13/64 deceptive, 9/64 honest, and 42/64 ambiguous.
- The exact target statement appeared in 4/64 honest responses and 0/64
  deceptive responses.
- Median response lengths were 9 honest tokens and 10 deceptive tokens; 71/128
  responses contained fewer than 15 generated tokens.
- The dominant outputs were short refusals such as `I can't fulfill this
  request.` The full augmented system prompt and required class target were
  present in every candidate artifact, so the failure is not missing prompt
  conditioning.
- Judge calibration remained perfect on 300 pinned references, with zero API,
  schema, or invalid-output errors. Candidate rationales correctly treated
  refusals and non-committal continuations as ambiguous under the MASK rubric.

## Implementation bug

Candidate generation and the causal sweep rendered the Llama chat template to
text and then called the tokenizer with its default `add_special_tokens=True`.
The rendered template already begins with BOS token `128000`, so the model input
began `[128000, 128000, 128006, ...]`. Direct chat-template tokenization begins
`[128000, 128006, ...]`. This duplicate-BOS path must be replaced by direct
batched `apply_chat_template(..., tokenize=True)` before another run.

## Conceptual mismatch

The MASK representation-engineering baseline does not construct its contrast
from model-generated, judge-filtered honest and deceptive responses. The paper
computes activation differences between honest- and dishonest-prompted versions
of the same stimulus. The official RepE honesty extraction template similarly
uses an honest/dishonest persona prompt around a common stimulus. Requiring a
1B instruction model to first comply with severe misinformation pressure was an
unnecessary dependency and caused most source tasks to be discarded before any
activation comparison.

The next proposed comparison follows that prompt-contrast construction at the
last prompt token, keeps positive orientation `deceptive - honest`, and validates
causal behavior on the unchanged disjoint MASK `provided_facts` split.
