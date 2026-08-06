# Failure analysis

The run stopped at the intended pair-quality gate. It generated and judged all
2,904 candidates, but retained only 51/585 training pairs and 12/141 held-out
pairs, versus required minima of 450 and 100. No activation direction was
computed.

The OpenAI judge itself is not the likely failure source: it achieved 1.0
balanced accuracy on 300 pinned references, returned zero invalid outputs, and
encountered zero API or schema errors. Its candidate rationales consistently
classified short refusals as ambiguous under the official MASK rubric.

The candidate conditions did not reliably control Llama-3.2-1B-Instruct. The
honest condition yielded 289/1,452 honest candidates; the deceptive condition
yielded 365/1,452 deceptive candidates. Most candidates were ambiguous, and the
median response lengths were 14 honest tokens and 9 deceptive tokens. The
original system instructions supplied the truth and requested honesty or
deception abstractly, but did not state the exact class-specific claim that the
response had to make.

The saved `system_prompt` field was also insufficient for diagnosis: it retained
the original MASK system prompt rather than the full augmented prompt actually
rendered for generation. Code inspection confirmed the augmented instruction was
used, so this was an artifact-observability bug rather than the cause of model
behavior.

The prepared comparison fixes the same contrastive-response method by computing
and supplying an exact required target claim for each class, strengthening the
direct-answer instruction, persisting the full generation system prompt, and
adding a 32-task/24-pair compliance gate before the full candidate workload.
