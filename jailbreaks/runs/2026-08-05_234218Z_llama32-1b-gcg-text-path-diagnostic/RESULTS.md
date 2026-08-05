# Llama-3.2 reference-text GCG diagnostic

Status: failed at startup.

After masking non-printable and special control tokens, no candidate survived
the decoded-text length filter in 16 batches. No optimization step or ASR
evaluation ran. The next required check is a one-step chat-template boundary
probe that records the rendered control span, standalone suffix tokenization,
and candidate re-tokenized length distribution.
