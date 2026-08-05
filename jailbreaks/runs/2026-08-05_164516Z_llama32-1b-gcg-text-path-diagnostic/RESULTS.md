# Llama-3.2 reference-text GCG diagnostic

Status: failed at startup.

No candidate survived the reference decoded-text length filter after 16
batches. The implementation audit identified a missing pre-sampling mask for
non-printable and tokenizer-special control tokens; no optimization or ASR
evaluation ran.
