# Soft-suffix reachability

This directory implements the Phase-1 protocol for testing whether an activation-steered teacher can be reproduced by a soft suffix at the exact future discrete-suffix slot.

The run order is deliberately diagnostic:

1. C1 on one prompt, then five prompts (the natural-language instruction occupies the exact suffix slot, making the target reachable by construction).
2. C2 naive versus position-matched steered teachers.
3. C3 unsteered baseline normalization.
4. C4 matched-length random-token and natural-language suffixes.
5. The full alpha × suffix length × norm constraint × seed sweep.

Teacher continuations and top-2000 distributions are frozen per sweep cell. Teacher-forced batches contain all eight continuations for a prompt. The steering hook begins at user content and remains active through every continuation position. Student suffix vectors are inserted after instruction content and before the assistant header.

## Artifacts

Each run writes `config.yaml`, `resolved_config.json`, `checkpoint.json`, `progress.json`, `metrics.jsonl`, cached teacher targets, optimizer states, learned soft prompts, paired baseline/suffix behavior generations and judgments, `results.json`, `RESULTS.md`, and `normalized_kl.png`.

The checkpoint contract supports `fresh` and `resume`; scientific configuration fingerprints deliberately exclude only the execution mode. SIGTERM is converted into a checkpointed stop at the next 50-step diagnostic boundary.

## Launch

Do not launch without the repository-required experiment card and a subsequent explicit `proceed` from the user.

```bash
modal run --detach steering_suffix_optimization/modal_app.py \
  --run-id YYYY-MM-DD_HHMMSSZ_soft-suffix-reachability \
  --mode fresh
```

The default configuration is [`configs/jailbreak_reachability.yaml`](configs/jailbreak_reachability.yaml). Outputs live on `bt-outputs` under `/steering_suffix_optimization/runs/<run-id>` and must be pulled into `steering_suffix_optimization/runs/` at completion or controlled stop.
