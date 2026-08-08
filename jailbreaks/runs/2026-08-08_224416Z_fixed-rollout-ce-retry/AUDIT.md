# Closeout audit

## Integrity

- Terminal checkpoint: `complete`.
- Teacher records: 96 / 96; all `(behavior_id, rollout_index)` keys unique.
- Dataset rows: exactly 0–23, four rollouts per row.
- Continuation tokens: 12,073 in each scoring pass.
- Teacher cache SHA-256:
  `f6d1880270d43528c3030cee27a5c789f3e94d5083cde08f376079fa654a5870`;
  matches the manifest and result payload.
- Each floor and student token count equals its cached continuation length.
- The cache hash is unchanged after scoring.
- Dashboard history: 50 structured checkpoints from `initializing_model` to
  `complete`; the self-contained final dashboard is present.
- Errors/retries: 0 / 0.

## Independent metric recomputation

- Steered floor CE: 0.5975859991253869 nats/token.
- Clean empty-suffix student CE: 1.1951661834678882 nats/token.
- Student CE minus floor: 0.5975801843425013 nats/token.
- Record-mean diagnostic: floor 0.5930453642822067; student
  1.1941105681288946.

The recomputed token-weighted values match `results.json` to absolute tolerance
`1e-12`. The gap corresponds to a 1.8177x continuation-token perplexity ratio
for the empty clean student relative to the steered teacher floor.

## Implementation notes

The generation library warned that an attention mask could not be inferred
because pad and EOS IDs are equal. Every generation input was a single unpadded
sequence, so the effective all-ones mask is equivalent and no label or prefix
position was omitted. Continuation-only mask indexing is independently covered
by the CPU unit test that includes the first target-token boundary.

The run measures the empty-suffix baseline and fixed floor only. It does not yet
show that a textual suffix can close the gap; subsequent suffix candidates must
reuse the exact cache hash above for a comparable result.
