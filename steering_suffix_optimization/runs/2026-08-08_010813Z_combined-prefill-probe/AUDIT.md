# Negative-result audit

## Conclusion

No verified implementation defect was found. The failed cosine gate is best
interpreted as a capacity/objective tradeoff for a five-token soft prefill, not
as evidence that prefill inversion is impossible. The position-aware KL term
fixed the previous run's dominant boundary failure, but the solution retained a
large activation component orthogonal to the desired steering delta.

## Causal-path checks

- Source identity: the resolved configuration points to run
  `2026-08-07_231417Z_prefill-inversion-probe`; the source soft prompt, teacher
  cache, steering vector, model revision, and configuration fingerprints are
  recorded with SHA-256 hashes.
- Input construction: the same prompt, five-token soft-prefill length, chat
  formatting path, continuation length (32), hidden-state index (10), and cached
  teacher target were reused. There is no token-count or boundary mismatch in
  the resolved configuration.
- Intervention target: the target projection is -3.352289 and the learned
  prefill reaches -2.346088. Its delta-norm ratio is 0.916250, showing that the
  failure is directional rather than merely insufficient norm.
- Objective and optimization: all three registered components are finite and
  improve coherently; the best objective occurs at the final step. There were
  200/200 completed steps and zero recorded errors, with checkpoint, optimizer
  state, soft prompt, and full metric history present.
- Forward measurement: position-0 KL falls from the source probe's 11.5665 to
  0.028899, mean KL to 0.087722, and later-position KL to 0.089620. This large,
  position-specific change is inconsistent with accidentally re-reporting the
  old probe or measuring only the activation intervention itself.
- Serialization: `results.json`, `progress.json`, `metrics.jsonl`, generations,
  configuration files, plot, checkpoints, and tensors agree on step 200 and the
  final metrics.

## Conceptual diagnosis

The projection penalty successfully increases the desired signed component, and
the weighted KL successfully controls the next-token boundary. However, cosine
only moves from 0.746204 to 0.763716. A scalar projection constraint cannot
remove the remaining orthogonal activation error. The five-token parameterization
may also lack enough degrees of freedom to satisfy both the full hidden-state
target and the teacher distribution over the continuation.

## Smallest useful follow-up

Run a localizable capacity/objective sweep on the same single prompt and cached
teacher target: suffix lengths 5 and 10 crossed with full-vector activation
weights 1 and 3, keeping the projection and position-weighted KL definitions,
seed, ordering, and 200-step budget fixed. Report the Pareto frontier of cosine
versus position-0 KL, not only a summed objective. Do not launch this follow-up
without a new experiment card and explicit `proceed` authorization.
