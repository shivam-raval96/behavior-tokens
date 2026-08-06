# MASK deception interruption checkpoint

- Run: `2026-08-06_080827Z_llama32-1b-layer10-mask-deception`
- Initial Modal app: `ap-mKIZdnubkOKoWsTN8jix1i`
- Purpose of stop: required interruption/resume validation.
- Judge calibration: 1.000 balanced accuracy on 300 references; 0% invalid.
- Durable candidate records: 128/2,904, balanced 64 honest and 64 deceptive.
- Checkpoint fingerprint:
  `eeb30f41d43ba11078e4352759a15732732d609133b76231c919d39f3fa349f9`.
- The checkpoint status is `running`, which is an allowed resume state. The
  experiment will resume with the identical fingerprint and must advance beyond
  candidate 128 without duplicate candidate IDs.
