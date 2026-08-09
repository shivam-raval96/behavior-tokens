# Suffix optimization algorithm

This directory is the iteration log and runnable home for methods that try to
replace an activation-steering intervention with a textual suffix. Each method
gets a numbered note, a frozen config, tests, and a unique dated run directory.

The first experiment is a fixed-rollout likelihood-gap test for the validated
jailbreak steering vector. It asks a deliberately narrow question: how much of
the steered teacher's continuation likelihood can a suffix recover when the
same model is run clean?

## Layout

- `METHOD_001_FIXED_ROLLOUT_CE.md`: exact method and masking convention.
- `EXPERIMENT_001.md`: preflight card and launch/closeout checklist.
- `configs/experiment_001.yaml`: frozen scientific configuration.
- `artifacts/`: copied vector plus provenance metadata.
- `fixed_rollout_ce.py`: cache, scoring, checkpoint, and reporting code.
- `modal_app.py`: detached Modal entrypoint.
- `tests/`: CPU-only tests for masking, CE aggregation, and suffix splicing.
- `runs/`: permanent, uniquely dated output folders (created at launch).

No run is launched by creating or editing this directory. After the experiment
card is approved with the word `proceed`, launch with a fresh UTC run ID.
Experiment 004 adds a discrete GCG search against the calibrated sampled-q
teacher-forced loss. See `EXPERIMENT_004_GCG.md` for its alignment gates,
placement convention, preflight, and warm/cold production configurations.
