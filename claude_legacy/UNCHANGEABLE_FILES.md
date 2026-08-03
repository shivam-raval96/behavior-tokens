# UNCHANGEABLE_FILES.md

Files/contracts an agent must NOT change without a very good reason. Breaking these
silently invalidates saved results or breaks resume/cloud runs.

## Hard contracts — do not modify

- **`steering_vectors/config.py` field NAMES.** `Config` is the shared contract
  between both packages, every YAML, and `modal_app.py`. Renaming/removing a field
  breaks all configs and cloud runs. *Adding* a new optional field with a default is
  OK; renaming/removing is not.
- **Checkpoint layout & filenames** (`steering_vectors/checkpoint.py`,
  `token_optimization/checkpoint.py`): `steering_vector.json`, `classifier.json`,
  `activations.pt`, `curve.jsonl`, `artifact.json`, GCG `run_label` scheme. Changing
  these orphans existing `outputs/` and breaks resume/`load_*`.
- **Output dir scheme:** `outputs/steering_vectors/...` and
  `outputs/token_optimization/<label>/`. Downstream tooling and pulls assume it.
- **Suffix-insertion logic** in `token_optimization/gcg.py` (`_split`: insert before
  the **last** `<|eot_id|>`). Getting this wrong silently corrupts every GCG run.
- **`modal_app.py` volume mounts/names** (`bt-outputs`, `bt-hf-cache`) and the
  two-package `add_local_dir` layout. Changing names loses access to stored results.

## Do not edit by hand (regenerate instead)

- Anything under `outputs/**` — vectors, `artifact.json`, `curve.jsonl`,
  `transcripts.jsonl`, plots, per-run `results.md`. These are produced by code;
  rerun the relevant task to change them.
- `outputs/token_optimization/*_legacy/` — archived pre-restructure runs; historical.

## Environment pins — change only deliberately

- `torch==2.5.1`, `transformers==4.49.0` (local venv). transformers 5.x needs torch
  ≥2.6 and broke here; 4.49 is known-good. Modal image pins the same.
- Model `unsloth/Llama-3.2-1B-Instruct` (ungated). The gated `meta-llama/...` needs
  an HF token; swap only if you have access.

## Safe to change

- `configs/*.yaml` (copy per experiment), new modules, new plots, docs (`*.md`
  except this file's contracts), `EXPERIMENTS.md` (append freely).
- `.gitignore` (keep `.venv/`, `*.pt`, `*.log`, `*_state.json`, `.modal.toml` ignored).
