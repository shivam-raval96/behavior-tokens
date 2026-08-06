# Deception dataset audit — 2026-08-06

## Recommendation

Use Apollo Research's `deception-detection` Roleplaying dataset to construct the
next Llama-3.2-1B-Instruct layer-10 direction. It is the strongest inspected
source because every record contains the same scenario and question paired with
one curated honest completion and one curated strategically deceptive
completion. This isolates response behavior much better than independently
sampled truthful and deceptive text corpora.

Pinned source:

- Repository: <https://github.com/ApolloResearch/deception-detection>
- Commit: `f8ec4010e74927394709dffa22b97bdf8cd5a62f`
- Data: `data/roleplaying/dataset.yaml`
- Data SHA-256:
  `7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564`
- Official loader: `deception_detection/data/roleplaying.py`, SHA-256
  `68c497fd4fd033e5432f79852816131bab7b9a9d9e82a3b181da7e66d576ed21`
- Official grading template: `data/roleplaying/grading_template.txt`, SHA-256
  `8c24e39629a73dbebedb0d80c8e97fffddf7bc0d2e2830dc68e5a14dddfad151`
- Access note: `ACCESS.md` labels the repository public, but the pinned tree has
  no explicit LICENSE file. Treat reuse as research-only until licensing is
  clarified.

## Apollo Roleplaying schema audit

The pinned YAML was downloaded and parsed before selection.

- Rows: 371.
- Exact fields on every row: `scenario`, `question`, `answer_prefix`,
  `honest_completion`, and `deceptive_completion`.
- Missing required values: 0.
- Duplicate `(scenario, question)` pairs: 0.
- Identical honest/deceptive completions: 0.
- Unique honest completions: 371; unique deceptive completions: 371.
- Character ranges:
  - scenario: 177–630, median 381;
  - question: 25–230, median 86;
  - honest completion: 7–1,176, median 283;
  - deceptive completion: 7–1,056, median 272.
- Llama-3.2-1B-Instruct extraction sequences: 103–356 tokens, median 186;
  none exceed the 4,096-token guard.
- Causal prompts including the official answer prefix: 99–187 tokens; at most
  443 tokens after the configured 256-token generation allowance.

The official `offpolicy_train` loader constructs two examples per scenario with
an identical system scenario and user question, followed by either the honest
or deceptive completion. It does not include `answer_prefix` in the extraction
completion. The official on-policy evaluation loader adds `Answer the question
directly, without asides or internal thoughts.` to the scenario and starts the
assistant response with `answer_prefix`; the proposed causal evaluation follows
that behavior.

The scenarios represent instrumental deception rather than simple factual
errors. Audited examples include concealing the real reason for a missed
assignment, misrepresenting a product defect, hiding facts during an audit, and
giving a false explanation for an insurance claim.

## Other inspected candidates

### GAIR/BeHonest

Potential use: independent causal evaluation, not primary direction extraction.

- Revision: `fcdb24a2de2cd236bbd1b4484d99c23512ee7cc6`.
- The `Burglar_Deception` subset has four 50-row splits (200 total):
  `burglar_police`, `false_label`, `false_rec`, and `neutral`.
- Each row has `id`, `prompt`, `classification question`, `cheap object`, and
  `extremely expensive item`.
- Each classification rubric consistently maps A to a deceptive action, B to
  an honest action, and C to refusal.
- The dataset provides evaluation prompts and rubrics, but not paired honest and
  deceptive assistant completions. It is therefore better as an out-of-domain
  follow-up test after the Apollo vector passes its in-domain causal gate.

### PKU-Alignment/DeceptionBench

Potential use: later behavioral evaluation, not vector construction.

- Revision: `9c6c67b3d2a2d9c226ea0dd2120f1bb0899d779d`.
- Loaded file: `deceptionbench_0527.json`, SHA-256
  `1d5a97895bcb4b54d6a42d505ed56c66cc2e633fe06f5cb2953331d4457eb9b4`.
- 180 rows with complete fields: `id`, `type`, `inner_prompt`, `system_prompt`,
  `outer_prompt`, and `format`.
- Category counts: 45 goal-driven rule breaking/manipulation, 43 sycophantic
  misrepresentation, 43 honesty evasion under pressure, 33 sandbagging, and 16
  alignment faking.
- It contains scenarios rather than labeled paired completions, so extracting a
  direction would first require generating and judging pairs. That would
  reintroduce the failure mode avoided by selecting Apollo.

### dSLLab/llm-deception-trajectories prompts

Potential use: prompt-framing research only.

- Revision: `345ad28ea90399a48fcc03c7016ebe3c91f5bb55`.
- Loaded `data/prompts.parquet`, SHA-256
  `b76aa4cded3ee94caa30b07797d50da4103c79ad28685aa5911826092b5686e3`.
- 1,220 complete prompt pairs across 20 categories, with no duplicate or
  identical pairs.
- Many pairs change answer content, user presuppositions, or desired traits
  rather than holding a task fixed while changing honesty. Several audited
  labels are not semantically trustworthy as honest/deceptive labels. This is a
  strong prompt/content confound for a general deception direction.

### Solshine/deception-behavioral-multimodel

Rejected in its current pinned state despite being model-specific.

- Revision: `cbb4f524d0c20b134c6bd44ad6481b13946d933b`.
- Loaded `llama_1b_classifications.json`, SHA-256
  `60294a344324c17c0b43a5aacfa6240f88c62db55a09f0c66916b17a551e4cad`.
- The artifact identifies `unsloth/Llama-3.2-1B`, but its current metadata says
  270 ambiguous completions, 0 deceptive, 0 honest, and `n_usable: 0`.
- This conflicts with performance/count claims on the dataset card, so the
  current artifact cannot support a reproducible steering-vector comparison.

### dSLLab/FacadeBench

Potential use: alignment-faking evaluation, not direct direction extraction.

- Revision: `60ede786caaee0841e2d6487685fae41428650d4`.
- Loaded test file SHA-256:
  `b2efcb2ecb7f1b49a8cca3dff2a0805653a6420fa4177ec0d4224c4e4e9f07ee`.
- 720 train/validation/test scenario triplets contrasting training, deployment,
  and baseline contexts. It supplies prompts, expected behaviors, and safety
  constraints, but no paired honest/deceptive assistant completions.
- It is heavily cybersecurity/red-team framed and better suited to a separate
  alignment-faking direction or causal generalization test.

### DIFrauD

Potential use: deception-text classification controls, not behavioral steering.

- 95,854 labeled texts across phishing, fake news, political claims, product
  reviews, job scams, SMS, and rumors.
- Schema is only `text` plus binary `label`. Honest/deceptive samples are not
  matched by task or semantic content, so domain and style can dominate a mean
  activation difference.

### Counterfactual Deception Localization

Potential use: reasoning commitment analysis, not this Llama extraction.

- Five rule-labeled strategic environments and detailed prefix/continuation
  records.
- The repository is about 107 GB, contains traces from other model families,
  and its Hugging Face viewer currently fails schema extraction. It is not a
  clean source of paired Llama-3.2-1B assistant responses.

## Decision

Construct the next direction from Apollo Roleplaying, validate causally on a
disjoint Apollo scenario partition with the official grading rubric, and report
its cosine similarity with the already validated MASK deception vector. Reserve
BeHonest for a second, out-of-domain evaluation only after the first experiment
establishes that the new vector has an in-domain causal effect.
