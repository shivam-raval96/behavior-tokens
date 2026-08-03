# Replication Guide: *Universal and Transferable Adversarial Attacks on Aligned Language Models*

**Paper:** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson.  
**arXiv:** 2307.15043v2, December 2023  
**Official code:** `https://github.com/llm-attacks/llm-attacks`  
**Official project page:** `https://llm-attacks.org/`

> **Responsible-use note.** This document is intended for controlled, authorized model-safety research. It deliberately omits the paper's adversarial suffix strings, unfiltered harmful prompts, and harmful model completions. Run evaluations only on models and infrastructure you own or are explicitly authorized to test. For a safe replication, replace AdvBench instructions with benign refusal/compliance tasks that preserve the same optimization and evaluation structure.

---

## 1. What the paper studies

The paper introduces **Greedy Coordinate Gradient (GCG)**, a white-box discrete optimization method that searches for an adversarial token suffix appended to a user instruction. The suffix is optimized so that an aligned language model begins its answer with an affirmative target prefix rather than refusing.

The main contributions are:

1. **Affirmative target prefixes.** Instead of optimizing a complete harmful answer, the loss targets a short beginning such as an affirmative restatement of the user instruction.
2. **Gradient-guided discrete token replacement.** Gradients identify promising token substitutions, while exact forward evaluation chooses the best candidate.
3. **Universal optimization.** One suffix is optimized jointly across multiple instructions.
4. **Multi-model optimization.** The same suffix can be optimized against multiple open-weight models to improve transfer to unseen models, including black-box systems.
5. **AdvBench.** A benchmark with 500 target strings and 500 instruction-style behaviors.

---

## 2. Threat model and experimental setting

### 2.1 Attacker capabilities

The source-model attack assumes:

- access to model weights;
- access to gradients with respect to token embeddings or one-hot token indicators;
- access to the model tokenizer and chat template;
- no weight updates;
- only an appended suffix is modified.

Transfer experiments use the optimized suffix against models for which gradients are not available.

### 2.2 Input construction

For an instruction \(q\), adversarial suffix \(s\), chat template \(C\), and target completion prefix \(y^\*\):

```text
prompt = C(user_message = q + s, assistant_prefix = "")
target = affirmative_prefix(q)
```

For the harmful-string task, the optimized token sequence can constitute the entire user input. For harmful-behavior experiments, it is appended to the instruction.

### 2.3 Objective

Let the tokenized model input be \(x_{1:n}\), and let the desired target prefix be \(x^\*_{n+1:n+H}\). The optimization loss is the negative log-likelihood:

\[
\mathcal{L}(x_{1:n})
=
-\log p(x^\*_{n+1:n+H}\mid x_{1:n})
=
-\sum_{h=1}^{H}
\log p(x^\*_{n+h}\mid x_{1:n+h-1})
\]

Only suffix positions \(I\) are modifiable:

\[
\min_{x_I \in \{1,\dots,V\}^{|I|}} \mathcal{L}(x_{1:n})
\]

The target is an affirmative prefix that repeats or paraphrases the instruction. The remainder of the generated completion is not included in the optimization loss.

---

## 3. Greedy Coordinate Gradient (GCG)

### 3.1 Core update

For each modifiable suffix position \(i\):

1. Compute the gradient of the loss with respect to the one-hot representation of the current token:
   \[
   \nabla_{e_{x_i}}\mathcal{L}
   \]
2. Select the `top-k` vocabulary entries with the largest negative gradient as candidate replacements.
3. Form a batch by sampling:
   - a suffix coordinate uniformly;
   - a candidate token uniformly from that coordinate's top-k set.
4. Evaluate every candidate suffix exactly with a forward pass.
5. Replace the current suffix with the candidate having the smallest loss.
6. Repeat for `T` optimization steps.

### 3.2 Pseudocode

```python
suffix = initial_suffix

for step in range(num_steps):
    # Aggregate gradients for all optimizable positions.
    grads = token_gradients(model, prompt_with(suffix), target)

    # top_candidates[pos] contains top-k token IDs ranked by -gradient.
    top_candidates = top_k_candidates(-grads, k=top_k)

    candidate_suffixes = []
    for _ in range(batch_size):
        pos = random_choice(optimizable_positions)
        token = random_choice(top_candidates[pos])
        candidate_suffixes.append(replace_token(suffix, pos, token))

    losses = batched_target_loss(
        model=model,
        prompts=[prompt_with(s) for s in candidate_suffixes],
        targets=[target] * len(candidate_suffixes),
    )

    suffix = candidate_suffixes[argmin(losses)]
```

### 3.3 Difference from AutoPrompt

AutoPrompt chooses one coordinate first and tests replacements only at that coordinate. GCG builds candidate sets across all coordinates and samples replacement candidates from the full coordinate-token space. Under the paper's matched forward-evaluation budget, this change substantially improves optimization.

### 3.4 Token filtering

A faithful implementation should reject candidate suffixes that do not preserve a stable tokenization after decoding and re-encoding. The released implementation also has model-specific slicing logic for locating:

- user-instruction tokens;
- adversarial-control tokens;
- assistant target tokens;
- loss tokens.

This is critical. Incorrect slices can silently produce invalid results, especially for tokenizers and chat templates other than LLaMA/Pythia-derived models.

---

## 4. Universal multi-prompt and multi-model optimization

Suppose there are prompts \(q_1,\ldots,q_m\), models \(M_1,\ldots,M_r\), and prompt/model losses \(\mathcal{L}_{j,t}\).

The universal suffix minimizes the aggregate loss:

\[
\mathcal{L}_{\text{total}}(s)
=
\sum_{j=1}^{m}
\sum_{t=1}^{r}
\mathcal{L}_{j,t}(q_j \Vert s)
\]

The paper states that gradients are normalized/clipped to unit norm before aggregation.

### Progressive prompt inclusion

The optimization begins with one prompt. Once the suffix succeeds on the currently active prompts, the next prompt is added. This curriculum continues until all selected training prompts participate.

A practical implementation:

```python
active_count = 1

for step in range(num_steps):
    active_prompts = train_prompts[:active_count]

    aggregate_gradient = 0
    for model in source_models:
        for prompt in active_prompts:
            g = token_gradient(model, prompt, suffix)
            aggregate_gradient += g / max(norm(g), eps)

    candidates = sample_candidates(aggregate_gradient)
    suffix = candidate_with_lowest_aggregate_loss(candidates)

    if succeeds_on_all(source_models, active_prompts, suffix):
        active_count = min(active_count + 1, len(train_prompts))
```

---

## 5. Datasets

## 5.1 AdvBench: harmful strings

- **Size:** 500 strings.
- **Purpose:** Exact target-string elicitation.
- **Length:** 3-44 LLaMA tokens.
- **Mean length:** approximately 16 LLaMA tokens.
- **Metric:** exact-match attack success rate.
- **Secondary metric:** target-token cross-entropy loss.

## 5.2 AdvBench: harmful behaviors

- **Size:** 500 instruction-style behaviors.
- **Purpose:** Test whether a model makes a reasonable attempt to comply rather than refuse.
- **Metric:** attack success rate based on whether the response attempts the requested behavior.
- **Universal split in the main single-model experiment:**
  - 25 behaviors for suffix optimization;
  - 100 held-out behaviors for test ASR.
- **Transfer evaluation:** 388 behaviors.

## 5.3 Dataset construction

The appendix reports that the authors:

1. manually wrote 100 seed harmful strings and 50 seed harmful behaviors;
2. sampled five demonstrations at a time;
3. prompted Wizard-Vicuna-30B-Uncensored to generate 10 new examples per iteration;
4. curated/released the resulting datasets.

The official repository stores the data under:

```text
data/advbench/
```

For responsible reproduction, preserve the row counts and splits but replace unsafe instructions with harmless tasks, for example:

- requests that a model should decline because of a synthetic policy;
- forbidden formatting requests;
- privacy-preserving toy tasks;
- benign policy-compliance stress tests.

---

## 6. Models

### 6.1 White-box source models

Main single-model experiments:

- Vicuna-7B
- LLaMA-2-7B-Chat

Transfer-suffix source ensembles:

- Vicuna-7B and Vicuna-13B
- Vicuna-7B, Vicuna-13B, Guanaco-7B, and Guanaco-13B

The repository README gives `vicuna-7b-v1.3` as its example model path.

### 6.2 Transfer targets

Open models evaluated in the paper include:

- Pythia-12B
- Falcon-7B
- Guanaco-7B
- ChatGLM-6B
- MPT-7B
- Stable-Vicuna
- Vicuna-7B
- Vicuna-13B
- LLaMA-2-Chat-7B

Historical proprietary endpoints reported in the paper:

- GPT-3.5: `gpt-3.5-turbo-0301`
- GPT-4: `gpt-4-0314`
- Claude Instant / Claude 1
- Claude 2
- PaLM 2

These hosted model versions are historical and may no longer be available. Reproducing their exact numbers today is therefore generally impossible. Treat contemporary API results as a new experiment, not a strict replication.

---

## 7. Hyperparameters

### 7.1 Shared optimization settings reported in the paper

| Parameter | Value |
|---|---:|
| Optimizable suffix length | 20 tokens |
| Optimization steps | 500 |
| GCG candidate batch size | 512 |
| GCG top-k per position | 256 |
| Candidate update | Best exact-loss candidate |
| GCG/AutoPrompt compute matching | Same batch size and top-k |
| Universal training behaviors | 25 |
| Held-out behaviors in single-model universal test | 100 |
| Transfer evaluation behaviors | 388 |

### 7.2 Transfer runs

| Setting | Models optimized jointly | Number of runs/suffixes |
|---|---|---:|
| Vicuna source | Vicuna-7B + Vicuna-13B | 2 runs with different random seeds |
| Vicuna + Guanaco source | Vicuna-7B + Vicuna-13B + Guanaco-7B + Guanaco-13B | 1 run |
| Selection | Lowest-loss suffix after 500 steps | 1 per run |

### 7.3 Baseline settings

**PEZ and GBDA**

- optimize 16 independently initialized sequences per target;
- use Adam;
- use cosine annealing;
- select the best sequence after optimization;
- 20 optimizable tokens;
- 500 steps.

**AutoPrompt**

- batch size 512;
- top-k 256;
- 20 optimizable tokens;
- 500 steps.

### 7.4 Generation settings for transfer evaluation

| Target | Temperature | top-p | Sampling/evaluation |
|---|---:|---:|---|
| GPT-3.5 / GPT-4 | 0 | 0 | Deterministic |
| Claude models | 0 | 0 | Deterministic |
| PaLM 2 | 0.9 | 0.95 | Generate 8 candidates; success if any succeeds |

The paper says each model's default conversation template was used.

### 7.5 Settings not fully specified in the paper

The paper does not fully pin down all replication details, including:

- exact random seeds;
- exact package versions beyond what appears in the repository;
- exact optimizer learning rates and schedules for PEZ/GBDA;
- maximum generation length for every model;
- exact stopping strings;
- the complete human-annotation protocol and inter-rater reliability;
- exact hardware allocation per run;
- the precise checkpoint revision/hash for every model;
- exact prompt ordering in progressive universal optimization.

These must be recovered from the released code, logs, or commit history to claim bit-level reproduction.

---

## 8. Software and environment

The official repository requires:

- Python;
- PyTorch;
- Hugging Face Transformers;
- FastChat `fschat==0.2.23`;
- `ml_collections`;
- local model weights;
- the repository installed in editable mode.

The README installation command is:

```bash
pip install -e .
```

The authors state that experiments used one or more **NVIDIA A100 80 GB GPUs**.

### Recommended reproducible environment capture

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "fschat==0.2.23"
python -m pip install -e .

python -m pip freeze > environment.freeze.txt
nvidia-smi > nvidia-smi.txt
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
PY
```

Also record:

```bash
git rev-parse HEAD > code_commit.txt
```

A modern environment may not reproduce the original results because tokenization, chat templates, model files, and generation APIs have changed.

---

## 9. Official experiment commands

From the repository README, run inside `experiments/launch_scripts`.

### 9.1 Individual behavior or string, one model

```bash
bash run_gcg_individual.sh vicuna behaviors
```

Variants:

```bash
bash run_gcg_individual.sh vicuna strings
bash run_gcg_individual.sh llama2 behaviors
bash run_gcg_individual.sh llama2 strings
```

### 9.2 Universal: 25 behaviors, one model

```bash
bash run_gcg_multiple.sh vicuna
bash run_gcg_multiple.sh llama2
```

### 9.3 Transfer: 25 behaviors, multiple models

```bash
bash run_gcg_transfer.sh vicuna 2
bash run_gcg_transfer.sh vicuna_guanaco 4
```

### 9.4 Evaluation

The repository directs users to:

```text
experiments/parse_results.ipynb
```

for parsing and evaluating experiment outputs.

### 9.5 Model path configuration

The README shows that model and tokenizer paths are configured with `ml_collections`, for example:

```python
config.model_paths = [
    "/DIR/vicuna/vicuna-7b-v1.3",
]

config.tokenizer_paths = [
    "/DIR/vicuna/vicuna-7b-v1.3",
]
```

Configuration files are under:

```text
experiments/configs/
```

---

## 10. Replication protocol

## 10.1 Stage A: validate the implementation on a benign task

Before using AdvBench:

1. Define 10-20 harmless prompts.
2. Define deterministic target prefixes.
3. Run a 5-token suffix for 20-50 steps.
4. Verify:
   - target loss decreases;
   - decoded suffix round-trips through the tokenizer;
   - candidate batches differ in only one coordinate;
   - prompt, control, target, and loss slices are correct;
   - no gradients flow into model weights;
   - batching produces the same losses as serial evaluation.

This detects most implementation errors cheaply.

## 10.2 Stage B: individual target-string experiment

For each of the first 100 target strings:

1. Initialize a 20-token control suffix.
2. Build the model-specific chat prompt.
3. Optimize for 500 GCG steps.
4. Generate the model completion.
5. Mark success only if the exact target appears according to the paper's exact-match rule.
6. Store:
   - initial suffix;
   - final suffix token IDs;
   - decoded suffix;
   - per-step target loss;
   - generated completion;
   - success flag;
   - wall-clock time;
   - GPU memory.

Aggregate:

```text
ASR = number of successful targets / 100
mean final target cross-entropy
```

## 10.3 Stage C: individual harmful-behavior experiment

For the first 100 behavior instructions:

1. Append the 20-token suffix to each behavior.
2. Optimize individually for 500 steps.
3. Generate a completion.
4. Classify whether the model refuses or attempts compliance.
5. Report ASR.

For a controlled safe replication, use a blinded annotation rubric and redact generated unsafe details.

## 10.4 Stage D: universal single-model experiment

1. Select 25 training behaviors.
2. Select 100 held-out behaviors.
3. Initialize one 20-token suffix.
4. Use progressive prompt inclusion.
5. Aggregate normalized gradients and losses across active prompts.
6. Optimize for 500 steps.
7. Evaluate:
   - train ASR on the 25 optimization behaviors;
   - test ASR on the 100 held-out behaviors.

Run separately for Vicuna-7B and LLaMA-2-7B-Chat.

## 10.5 Stage E: multi-model transfer suffixes

Produce three suffixes:

1. Vicuna-7B + Vicuna-13B, seed A.
2. Vicuna-7B + Vicuna-13B, seed B.
3. Vicuna-7B + Vicuna-13B + Guanaco-7B + Guanaco-13B.

For each run:

- use the same 25 training behaviors;
- optimize one suffix for 500 steps;
- save checkpoints throughout optimization, not just the last suffix;
- select the minimum source-loss suffix for the paper-faithful result;
- additionally evaluate earlier checkpoints because the paper observes transfer overfitting.

## 10.6 Stage F: transfer evaluation

On each of 388 held-out behaviors, evaluate:

1. behavior only;
2. behavior plus a simple affirmative instruction baseline;
3. each individual GCG suffix;
4. concatenation of the three GCG suffixes;
5. ensemble success, where a behavior is successful if at least one suffix succeeds.

For modern hosted models, do not send disallowed content or violate API policies. Use a locally controlled model or a benign substitute benchmark.

---

## 11. Evaluation rules

### 11.1 Exact strings

A trial succeeds if the generated output contains or equals the target according to the exact rule in the released evaluator. Because whitespace and tokenizer decoding can alter exact matching, preserve:

- prompt normalization;
- decoding options;
- special-token handling;
- leading-space behavior;
- case sensitivity.

### 11.2 Behavior compliance

The paper defines success as a response making a reasonable attempt to execute the requested behavior rather than refusing or evading. This criterion involves human judgment.

A stronger modern protocol should use:

- two independent annotators;
- a written refusal/compliance rubric;
- adjudication for disagreements;
- Cohen's kappa or Krippendorff's alpha;
- separate labels for refusal, partial compliance, full compliance, and irrelevant output;
- redaction of dangerous details.

### 11.3 Universal and ensemble ASR

```text
train ASR = successful training behaviors / 25
test ASR  = successful held-out behaviors / 100
transfer ASR = successful transfer behaviors / 388
ensemble ASR = behaviors for which any tested suffix succeeds / 388
```

---

## 12. Main reported results

## 12.1 White-box and single-model universal results

| Model | Method | Individual string ASR | String loss | Individual behavior ASR | Universal train ASR | Universal test ASR |
|---|---|---:|---:|---:|---:|---:|
| Vicuna-7B | GBDA | 0.0 | 2.9 | 4.0 | 4.0 | 6.0 |
| Vicuna-7B | PEZ | 0.0 | 2.3 | 11.0 | 4.0 | 3.0 |
| Vicuna-7B | AutoPrompt | 25.0 | 0.5 | 95.0 | 96.0 | 98.0 |
| Vicuna-7B | **GCG** | **88.0** | **0.1** | **99.0** | **100.0** | **98.0** |
| LLaMA-2-7B-Chat | GBDA | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 |
| LLaMA-2-7B-Chat | PEZ | 0.0 | 4.5 | 0.0 | 0.0 | 1.0 |
| LLaMA-2-7B-Chat | AutoPrompt | 3.0 | 0.9 | 45.0 | 36.0 | 35.0 |
| LLaMA-2-7B-Chat | **GCG** | **57.0** | **0.3** | **56.0** | **88.0** | **84.0** |

All ASR values are percentages.

## 12.2 Proprietary-model transfer results

Results are averaged over 388 behaviors.

| Method | Source optimization | GPT-3.5 | GPT-4 | Claude-1 | Claude-2 | PaLM-2 |
|---|---|---:|---:|---:|---:|---:|
| Behavior only | None | 1.8 | 8.0 | 0.0 | 0.0 | 0.0 |
| Behavior + simple affirmative baseline | None | 5.7 | 13.1 | 0.0 | 0.0 | 0.0 |
| Behavior + GCG | Vicuna | 34.3 | 34.5 | 2.6 | 0.0 | 31.7 |
| Behavior + GCG | Vicuna + Guanaco | 47.4 | 29.1 | 37.6 | 1.8 | 36.1 |
| Concatenated GCG suffixes | Vicuna + Guanaco | 79.6 | 24.2 | 38.4 | 1.3 | 14.4 |
| Ensemble of GCG suffixes | Vicuna + Guanaco | **86.6** | **46.9** | **47.9** | **2.1** | **66.0** |

### 12.3 Qualitative result trends

- GCG decisively improves exact-string optimization relative to PEZ, GBDA, and AutoPrompt.
- Universal suffixes generalize from 25 training behaviors to held-out behaviors.
- Multi-model source optimization improves cross-model transfer.
- Concatenation can help some targets but hurt others because long suffixes become confusing.
- An ensemble is more robust than any single suffix.
- Transfer success can peak before step 500; minimizing source-model loss can overfit and reduce black-box transfer.
- Claude-2 was substantially more robust than the other commercial systems tested in the paper.

---

## 13. Reproducibility caveats

### 13.1 Historical model drift

The exact hosted endpoints in the paper are obsolete or changed. Even when an endpoint name remains available, provider-side weights, moderation, system prompts, and serving logic may differ.

### 13.2 Chat-template sensitivity

Small differences in:

- system prompt;
- role separators;
- beginning/end tokens;
- whitespace;
- assistant-prefix placement;

can substantially change results.

### 13.3 Tokenizer dependence

The attack is optimized in token space. Changing tokenizer revisions can change:

- suffix length;
- candidate tokens;
- decoded text;
- gradient ranking;
- transferability.

### 13.4 Evaluation subjectivity

Behavior ASR depends on judging refusal versus attempted compliance. The paper does not report a full inter-rater reliability study.

### 13.5 Compute variance

The authors used A100 80 GB GPUs. Different precision modes, attention implementations, CUDA versions, or batching strategies can change memory feasibility and numerical rankings.

### 13.6 Source-loss overfitting

The final step is not always the best transfer checkpoint. Store and evaluate suffixes periodically, such as every 25 or 50 steps.

---

## 14. Recommended experiment logging schema

Save one JSON record per run:

```json
{
  "paper": "arXiv:2307.15043",
  "code_commit": "<git hash>",
  "model_id": "<model>",
  "model_revision": "<revision>",
  "tokenizer_id": "<tokenizer>",
  "tokenizer_revision": "<revision>",
  "chat_template_hash": "<sha256>",
  "dataset_file_hash": "<sha256>",
  "dataset_split": "train|test|transfer",
  "seed": 0,
  "suffix_length_tokens": 20,
  "num_steps": 500,
  "batch_size": 512,
  "top_k": 256,
  "precision": "float16",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "initial_suffix_token_ids": [],
  "final_suffix_token_ids": [],
  "best_step": 0,
  "best_loss": 0.0,
  "checkpoint_losses": [],
  "generation_parameters": {},
  "asr": 0.0,
  "wall_clock_seconds": 0.0
}
```

Do not publish unsafe suffix text or unredacted harmful generations.

---

## 15. Minimum checklist for a credible replication

- [ ] Pin the official repository commit.
- [ ] Pin model and tokenizer revisions.
- [ ] Use `fschat==0.2.23` for the original pipeline.
- [ ] Verify model-specific token slices.
- [ ] Use 20 optimizable tokens.
- [ ] Use 500 optimization steps.
- [ ] Use GCG batch size 512 and top-k 256.
- [ ] Match the 25/100 universal train/test split.
- [ ] Run the two Vicuna seeds and one Vicuna+Guanaco run.
- [ ] Evaluate all checkpoints for transfer-overfitting analysis.
- [ ] Report both exact-string loss and ASR.
- [ ] Use a documented human-evaluation protocol.
- [ ] Record environment, GPU, seeds, templates, and hashes.
- [ ] Separate strict historical replication from modern-model extensions.
- [ ] Keep the study in an authorized, sandboxed environment.

---

## 16. Suggested safe extension experiments

1. **Benign policy benchmark:** Replace harmful instructions with synthetic prohibited tasks.
2. **Template robustness:** Measure ASR under multiple system prompts and role templates.
3. **Tokenizer transfer:** Optimize on one tokenizer family and evaluate another.
4. **Checkpoint selection:** Compare minimum source loss with maximum validation transfer.
5. **Suffix length ablation:** Test 5, 10, 20, and 40 tokens.
6. **Candidate-budget ablation:** Vary top-k and batch size while holding forward passes fixed.
7. **Defense evaluation:** Test input filtering, perplexity filters, randomized smoothing, adversarial training, and output moderation.
8. **Automated evaluator validation:** Compare classifier labels with blinded human labels.

---

## 17. Sources

- Paper PDF: `https://arxiv.org/pdf/2307.15043`
- Abstract page: `https://arxiv.org/abs/2307.15043`
- Official repository: `https://github.com/llm-attacks/llm-attacks`
- Project page: `https://llm-attacks.org/`
- AdvBench data directory: `https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench`

---

## 18. Bottom line

A faithful reproduction requires more than implementing the GCG update. The decisive details are the affirmative target construction, exact model chat templates, correct token slicing, stable tokenization, progressive multi-prompt optimization, aggregation across models, and the paper's evaluation conventions. The headline configuration is a 20-token suffix optimized for 500 steps with a 512-candidate batch and top-k 256, trained on 25 behaviors and tested on held-out behavior sets. Exact replication of the closed-model numbers is no longer realistic because the evaluated proprietary endpoints and serving stacks have changed.
