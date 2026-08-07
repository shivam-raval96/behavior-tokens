# Stopped run record

- Optimization and generation model: `meta-llama/Llama-3.2-1B-Instruct`.
- Optimization completed: 200/200 steps.
- Saved generations: 75 total; 25 each for baseline, optimized suffix, and
  random suffix.
- The configured evaluator was `cais/HarmBench-Llama-2-13b-cls`. Its 75 saved
  labels are retained for provenance but are not the requested primary judge
  evaluation.
- The first closeout attempt exhausted a 40 GB A100 while reloading the local
  13B evaluator after all judgments had already been saved.
- A recovery skipped the completed judgments but stopped while plotting because
  the suffix contained an unescaped `$`, which Matplotlib interpreted as math
  text.
- No OpenAI API judgments are present in this folder. A separately authorized
  re-evaluation must score the frozen generations with the OpenAI judge.

The stopped checkpoint on the remote Volume still says `running/judge 75/75`
because both failures happened before terminal result serialization.
