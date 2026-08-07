"""Detached Modal entrypoint for steering-suffix experiments."""
from pathlib import Path
import modal

ROOT = "/workspace"
OUTPUT_ROOT = "/outputs/jailbreaks/runs"
app = modal.App("steering-suffix-optimization")
outputs = modal.Volume.from_name("bt-outputs", create_if_missing=True)
model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "transformers==4.53.2", "accelerate==1.9.0", "numpy", "pyyaml", "tqdm", "sentencepiece", "matplotlib")
    .add_local_dir("steering_suffix_optimization", f"{ROOT}/steering_suffix_optimization")
    .add_local_file("jailbreaks/llm-attacks-reference/data/advbench/harmful_behaviors.csv",
                    f"{ROOT}/harmful_behaviors.csv"))


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4,
              retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
              secrets=[modal.Secret.from_name("hf-llama-stage-a")],
              volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache})
def optimize(run_id: str, mode: str = "fresh", config_name: str = "jailbreak_pilot.yaml"):
    import os
    import sys
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.run import run
    return run(Path(f"{ROOT}/steering_suffix_optimization/configs/{config_name}"),
               Path(f"{ROOT}/harmful_behaviors.csv"), Path(OUTPUT_ROOT) / run_id,
               mode, commit=outputs.commit)


@app.local_entrypoint()
def main(run_id: str, mode: str = "fresh", config_name: str = "jailbreak_pilot.yaml"):
    optimize.remote(run_id, mode, config_name)


@app.function(image=image, gpu="A100", timeout=60 * 60 * 2,
              retries=modal.Retries(max_retries=1),
              secrets=[modal.Secret.from_name("hf-llama-stage-a")],
              volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache})
def judge_saved(run_id: str):
    import os, sys
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    from steering_suffix_optimization.harmbench_eval import run
    return run(Path(f"{ROOT}/steering_suffix_optimization/configs/harmbench_saved_generations.yaml"),
               Path(OUTPUT_ROOT), run_id, commit=outputs.commit)


@app.local_entrypoint(name="judge")
def judge_entrypoint(run_id: str):
    judge_saved.remote(run_id)


@app.function(image=image, gpu="A100", timeout=60*60*3,
              retries=modal.Retries(max_retries=1),
              secrets=[modal.Secret.from_name("hf-llama-stage-a")],
              volumes={"/outputs":outputs,"/root/.cache/huggingface":model_cache})
def positive_control(run_id: str):
    import os,sys
    os.chdir(ROOT); sys.path.insert(0,ROOT)
    from steering_suffix_optimization.positive_control_run import run
    return run(Path(f"{ROOT}/steering_suffix_optimization/configs/jailbreak_v2_response_position.yaml"),
               Path(f"{ROOT}/harmful_behaviors.csv"),Path(OUTPUT_ROOT),run_id,commit=outputs.commit)


@app.local_entrypoint(name="positive-control")
def positive_control_entrypoint(run_id: str):
    positive_control.remote(run_id)


@app.function(image=image, gpu="A100", timeout=60*60*4,
              retries=modal.Retries(max_retries=1),
              secrets=[modal.Secret.from_name("hf-llama-stage-a")],
              volumes={"/outputs":outputs,"/root/.cache/huggingface":model_cache})
def optimize_v2(run_id: str, mode: str="fresh"):
    import os,sys
    os.chdir(ROOT); sys.path.insert(0,ROOT)
    from steering_suffix_optimization.run_v2 import run
    checkpoint=Path(OUTPUT_ROOT)/run_id/"checkpoint.json"
    if mode=="fresh" and checkpoint.exists():
        mode="resume"
    return run(Path(f"{ROOT}/steering_suffix_optimization/configs/jailbreak_v2_response_position.yaml"),
               Path(f"{ROOT}/harmful_behaviors.csv"),Path(OUTPUT_ROOT),run_id,mode,commit=outputs.commit)


@app.local_entrypoint(name="optimize-v2")
def optimize_v2_entrypoint(run_id: str, mode: str="fresh"):
    optimize_v2.remote(run_id,mode)
