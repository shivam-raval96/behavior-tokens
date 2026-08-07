"""Detached Modal entrypoint for the reachability protocol."""

from pathlib import Path

import modal

ROOT = "/workspace"
OUTPUT_ROOT = "/outputs/steering_suffix_optimization/runs"
app = modal.App("steering-soft-suffix-reachability")
outputs = modal.Volume.from_name("bt-outputs", create_if_missing=True)
model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.53.2",
        "accelerate==1.9.0",
        "numpy",
        "pyyaml",
        "tqdm",
        "sentencepiece",
        "matplotlib",
        "openai",
    )
    .add_local_dir(
        "steering_suffix_optimization", f"{ROOT}/steering_suffix_optimization"
    )
    .add_local_file(
        "jailbreaks/llm-attacks-reference/data/advbench/harmful_behaviors.csv",
        f"{ROOT}/harmful_behaviors.csv",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[
        modal.Secret.from_name("hf-llama-stage-a"),
        modal.Secret.from_name("openai-secret"),
    ],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def optimize(
    run_id: str, mode: str = "fresh", config_name: str = "jailbreak_reachability.yaml"
):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.experiment import run
    from steering_suffix_optimization.plot_results import plot
    from steering_suffix_optimization.report import write_summary

    run_dir = Path(OUTPUT_ROOT) / run_id
    # Modal retries repeat the original arguments. Once a committed checkpoint
    # exists, the retry must enter through the validated resume gate.
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    result = run(
        Path(ROOT) / "steering_suffix_optimization/configs" / config_name,
        run_dir,
        mode,
        commit=outputs.commit,
    )
    plot(run_dir)
    write_summary(run_dir)
    outputs.commit()
    return result


@app.local_entrypoint()
def main(
    run_id: str, mode: str = "fresh", config_name: str = "jailbreak_reachability.yaml"
):
    optimize.remote(run_id, mode, config_name)
