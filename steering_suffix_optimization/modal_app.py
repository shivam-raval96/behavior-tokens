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
        "scipy",
        "fastapi[standard]",
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


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def prefill_probe(run_id: str, mode: str = "fresh"):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.prefill_probe import run

    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(
        Path(ROOT)
        / "steering_suffix_optimization/configs/prefill_inversion_probe.yaml",
        run_dir,
        mode,
        commit=outputs.commit,
    )


@app.local_entrypoint(name="prefill-probe")
def prefill_probe_entrypoint(run_id: str, mode: str = "fresh"):
    prefill_probe.remote(run_id, mode)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def combined_prefill_probe(run_id: str, mode: str = "fresh"):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.combined_prefill import run

    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(
        Path(ROOT) / "steering_suffix_optimization/configs/combined_prefill_probe.yaml",
        run_dir,
        mode,
        commit=outputs.commit,
    )


@app.local_entrypoint(name="combined-prefill-probe")
def combined_prefill_probe_entrypoint(run_id: str, mode: str = "fresh"):
    combined_prefill_probe.remote(run_id, mode)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[
        modal.Secret.from_name("hf-llama-stage-a"),
        modal.Secret.from_name("openai-secret"),
    ],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def prefill_asr_eval(run_id: str, mode: str = "fresh"):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.prefill_asr_eval import run

    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(
        Path(ROOT) / "steering_suffix_optimization/configs/prefill_asr_eval.yaml",
        run_dir,
        mode,
        commit=outputs.commit,
    )


@app.local_entrypoint(name="prefill-asr-eval")
def prefill_asr_eval_entrypoint(run_id: str, mode: str = "fresh"):
    prefill_asr_eval.remote(run_id, mode)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a"), modal.Secret.from_name("openai-secret")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def prefill_direct_diagnostic(run_id: str, mode: str = "fresh"):
    import os
    import sys
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.prefill_direct_diagnostic import run
    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(Path(ROOT) / "steering_suffix_optimization/configs/prefill_direct_diagnostic.yaml", run_dir, mode, commit=outputs.commit)


@app.function(
    image=image, gpu="A100-80GB", timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a"), modal.Secret.from_name("openai-secret")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def direct_steering_calibration(run_id: str, mode: str = "fresh"):
    import os, sys
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    from steering_suffix_optimization.direct_steering_calibration import run
    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists(): mode = "resume"
    return run(Path(ROOT) / "steering_suffix_optimization/configs/direct_steering_calibration.yaml", run_dir, mode, commit=outputs.commit)


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60, retries=modal.Retries(max_retries=2, backoff_coefficient=2.0), secrets=[modal.Secret.from_name("hf-llama-stage-a"),modal.Secret.from_name("openai-secret")], volumes={"/outputs":outputs,"/root/.cache/huggingface":model_cache})
def frozen_direct_validation(run_id: str, mode: str="fresh"):
    import os,sys
    os.chdir(ROOT);sys.path.insert(0,ROOT)
    from steering_suffix_optimization.frozen_direct_validation import run
    run_dir=Path(OUTPUT_ROOT)/run_id
    if mode=="fresh" and (run_dir/"checkpoint.json").exists():mode="resume"
    return run(Path(ROOT)/"steering_suffix_optimization/configs/frozen_direct_validation.yaml",run_dir,mode,commit=outputs.commit)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a"), modal.Secret.from_name("openai-secret")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def trajectory_imitation(run_id: str, mode: str = "fresh"):
    import os
    import sys
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from steering_suffix_optimization.trajectory_imitation import run
    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(
        Path(ROOT) / "steering_suffix_optimization/configs/trajectory_imitation.yaml",
        run_dir,
        mode,
        commit=outputs.commit,
    )


@app.function(image=image,volumes={"/outputs":outputs})
@modal.fastapi_endpoint()
def run_dashboard(run_id: str):
    from fastapi.responses import HTMLResponse
    outputs.reload()
    path=Path(OUTPUT_ROOT)/run_id/"dashboard.html"
    if not path.exists():return HTMLResponse("<h1>Dashboard not ready</h1>",status_code=404)
    return HTMLResponse(path.read_text(),headers={"Cache-Control":"no-store"})
