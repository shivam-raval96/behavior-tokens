from pathlib import Path

import modal

ROOT = "/workspace"
OUTPUT_ROOT = "/outputs/jailbreaks/runs"
app = modal.App("suffix-fixed-rollout-ce")
outputs = modal.Volume.from_name("bt-outputs", create_if_missing=True)
model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "transformers==4.53.2", "accelerate==1.9.0", "numpy", "pyyaml", "tqdm", "sentencepiece", "matplotlib", "fastapi[standard]")
    .add_local_dir("suffix_optimization_algorithm", f"{ROOT}/suffix_optimization_algorithm")
    .add_local_file("jailbreaks/llm-attacks-reference/data/advbench/harmful_behaviors.csv", f"{ROOT}/harmful_behaviors.csv")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def experiment_001(run_id: str, mode: str = "fresh"):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from suffix_optimization_algorithm.fixed_rollout_ce import run

    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(Path(ROOT) / "suffix_optimization_algorithm/configs/experiment_001.yaml", run_dir, mode, commit=outputs.commit)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def full_kl_crosscheck(run_id: str, mode: str = "fresh"):
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from suffix_optimization_algorithm.full_kl_crosscheck import run

    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists():
        mode = "resume"
    return run(Path(ROOT) / "suffix_optimization_algorithm/configs/experiment_002_full_kl.yaml", run_dir, mode, commit=outputs.commit)


@app.local_entrypoint(name="full-kl-crosscheck")
def full_kl_crosscheck_entrypoint(run_id: str, mode: str = "fresh"):
    full_kl_crosscheck.remote(run_id, mode)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def q_weighted_crosscheck(run_id: str, mode: str = "fresh"):
    import os
    import sys
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    from suffix_optimization_algorithm.q_weighted_crosscheck import run
    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists(): mode = "resume"
    return run(Path(ROOT) / "suffix_optimization_algorithm/configs/experiment_003_q_weighted.yaml", run_dir, mode, commit=outputs.commit)


@app.local_entrypoint(name="q-weighted-crosscheck")
def q_weighted_crosscheck_entrypoint(run_id: str, mode: str = "fresh"):
    q_weighted_crosscheck.remote(run_id, mode)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": model_cache},
)
def gcg_q_search(run_id: str, config_name: str, mode: str = "fresh"):
    import os
    import sys
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    from suffix_optimization_algorithm.gcg_q_search import run
    allowed = {"experiment_004_gcg_preflight.yaml", "experiment_004_gcg_warm.yaml", "experiment_004_gcg_cold.yaml"}
    if config_name not in allowed:
        raise ValueError("unknown GCG configuration")
    run_dir = Path(OUTPUT_ROOT) / run_id
    if mode == "fresh" and (run_dir / "checkpoint.json").exists(): mode = "resume"
    return run(Path(ROOT) / "suffix_optimization_algorithm/configs" / config_name, run_dir, mode, commit=outputs.commit)


@app.local_entrypoint(name="gcg-q-search")
def gcg_q_search_entrypoint(run_id: str, config_name: str, mode: str = "fresh"):
    gcg_q_search.remote(run_id, config_name, mode)


@app.local_entrypoint()
def main(run_id: str, mode: str = "fresh"):
    experiment_001.remote(run_id, mode)


@app.function(image=image, volumes={"/outputs": outputs})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def dashboard():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    @web.get("/runs/{run_id}/dashboard.html", response_class=HTMLResponse)
    def get_dashboard(run_id: str):
        if "/" in run_id or run_id in {".", ".."}:
            raise HTTPException(400, "invalid run id")
        outputs.reload()
        path = Path(OUTPUT_ROOT) / run_id / "dashboard.html"
        if not path.exists():
            raise HTTPException(404, "dashboard not available yet")
        return HTMLResponse(path.read_text())

    return web
