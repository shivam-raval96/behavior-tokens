import time
from pathlib import Path

import modal

MODEL_ID = "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill"
MODEL_REVISION = "29d1636fc20c9a8599161fc008a7dbdd701be147"

app = modal.App("gemini-distill-terminal-wrench-singleton")
outputs = modal.Volume.from_name("bt-outputs", create_if_missing=True)
cache = modal.Volume.from_name("bt-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.7", "transformers>=5.0", "accelerate", "numpy", "fastapi")
    .add_local_file("rough/terminal_wrench_probe.py", "/root/terminal_wrench_probe.py")
    .add_local_file("rough/repair_activation_cache.py", "/root/repair_activation_cache.py")
)


@app.function(image=image, timeout=1200, volumes={"/outputs": outputs})
def prepare(run_id: str, prepared_run_id: str):
    from repair_activation_cache import prepare_full_singleton

    result = prepare_full_singleton(
        f"/outputs/rough/runs/{run_id}", prepared_run_id, MODEL_ID, MODEL_REVISION
    )
    outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=43200,
    retries=modal.Retries(max_retries=2),
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": cache},
)
def extract(run_id: str, worker: int):
    from repair_activation_cache import extract_shard

    outputs.reload()
    result = extract_shard(f"/outputs/rough/runs/{run_id}", worker, outputs.commit)
    outputs.commit()
    return result


@app.function(image=image, timeout=7200, memory=4096, volumes={"/outputs": outputs})
def compact(run_id: str, worker: int):
    from repair_activation_cache import compact_worker

    outputs.reload()
    return compact_worker(f"/outputs/rough/runs/{run_id}", worker, outputs.commit)


@app.function(image=image, timeout=14400, memory=16384, volumes={"/outputs": outputs})
def finalize(run_id: str):
    from repair_activation_cache import finalize_full_singleton

    outputs.reload()
    return finalize_full_singleton(f"/outputs/rough/runs/{run_id}", outputs.commit)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-llama-stage-a")],
    volumes={"/outputs": outputs, "/root/.cache/huggingface": cache},
)
def audit(run_id: str):
    from repair_activation_cache import audit_saved_singletons

    outputs.reload()
    return audit_saved_singletons(f"/outputs/rough/runs/{run_id}", outputs.commit)


@app.function(image=image, volumes={"/outputs": outputs})
@modal.fastapi_endpoint()
def dashboard(run_id: str):
    from fastapi.responses import HTMLResponse

    outputs.reload()
    path = f"/outputs/rough/runs/{run_id}/dashboard.html"
    try:
        return HTMLResponse(open(path).read())
    except FileNotFoundError:
        return HTMLResponse("dashboard not ready", status_code=404)


@app.function(image=image, timeout=43200, volumes={"/outputs": outputs})
def orchestrate(run_id: str, prepared_run_id: str):
    prepare.remote(run_id, prepared_run_id)
    first = extract.spawn(run_id, 0)
    marker = Path(f"/outputs/rough/runs/{run_id}/activations/worker-00")
    deadline = time.time() + 3600
    while time.time() < deadline:
        outputs.reload()
        if list(marker.glob("batch-*.npy")):
            break
        time.sleep(10)
    else:
        raise TimeoutError("worker 0 produced no validated singleton batch")
    jobs = [first]
    for worker in range(1, 8):
        jobs.append(extract.spawn(run_id, worker))
        time.sleep(3)
    worker_results = [job.get() for job in jobs]
    compact_results = [compact.remote(run_id, worker) for worker in range(8)]
    final_result = finalize.remote(run_id)
    audit_result = audit.remote(run_id)
    return {
        "workers": worker_results,
        "compaction": compact_results,
        "final": final_result,
        "audit": audit_result,
    }


@app.local_entrypoint()
def main(run_id: str, prepared_run_id: str):
    call = orchestrate.spawn(run_id, prepared_run_id)
    print({"run_id": run_id, "call_id": call.object_id}, flush=True)
