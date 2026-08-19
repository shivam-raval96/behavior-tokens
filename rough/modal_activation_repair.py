import time
from pathlib import Path
import modal

app=modal.App("terminal-wrench-activation-repair")
outputs=modal.Volume.from_name("bt-outputs",create_if_missing=True); cache=modal.Volume.from_name("bt-hf-cache",create_if_missing=True)
image=(modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch>=2.7","transformers>=5.0","accelerate","numpy","fastapi")
       .add_local_file("rough/terminal_wrench_probe.py","/root/terminal_wrench_probe.py").add_local_file("rough/repair_activation_cache.py","/root/repair_activation_cache.py"))

@app.function(image=image,timeout=7200,volumes={"/outputs":outputs})
def prepare(run_id:str):
    from repair_activation_cache import prepare_repair
    result=prepare_repair(f"/outputs/rough/runs/{run_id}"); outputs.commit(); return result

@app.function(image=image,timeout=1200,volumes={"/outputs":outputs})
def prepare_full(run_id:str,prepared_run_id:str):
    from repair_activation_cache import prepare_full_singleton
    result=prepare_full_singleton(f"/outputs/rough/runs/{run_id}",prepared_run_id); outputs.commit(); return result

@app.function(image=image,gpu="A100-80GB",timeout=43200,retries=modal.Retries(max_retries=2),secrets=[modal.Secret.from_name("hf-llama-stage-a")],volumes={"/outputs":outputs,"/root/.cache/huggingface":cache})
def extract(run_id:str,worker:int):
    from repair_activation_cache import extract_shard
    outputs.reload(); result=extract_shard(f"/outputs/rough/runs/{run_id}",worker,outputs.commit); outputs.commit(); return result

@app.function(image=image,timeout=14400,memory=16384,volumes={"/outputs":outputs})
def finalize(run_id:str):
    from repair_activation_cache import finalize_repair, finalize_full_singleton
    outputs.reload(); path=f"/outputs/rough/runs/{run_id}"; import json
    config=json.load(open(path+"/resolved_config.json")); return (finalize_full_singleton if config.get("construction")=="fresh_singleton" else finalize_repair)(path,outputs.commit)

@app.function(image=image,gpu="A100-80GB",timeout=7200,secrets=[modal.Secret.from_name("hf-llama-stage-a")],volumes={"/outputs":outputs,"/root/.cache/huggingface":cache})
def validate(run_id:str):
    from repair_activation_cache import validate_batch_equivalence
    outputs.reload(); return validate_batch_equivalence(f"/outputs/rough/runs/{run_id}",outputs.commit)

@app.function(image=image,timeout=600,volumes={"/outputs":outputs})
def configure(run_id:str):
    from repair_activation_cache import configure_singleton
    outputs.reload(); result=configure_singleton(f"/outputs/rough/runs/{run_id}"); outputs.commit(); return result

@app.function(image=image,gpu="A100-80GB",timeout=7200,secrets=[modal.Secret.from_name("hf-llama-stage-a")],volumes={"/outputs":outputs,"/root/.cache/huggingface":cache})
def audit(run_id:str):
    from repair_activation_cache import audit_saved_singletons
    outputs.reload(); return audit_saved_singletons(f"/outputs/rough/runs/{run_id}",outputs.commit)

@app.function(image=image,timeout=7200,memory=4096,volumes={"/outputs":outputs})
def compact(run_id:str,worker:int):
    from repair_activation_cache import compact_worker
    outputs.reload(); return compact_worker(f"/outputs/rough/runs/{run_id}",worker,outputs.commit)

@app.function(image=image,timeout=7200,memory=16384,volumes={"/outputs":outputs})
def compact_safe(run_id:str):
    from repair_activation_cache import compact_all_workers
    outputs.reload(); result=compact_all_workers(f"/outputs/rough/runs/{run_id}"); outputs.commit(); return result

@app.function(image=image,timeout=14400,volumes={"/outputs":outputs})
def compact_orchestrate(run_id:str):
    shards=compact_safe.remote(run_id); final=finalize.remote(run_id); validation=audit.remote(run_id); return {"shards":shards,"final":final,"audit":validation}

@app.function(image=image,volumes={"/outputs":outputs})
@modal.fastapi_endpoint()
def dashboard(run_id:str):
    from fastapi.responses import HTMLResponse
    outputs.reload(); p=f"/outputs/rough/runs/{run_id}/dashboard.html"
    try:return HTMLResponse(open(p).read())
    except FileNotFoundError:return HTMLResponse("dashboard not ready",status_code=404)

@app.function(image=image,timeout=43200,volumes={"/outputs":outputs})
def orchestrate(run_id:str,prepared_run_id:str=""):
    outputs.reload()
    if not Path(f"/outputs/rough/runs/{run_id}/resolved_config.json").exists():
        (prepare_full.remote(run_id,prepared_run_id) if prepared_run_id else prepare.remote(run_id))
    first=extract.spawn(run_id,0)
    # Require a real activation batch checkpoint before fanning out model loads.
    marker=Path(f"/outputs/rough/runs/{run_id}/activations/worker-00")
    deadline=time.time()+3600
    while time.time()<deadline:
        outputs.reload()
        if list(marker.glob("batch-*.npy")):break
        time.sleep(10)
    else:raise TimeoutError("worker 0 produced no validation batch")
    jobs=[first]
    for worker in range(1,8):jobs.append(extract.spawn(run_id,worker)); time.sleep(3)
    results=[job.get() for job in jobs]
    final=finalize.remote(run_id)
    return {"workers":results,"final":final,"audit":audit.remote(run_id) if prepared_run_id else None}

@app.local_entrypoint()
def main(run_id:str,validate_only:bool=False,configure_singleton:bool=False,prepared_run_id:str="",compact_only:bool=False):
    if validate_only:
        print(validate.remote(run_id),flush=True)
    elif configure_singleton:
        print(configure.remote(run_id),flush=True)
    elif compact_only:
        call=compact_orchestrate.spawn(run_id); print({"run_id":run_id,"compact_call_id":call.object_id},flush=True)
    else:
        call=orchestrate.spawn(run_id,prepared_run_id); print({"run_id":run_id,"call_id":call.object_id,"validate_only":False},flush=True)
