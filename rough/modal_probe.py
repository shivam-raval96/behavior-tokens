import modal

app=modal.App("terminal-wrench-qwen35-probe")
outputs=modal.Volume.from_name("bt-outputs",create_if_missing=True)
cache=modal.Volume.from_name("bt-hf-cache",create_if_missing=True)
image=(modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install(
    "torch>=2.7","transformers>=5.0","accelerate","scikit-learn","joblib","numpy","matplotlib","fastapi")
    .add_local_file("rough/terminal_wrench_probe.py","/root/terminal_wrench_probe.py")
    .add_local_file("rough/prepare_run.py","/root/prepare_run.py"))

@app.function(image=image,timeout=7200,volumes={"/outputs":outputs})
def prepare(run_id:str):
    import subprocess
    subprocess.run(["python","/root/prepare_run.py","--run-id",run_id],check=True)
    outputs.commit()

@app.function(image=image,gpu="A100-80GB",timeout=43200,retries=modal.Retries(max_retries=1),
              secrets=[modal.Secret.from_name("hf-llama-stage-a")],volumes={"/outputs":outputs,"/root/.cache/huggingface":cache})
def extract(run_id:str,shard:int):
    from terminal_wrench_probe import extract_worker
    outputs.reload()
    result=extract_worker(f"/outputs/rough/runs/{run_id}",shard,checkpoint_callback=outputs.commit); outputs.commit(); return result

@app.function(image=image,timeout=14400,memory=8192,volumes={"/outputs":outputs})
def finalize(run_id:str):
    from terminal_wrench_probe import fit_probes
    outputs.reload(); result=fit_probes(f"/outputs/rough/runs/{run_id}"); outputs.commit(); return result

@app.function(image=image,volumes={"/outputs":outputs})
@modal.fastapi_endpoint()
def dashboard(run_id:str):
    from fastapi.responses import HTMLResponse
    outputs.reload(); p=f"/outputs/rough/runs/{run_id}/dashboard.html"
    try: return HTMLResponse(open(p).read())
    except FileNotFoundError: return HTMLResponse("dashboard not ready",status_code=404)

@app.function(image=image,timeout=43200,volumes={"/outputs":outputs})
def orchestrate(run_id:str):
    from pathlib import Path
    outputs.reload()
    if not Path(f"/outputs/rough/runs/{run_id}/manifest.jsonl").exists():
        prepare.remote(run_id)
    jobs=[extract.spawn(run_id,i) for i in range(4)]
    print([j.get() for j in jobs],flush=True)
    return finalize.remote(run_id)

@app.local_entrypoint()
def main(run_id:str,finalize_only:bool=False):
    call=(finalize if finalize_only else orchestrate).spawn(run_id)
    print({"orchestrator_call_id":call.object_id,"run_id":run_id},flush=True)
