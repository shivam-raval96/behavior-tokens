import modal

app=modal.App("singleton-layer20-repeated-holdout")
outputs=modal.Volume.from_name("bt-outputs",create_if_missing=True)
image=(modal.Image.debian_slim(python_version="3.11").pip_install("numpy","scikit-learn","joblib","matplotlib","fastapi").add_local_file("rough/singleton_paired_repeated_holdout.py","/root/singleton_paired_repeated_holdout.py"))

@app.function(image=image,timeout=7200,memory=8192,volumes={"/outputs":outputs})
def run(run_id:str,source_run_id:str):
    from singleton_paired_repeated_holdout import run_experiment
    outputs.reload(); result=run_experiment(f"/outputs/rough/runs/{source_run_id}",f"/outputs/rough/runs/{run_id}"); outputs.commit(); return result

@app.function(image=image,volumes={"/outputs":outputs})
@modal.fastapi_endpoint()
def dashboard(run_id:str):
    from fastapi.responses import HTMLResponse
    outputs.reload(); path=f"/outputs/rough/runs/{run_id}/dashboard.html"
    try:return HTMLResponse(open(path).read())
    except FileNotFoundError:return HTMLResponse("dashboard not ready",status_code=404)

@app.local_entrypoint()
def main(run_id:str,source_run_id:str):
    call=run.spawn(run_id,source_run_id); print({"run_id":run_id,"call_id":call.object_id},flush=True)
