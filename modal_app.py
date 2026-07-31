"""Run the steering / behavior-token pipeline on a Modal cloud GPU.

Fixes the local MPS memory/throughput wall — a 1B model on an A10G runs the
whole pipeline in seconds-to-minutes. Code + configs are shipped in the image;
outputs and the HF cache live on persistent Volumes.

Examples (from the repo root, base env has `modal`):
    modal run modal_app.py --task experiment --config sadness.yaml
    modal run modal_app.py --task gcg        --config sadness.yaml
    modal run modal_app.py --task sweep      --config sadness.yaml --lengths 1,8,16,32
    modal run modal_app.py --task seedsweep  --config sadness.yaml --lengths 8 --seeds 10 --out seed_sweep_L8.json

Pull results back:
    modal volume get bt-outputs / ./modal_outputs
"""
import modal

app = modal.App("behavior-tokens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.49.0", "datasets", "scikit-learn",
        "accelerate", "pyyaml", "matplotlib", "tqdm", "huggingface_hub",
    )
    # ship the package + configs, but NOT the big local outputs / caches
    .add_local_dir("steering", "/root/steering",
                   ignore=["outputs", "outputs/**", "**/__pycache__/**", "*.pt"])
)

outputs = modal.Volume.from_name("bt-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("bt-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={"/outputs": outputs, "/root/.cache/huggingface": hf_cache})
def run_task(task: str, config_name: str, lengths=None, seeds: int = 10, out: str = None):
    import os
    import sys
    import yaml

    sys.path.insert(0, "/root")
    os.chdir("/root")

    # rewrite the config: write outputs to the Volume, let device resolve to CUDA
    with open(f"/root/steering/configs/{config_name}") as f:
        cfg = yaml.safe_load(f)
    cfg["output_dir"] = "/outputs"
    cfg["device"] = "auto"
    tmp = "/root/_cfg.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)

    if task == "experiment":
        from steering.run_experiment import run
        run(tmp)
    elif task == "gcg":
        from steering.run_gcg import run
        run(tmp)
    elif task == "sweep":
        from steering.sweep_len import run
        run(tmp, lengths or [1, 8])
    elif task == "seedsweep":
        from steering.seed_sweep import run
        run(tmp, lengths or [8], seeds, out or "seed_sweep.json")
    else:
        raise ValueError(f"unknown task {task}")

    outputs.commit()
    return f"{task} done -> /outputs"


@app.local_entrypoint()
def main(task: str = "experiment", config: str = "sadness.yaml",
         lengths: str = "", seeds: int = 10, out: str = ""):
    L = [int(x) for x in lengths.split(",") if x.strip()] or None
    print(run_task.remote(task, config, L, seeds, out or None))
