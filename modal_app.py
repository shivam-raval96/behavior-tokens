"""Run the steering / behavior-token pipeline on a Modal cloud GPU.

Fixes the local MPS memory/throughput wall — a 1B model on an A10G runs the
whole pipeline in seconds-to-minutes. Code + configs are shipped in the image;
outputs and the HF cache live on persistent Volumes.

Examples (from the repo root, base env has `modal`):
    modal run modal_app.py --task experiment --config sadness.yaml   # steering_vectors.run
    modal run modal_app.py --task gcg        --config sadness.yaml   # token_optimization.run
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
    # ship both packages + shared configs (outputs live on the Volume, not the image)
    .add_local_dir("steering_vectors", "/root/steering_vectors",
                   ignore=["**/__pycache__/**", "*.pt"])
    .add_local_dir("token_optimization", "/root/token_optimization",
                   ignore=["**/__pycache__/**", "*.pt"])
    .add_local_dir("jailbreaks", "/root/jailbreaks",
                   ignore=["**/__pycache__/**", "outputs/**", "*.png"])
    .add_local_dir("configs", "/root/configs")
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

    # rewrite the config: outputs -> the Volume root, device -> CUDA
    with open(f"/root/configs/{config_name}") as f:
        cfg = yaml.safe_load(f)
    cfg["output_dir"] = "/outputs"
    cfg["device"] = "auto"
    tmp = "/root/_cfg.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)

    if task == "experiment":
        from steering_vectors.run import run
        run(tmp)
    elif task == "gcg":
        from token_optimization.run import run
        run(tmp)
    elif task == "sweep":
        from token_optimization.sweep_len import run
        run(tmp, lengths or [1, 8])
    elif task == "seedsweep":
        from token_optimization.seed_sweep import run
        run(tmp, lengths or [8], seeds, out or "seed_sweep.json")
    else:
        raise ValueError(f"unknown task {task}")

    outputs.commit()
    return f"{task} done -> /outputs"


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("hf-gated")],  # HF_TOKEN for gated meta-llama
              volumes={"/outputs": outputs, "/root/.cache/huggingface": hf_cache})
def run_jbgcg(datasets, n_behaviors: int, steps: int, suffix_len: int,
              search_batch: int, dtype: str,
              model: str = "unsloth/Llama-3.2-1B-Instruct",
              out: str = "/outputs/gcg_zou", eval_chunk: int = 128):
    """Zou GCG jailbreak benchmark (jailbreaks/gcg_bench) -> `out`. A100 for 7B."""
    import os
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from jailbreaks.gcg_bench.benchmark import run

    args = SimpleNamespace(
        model=model, datasets=datasets,
        n_behaviors=n_behaviors, steps=steps, suffix_len=suffix_len, topk=256,
        search_batch=search_batch, eval_chunk=eval_chunk, max_new_tokens=128,
        device="auto", dtype=dtype, seed=0, out=out, quick=False,
    )
    run(args)
    outputs.commit()
    return f"jbgcg done -> {out}"


@app.function(image=image, timeout=1800,
              secrets=[modal.Secret.from_name("hf-gated")],
              volumes={"/root/.cache/huggingface": hf_cache})
def prewarm(model_name: str):
    """Download the model to the HF cache volume ONCE so fan-out workers all hit cache."""
    from huggingface_hub import snapshot_download
    snapshot_download(model_name)
    hf_cache.commit()
    return f"prewarmed {model_name}"


@app.function(image=image, gpu="A100", timeout=5400, max_containers=30,
              secrets=[modal.Secret.from_name("hf-gated")],
              volumes={"/root/.cache/huggingface": hf_cache})
def attack_one(model_name: str, goal: str, target: str, steps: int, suffix_len: int,
               search_batch: int, eval_chunk: int, dtype: str, max_new_tokens: int, seed: int):
    """GCG on a SINGLE behavior — the unit of parallelism (one container each)."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from jailbreaks.gcg_bench.asr import is_success
    from jailbreaks.gcg_bench.benchmark import load_model, resolve_device
    from jailbreaks.gcg_bench.gcg_zou import GCG, GCGConfig

    model, tok = load_model(model_name, resolve_device("auto"), dtype)
    gcfg = GCGConfig(suffix_len=suffix_len, steps=steps, topk=256,
                     search_batch=search_batch, eval_chunk=eval_chunk, seed=seed)
    atk = GCG(model, tok, gcfg)
    # per-worker step logs go to `modal app logs` (one container per behavior)
    res = atk.optimize(goal, target, log=lambda m: print(f"[w] {goal[:30]!r} {m}", flush=True))
    a = atk.generate(goal, res.suffix_ids, max_new_tokens)
    b = atk.generate(goal, None, max_new_tokens)
    return {"goal": goal, "target": target, "suffix": res.suffix,
            "final_loss": res.loss, "baseline_gen": b[:400], "attacked_gen": a[:400],
            "baseline_success": is_success(b), "attacked_success": is_success(a)}


@app.function(image=image, timeout=14400, volumes={"/outputs": outputs})
def run_jbgcg_par(dataset: str, n_behaviors: int, steps: int, suffix_len: int,
                  search_batch: int, eval_chunk: int, dtype: str, model: str, out: str):
    """Fan out behaviors across attack_one containers, aggregate ASR, write results."""
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from jailbreaks.gcg_bench.asr import asr
    from jailbreaks.gcg_bench.benchmark import make_plot
    from jailbreaks.gcg_bench.datasets_hf import load as load_ds

    recs = load_ds(dataset, n=n_behaviors, seed=0)
    print(f"[par] prewarming {model} …")
    prewarm.remote(model)
    N = len(recs)
    print(f"[par] fanning out {N} behaviors on A100 …", flush=True)
    t0 = time.time()
    call_args = [(model, r["goal"], r["target"], steps, suffix_len, search_batch,
                  eval_chunk, dtype, 128, 0) for r in recs]
    os.makedirs(out, exist_ok=True)

    def flush(rows):
        """Write + COMMIT partial results after every completion — survives kill/timeout."""
        atk = [r["attacked_gen"] for r in rows]
        base = [r["baseline_gen"] for r in rows]
        summary = {dataset: {"n": len(rows),
                             "asr_baseline": round(asr(base), 4),
                             "asr_gcg": round(asr(atk), 4)}}
        payload = {
            "config": {"model": model, "dtype": dtype,
                       "metric": "refusal-string-match ASR + non-Latin-script guard",
                       "gcg": {"suffix_len": suffix_len, "steps": steps, "topk": 256,
                               "search_batch": search_batch, "eval_chunk": eval_chunk},
                       "parallel": True, "target_n": N},
            "runtime_sec": round(time.time() - t0, 1),
            "done": len(rows), "summary": summary, "per_behavior": {dataset: rows},
        }
        with open(f"{out}/results.json", "w") as f:
            json.dump(payload, f, indent=2)
        make_plot(summary, f"{out}/asr.png", model)
        outputs.commit()
        return summary

    # iterate completions AS THEY LAND; tolerate a failed worker (return_exceptions)
    rows, done, succ, errs = [], 0, 0, 0
    for res in attack_one.starmap(call_args, order_outputs=False, return_exceptions=True):
        done += 1
        if isinstance(res, Exception):
            errs += 1
            print(f"[par] worker {done}/{N} FAILED: {type(res).__name__}: {str(res)[:80]}",
                  flush=True)
            continue
        rows.append(res)
        succ += int(res["attacked_success"])
        filled = round(20 * done / N)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"[par] {bar} {done}/{N} | ASR_gcg={succ/max(len(rows),1):.2f} | "
              f"{round(time.time()-t0)}s | errs={errs} | "
              f"last={'JAILBROKEN' if res['attacked_success'] else 'refused  '} "
              f"{res['goal'][:34]}", flush=True)
        flush(rows)                      # checkpoint every completion

    summary = flush(rows)
    print(f"[par] done {round(time.time()-t0,1)}s  {summary}  ({errs} failed)", flush=True)
    return summary


@app.local_entrypoint()
def main(task: str = "experiment", config: str = "sadness.yaml",
         lengths: str = "", seeds: int = 10, out: str = "",
         datasets: str = "advbench,jbb,forbidden", n_behaviors: int = 25,
         steps: int = 500, suffix_len: int = 20, search_batch: int = 512,
         dtype: str = "bfloat16", model: str = "unsloth/Llama-3.2-1B-Instruct",
         gcg_out: str = "/outputs/gcg_zou", eval_chunk: int = 128):
    if task == "jbgcg":
        ds = [d for d in datasets.split(",") if d.strip()]
        print(run_jbgcg.remote(ds, n_behaviors, steps, suffix_len, search_batch,
                               dtype, model, gcg_out, eval_chunk))
        return
    if task == "jbgcgpar":                       # fan behaviors across containers
        one = datasets.split(",")[0].strip()     # one dataset per parallel run
        print(run_jbgcg_par.remote(one, n_behaviors, steps, suffix_len, search_batch,
                                   eval_chunk, dtype, model, gcg_out))
        return
    L = [int(x) for x in lengths.split(",") if x.strip()] or None
    print(run_task.remote(task, config, L, seeds, out or None))
