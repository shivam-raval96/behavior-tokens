from __future__ import annotations

import hashlib, json, os, time
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3.5-9B"
DATA_REV = "d8a29613235a0ef56a8b70b3142626a533da28c2"
LAYERS, HIDDEN, MAX_LEN = 32, 4096, 16384


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def split_for(task_id: str) -> str:
    value = int(hashlib.sha256(f"42|{task_id}".encode()).hexdigest(), 16) % 100
    return "train" if value < 70 else "test"


def render_dashboard(run: Path, snapshot: dict) -> None:
    history = run / "dashboard_history.jsonl"
    with history.open("a") as f:
        f.write(json.dumps(snapshot, sort_keys=True) + "\n")
    payload = json.dumps(snapshot, indent=2)
    run.joinpath("dashboard.html").write_text(f"""<!doctype html><meta http-equiv=refresh content=15>
<title>Terminal Wrench probe</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;background:#101418;color:#e8eef2}}progress{{width:100%}}pre{{background:#182028;padding:18px;overflow:auto}}</style>
<h1>Qwen3.5-9B Terminal Wrench probe</h1><h2>{snapshot.get('phase')}</h2>
<progress value="{snapshot.get('completed',0)}" max="{max(1,snapshot.get('total',1))}"></progress>
<p>{snapshot.get('completed',0)} / {snapshot.get('total',0)} · elapsed {snapshot.get('elapsed_seconds',0):.1f}s · ETA {snapshot.get('eta_seconds')}</p>
<pre>{payload}</pre>""")


def trajectory_messages(data: dict) -> list[dict]:
    messages = []
    for step in data.get("steps", []):
        source = step.get("source")
        role = "assistant" if source in {"agent", "assistant"} else "user"
        content = step.get("message")
        if content is None:
            content = json.dumps(step, sort_keys=True, ensure_ascii=False)
        messages.append({"role": role, "content": str(content)})
    if not any(x["role"] == "assistant" for x in messages):
        raise ValueError("trajectory has no assistant response")
    return messages


def tokenize_with_boundary(tokenizer, messages: list[dict]) -> tuple[list[int], int, dict]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    starts = [i for i in range(len(ids)) if ids[i:i+len(assistant_start)] == assistant_start]
    if not starts:
        raise ValueError("assistant token boundary absent")
    start = starts[-1] + len(assistant_start)
    ends = [i for i in range(start, len(ids)) if ids[i] == end_id]
    if not ends or ends[0] <= start:
        raise ValueError("assistant end boundary absent or empty")
    end = ends[0]
    position = end - 1
    original_length = len(ids)
    left_cut = max(0, len(ids) - MAX_LEN)
    if position < left_cut:
        raise ValueError("probe position truncated")
    ids = ids[left_cut:]
    position -= left_cut
    if ids[position + 1] != end_id:
        raise AssertionError("probe token is not immediately before im_end")
    audit = {"original_length": original_length, "left_cut": left_cut, "position": position,
             "token_id": ids[position], "token": tokenizer.decode([ids[position]]),
             "neighbor_ids": ids[max(0, position-3):position+2],
             "neighbor_text": tokenizer.decode(ids[max(0, position-3):position+2])}
    return ids, position, audit


def extract_worker(run_dir: str, shard: int, shards: int = 4, checkpoint_callback=lambda: None) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    run = Path(run_dir); manifest = [json.loads(x) for x in (run / "manifest.jsonl").read_text().splitlines()]
    rows = [r for i, r in enumerate(manifest) if i % shards == shard]
    shard_dir = run / "activations" / f"worker-{shard:02d}"; shard_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if not (shard_dir / f"{r['example_id']}.npy").exists()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"; tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    core = model.model.language_model if hasattr(model.model, "language_model") else model.model
    blocks = core.layers
    captured = [None] * LAYERS
    positions = None
    hooks = []
    for layer_idx, block in enumerate(blocks):
        def hook(_module, _inputs, output, idx=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions].detach().to("cpu", torch.float16)
        hooks.append(block.register_forward_hook(hook))
    completed = 0; started = time.time(); audit_rows = []
    try:
        prepared = []
        for row in rows:
            data = json.loads(Path(row["path"]).read_text())
            ids, pos, audit = tokenize_with_boundary(tokenizer, trajectory_messages(data))
            prepared.append((len(ids), row, ids, pos, audit))
        prepared.sort(key=lambda x: x[0])
        cursor = 0; token_budget = 32768; max_batch_size = 16; oom_count = 0
        while cursor < len(prepared):
            batch = []; tokens = 0
            while cursor + len(batch) < len(prepared) and len(batch) < max_batch_size:
                candidate = prepared[cursor + len(batch)]
                projected = candidate[0] * (len(batch) + 1)
                if batch and projected > token_budget: break
                batch.append(candidate); tokens = projected
            maxlen = max(x[0] for x in batch)
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(input_ids)
            pos_list = []
            for b, (_, _, ids, pos, _) in enumerate(batch):
                offset = maxlen - len(ids); input_ids[b, offset:] = torch.tensor(ids); mask[b, offset:] = 1; pos_list.append(offset + pos)
            input_ids, mask = input_ids.cuda(), mask.cuda(); positions = torch.tensor(pos_list, device="cuda")
            try:
                with torch.inference_mode(): core(input_ids=input_ids, attention_mask=mask, use_cache=False)
            except torch.OutOfMemoryError:
                oom_count += 1; del input_ids, mask; torch.cuda.empty_cache()
                if len(batch) == 1: raise
                max_batch_size = max(1, len(batch)//2); token_budget = max(batch[0][0], token_budget//2)
                print(json.dumps({"phase":"oom_backoff","worker":shard,"token_budget":token_budget,"max_batch_size":max_batch_size,"oom_count":oom_count}),flush=True)
                continue
            values = torch.stack(captured, dim=1).numpy()
            for b, (_, row, _, _, audit) in enumerate(batch):
                np.save(shard_dir / f"{row['example_id']}.npy", values[b], allow_pickle=False)
                atomic_json(shard_dir / f"{row['example_id']}.json", {**row, **audit})
                if len(audit_rows) < 25: audit_rows.append({"example_id": row["example_id"], **audit})
            completed += len(batch); cursor += len(batch)
            if completed % 100 < len(batch) or completed == len(rows):
                elapsed=time.time()-started; rate=completed/max(elapsed,1e-6)
                snap={"phase":"extracting","worker":shard,"completed":completed,"total":len(rows),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(rows)-completed)/rate,"error_count":oom_count,"retry_count":0,"token_budget":token_budget,"max_batch_size":max_batch_size}
                atomic_json(shard_dir/"progress.json",snap); render_dashboard(run, snap); checkpoint_callback(); print(json.dumps(snap),flush=True)
    finally:
        for h in hooks: h.remove()
    atomic_json(shard_dir/"boundary_audit.json",audit_rows); checkpoint_callback()
    return {"worker": shard, "completed": completed}


def fit_probes(run_dir: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    run=Path(run_dir); rows=[json.loads(x) for x in (run/"manifest.jsonl").read_text().splitlines()]
    arrays=[]
    for i,row in enumerate(rows): arrays.append(np.load(run/"activations"/f"worker-{i%4:02d}"/f"{row['example_id']}.npy"))
    x=np.stack(arrays); y=np.array([r["label"] for r in rows])
    source=np.array([r["source_model"] for r in rows]); split=np.array([r["split"] for r in rows])
    train=(source=="claude-opus-4.6") & (split=="train"); test=(source=="claude-opus-4.6") & (split=="test")
    np.save(run/"activations"/"all_layers.float16.npy",x,allow_pickle=False)
    metrics=[]
    for layer in range(LAYERS):
        probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=42,n_jobs=1))
        probe.fit(x[train,layer].astype(np.float32),y[train])
        train_pred=probe.predict(x[train,layer].astype(np.float32)); pred=probe.predict(x[test,layer].astype(np.float32)); prob=probe.predict_proba(x[test,layer].astype(np.float32))[:,1]
        evaluations={}
        for model in ("claude-opus-4.6","gemini-3.1-pro","gpt-5.4"):
            for partition in ("train","test"):
                mask=(source==model)&(split==partition); model_pred=probe.predict(x[mask,layer].astype(np.float32))
                evaluations[f"{model}|{partition}"]={"accuracy":float(accuracy_score(y[mask],model_pred)),"examples":int(mask.sum()),"confusion_matrix":confusion_matrix(y[mask],model_pred,labels=[0,1]).tolist()}
        m={"layer":layer,"train_accuracy":accuracy_score(y[train],train_pred),"test_accuracy":accuracy_score(y[test],pred),"balanced_accuracy":balanced_accuracy_score(y[test],pred),"roc_auc":roc_auc_score(y[test],prob),"precision":precision_score(y[test],pred),"recall":recall_score(y[test],pred),"f1":f1_score(y[test],pred),"confusion_matrix":confusion_matrix(y[test],pred,labels=[0,1]).tolist(),"evaluations":evaluations}
        metrics.append(m); joblib.dump(probe,run/f"probe_layer_{layer:02d}.joblib")
        snap={"phase":"fitting","completed":layer+1,"total":LAYERS,"latest_objective":m["test_accuracy"],"best_metric":max(x["test_accuracy"] for x in metrics),"best_layer":max(metrics,key=lambda x:x["test_accuracy"])["layer"],"error_count":0,"retry_count":0}
        atomic_json(run/"progress.json",snap); render_dashboard(run,snap); print(json.dumps(snap),flush=True)
    best=max(metrics,key=lambda m:(m["test_accuracy"],-m["layer"])); best_layer=best["layer"]
    layers=[m["layer"] for m in metrics]
    plt.figure(figsize=(9,5)); plt.plot(layers,[m["train_accuracy"] for m in metrics],marker="o",label="Claude train"); plt.plot(layers,[m["test_accuracy"] for m in metrics],marker="o",label="Claude test"); plt.axvline(best_layer,color="gray",linestyle="--",alpha=.6,label=f"Best layer {best_layer}"); plt.xlabel("Transformer layer"); plt.ylabel("Accuracy"); plt.ylim(0,1.02); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(run/"layerwise_train_test_accuracy.png",dpi=180); plt.close()
    labels=["Claude Opus 4.6","Gemini 3.1 Pro","GPT-5.4"]; keys=["claude-opus-4.6","gemini-3.1-pro","gpt-5.4"]; positions=np.arange(3); width=.36
    train_scores=[best["evaluations"][f"{k}|train"]["accuracy"] for k in keys]; test_scores=[best["evaluations"][f"{k}|test"]["accuracy"] for k in keys]
    plt.figure(figsize=(8,5)); a=plt.bar(positions-width/2,train_scores,width,label="Train tasks"); b=plt.bar(positions+width/2,test_scores,width,label="Test tasks"); plt.bar_label(a,fmt="%.3f"); plt.bar_label(b,fmt="%.3f"); plt.xticks(positions,labels); plt.ylabel("Accuracy"); plt.ylim(0,1.08); plt.title(f"Claude-trained probe transfer at layer {best_layer}"); plt.grid(axis="y",alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(run/"best_layer_cross_model_accuracy.png",dpi=180); plt.close()
    config=json.loads((run/"resolved_config.json").read_text())
    result={"selection":"highest Claude test accuracy (descriptive)","best_layer":best_layer,"best_layer_metrics":best,"per_layer":metrics,"claude_train_examples":int(train.sum()),"claude_test_examples":int(test.sum()),"config_fingerprint":config["config_fingerprint"],"status":"complete"}
    atomic_json(run/"results.json",result); atomic_json(run/"checkpoint.json",result)
    table="\n".join(f"| {label} | {train_scores[i]:.4f} | {test_scores[i]:.4f} |" for i,label in enumerate(labels))
    run.joinpath("RESULTS.md").write_text(f"# Sanitized Terminal Wrench transfer probe\n\n- Best layer (selected descriptively on Claude test): {best_layer}\n- Claude test accuracy: {best['test_accuracy']:.4f}\n- Claude train/test examples: {train.sum()}/{test.sum()}\n- No generations were performed.\n\n| Source model | Train-task accuracy | Test-task accuracy |\n|---|---:|---:|\n{table}\n")
    render_dashboard(run,{"phase":"complete","completed":LAYERS,"total":LAYERS,"latest_objective":best["test_accuracy"],"best_metric":best["test_accuracy"],"best_layer":best_layer,"error_count":0,"retry_count":0})
    return result
