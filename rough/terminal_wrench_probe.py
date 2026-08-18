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
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"; tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    blocks = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
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
        cursor = 0
        while cursor < len(prepared):
            batch = []; tokens = 0
            while cursor + len(batch) < len(prepared) and len(batch) < 16:
                candidate = prepared[cursor + len(batch)]
                projected = candidate[0] * (len(batch) + 1)
                if batch and projected > 32768: break
                batch.append(candidate); tokens = projected
            maxlen = max(x[0] for x in batch)
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(input_ids)
            pos_list = []
            for b, (_, _, ids, pos, _) in enumerate(batch):
                offset = maxlen - len(ids); input_ids[b, offset:] = torch.tensor(ids); mask[b, offset:] = 1; pos_list.append(offset + pos)
            input_ids, mask = input_ids.cuda(), mask.cuda(); positions = torch.tensor(pos_list, device="cuda")
            with torch.inference_mode(): model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            values = torch.stack(captured, dim=1).numpy()
            for b, (_, row, _, _, audit) in enumerate(batch):
                np.save(shard_dir / f"{row['example_id']}.npy", values[b], allow_pickle=False)
                atomic_json(shard_dir / f"{row['example_id']}.json", {**row, **audit})
                if len(audit_rows) < 25: audit_rows.append({"example_id": row["example_id"], **audit})
            completed += len(batch); cursor += len(batch)
            if completed % 100 < len(batch) or completed == len(rows):
                elapsed=time.time()-started; rate=completed/max(elapsed,1e-6)
                snap={"phase":"extracting","worker":shard,"completed":completed,"total":len(rows),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(rows)-completed)/rate,"error_count":0,"retry_count":0}
                atomic_json(shard_dir/"progress.json",snap); render_dashboard(run, snap); checkpoint_callback(); print(json.dumps(snap),flush=True)
    finally:
        for h in hooks: h.remove()
    atomic_json(shard_dir/"boundary_audit.json",audit_rows); checkpoint_callback()
    return {"worker": shard, "completed": completed}


def fit_probes(run_dir: str) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    run=Path(run_dir); rows=[json.loads(x) for x in (run/"manifest.jsonl").read_text().splitlines()]
    arrays=[]
    for i,row in enumerate(rows): arrays.append(np.load(run/"activations"/f"worker-{i%4:02d}"/f"{row['example_id']}.npy"))
    x=np.stack(arrays); y=np.array([r["label"] for r in rows]); train=np.array([r["split"]=="train" for r in rows]); test=~train
    np.save(run/"activations"/"all_layers.float16.npy",x,allow_pickle=False)
    metrics=[]
    for layer in range(LAYERS):
        probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=42,n_jobs=1))
        probe.fit(x[train,layer].astype(np.float32),y[train]); pred=probe.predict(x[test,layer].astype(np.float32)); prob=probe.predict_proba(x[test,layer].astype(np.float32))[:,1]
        m={"layer":layer,"test_accuracy":accuracy_score(y[test],pred),"balanced_accuracy":balanced_accuracy_score(y[test],pred),"roc_auc":roc_auc_score(y[test],prob),"precision":precision_score(y[test],pred),"recall":recall_score(y[test],pred),"f1":f1_score(y[test],pred),"confusion_matrix":confusion_matrix(y[test],pred).tolist()}
        metrics.append(m); joblib.dump(probe,run/f"probe_layer_{layer:02d}.joblib")
        print(json.dumps({"phase":"fitting","completed":layer+1,"total":LAYERS,**m}),flush=True)
    result={"primary_layer":31,"test_accuracy":metrics[31]["test_accuracy"],"per_layer":metrics,"train_examples":int(train.sum()),"test_examples":int(test.sum()),"status":"complete"}
    atomic_json(run/"results.json",result); atomic_json(run/"checkpoint.json",result)
    run.joinpath("RESULTS.md").write_text(f"# Terminal Wrench last-token probe\n\n- Primary layer: 31\n- Test accuracy: {result['test_accuracy']:.4f}\n- Train/test: {train.sum()}/{test.sum()}\n- No generations were performed.\n")
    return result
