from __future__ import annotations

import hashlib, json, math, os, random, shutil, time, urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from terminal_wrench_probe import DATA_REV, HIDDEN, LAYERS, MAX_LEN, MODEL, atomic_json, render_dashboard, tokenize_with_boundary, trajectory_messages

MODELS = ["claude-opus-4.6", "gemini-3.1-pro", "gpt-5.4"]
OLD_RUN = "2026-08-18_224146Z_qwen35-9b-sanitized-cross-model-probe"
WORKERS = 8


def prepare_repair(run_dir: str) -> dict:
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=False); (run / "source").mkdir(); (run / "activations").mkdir()
    started=time.time(); render_dashboard(run,{"phase":"preparing","completed":0,"total":1284,"elapsed_seconds":0,"eta_seconds":None,"run_id":run.name,"error_count":0,"retry_count":0})
    old_run=run.parent/OLD_RUN
    old=[json.loads(x) for x in (old_run/"manifest.jsonl").read_text().splitlines()]
    old_ids={r["example_id"] for r in old}; tasks={r["task_id"] for r in old}
    base=f"https://raw.githubusercontent.com/few-sh/terminal-wrench/{DATA_REV}"
    with urllib.request.urlopen(base+"/index/tasks.json") as response: index=json.load(response)
    entries={(str(e["task_id"]),e["model"]):e for e in index}
    complete=[]
    for task in sorted(tasks):
        for model in MODELS:
            entry=entries[(task,model)]
            items=[("baseline",str(x["label"])) for x in entry["baselines"]]
            items += [("sanitized_hack",x["trajectory_label"]) for x in entry["trajectories"] if x["tree_name"]=="hack_trajectories"]
            for cls,label in items:
                tree="baseline_trajectories" if cls=="baseline" else "sanitized_trajectories"
                rel=f"tasks/{task}/{model}/{tree}/{label}/trial/agent/trajectory.json"
                eid=hashlib.sha256(rel.encode()).hexdigest()[:20]
                complete.append({"example_id":eid,"task_id":task,"source_model":model,"class":cls,"label":int(cls=="sanitized_hack"),"trajectory_label":label,"source_path":rel})
    complete=sorted(complete,key=lambda r:r["example_id"])
    missing=[r for r in complete if r["example_id"] not in old_ids]
    if len(complete)!=4360 or len({r['example_id'] for r in complete})!=4360 or len(missing)!=1284: raise AssertionError((len(complete),len(missing)))
    # Materialize missing trajectory JSON and use byte size as an LPT sharding proxy.
    for i,row in enumerate(missing,1):
        target=run/"source"/f"{row['example_id']}.json"
        with urllib.request.urlopen(base+"/"+row["source_path"],timeout=120) as response: target.write_bytes(response.read())
        row["path"]=str(target); row["estimated_bytes"]=target.stat().st_size
        if i%100==0 or i==len(missing):
            elapsed=time.time()-started; rate=i/max(elapsed,1e-6)
            render_dashboard(run,{"phase":"downloading_missing","completed":i,"total":len(missing),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(missing)-i)/rate,"run_id":run.name,"error_count":0,"retry_count":0})
    loads=[0]*WORKERS; shards=[[] for _ in range(WORKERS)]
    for row in sorted(missing,key=lambda r:r["estimated_bytes"],reverse=True):
        w=min(range(WORKERS),key=lambda j:loads[j]); row["worker"]=w; shards[w].append(row); loads[w]+=row["estimated_bytes"]
    missing=sorted(missing,key=lambda r:(r["worker"],r["example_id"])); complete_by_id={r["example_id"]:r for r in complete}
    # Preserve old provenance and attach canonical rows to all entries.
    old_by_id={r["example_id"]:r for r in old}
    canonical=[]
    for idx,eid in enumerate(sorted(complete_by_id)):
        row=dict(old_by_id.get(eid,complete_by_id[eid])); row["canonical_row"]=idx; canonical.append(row)
    for name,rows in (("missing_manifest.jsonl",missing),("canonical_manifest.jsonl",canonical)):
        with (run/name).open("w") as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    config={"run_id":run.name,"trial_type":"activation-cache repair","source_run":OLD_RUN,"model":MODEL,"dataset_revision":DATA_REV,"fixed_tasks":194,"existing_rows":3076,"missing_rows":1284,"complete_rows":4360,"workers":WORKERS,"gpu":"A100-80GB","layers":LAYERS,"hidden_size":HIDDEN,"max_length":MAX_LEN,"initial_token_budget":32768,"maximum_token_budget":65536,"maximum_batch_size":32,"assignment":"LPT by trajectory byte size","activation_position":"last assistant content token immediately before <|im_end|>"}
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); config["config_fingerprint"]=fp
    atomic_json(run/"resolved_config.json",config); (run/"config.yaml").write_text(json.dumps(config,indent=2,sort_keys=True)); atomic_json(run/"checkpoint.json",{"status":"running","phase":"prepared","next_step":"extract","config_fingerprint":fp})
    return config


def aggregate_dashboard(run: Path, run_id: str) -> None:
    progress=[]
    for p in sorted((run/"activations").glob("worker-*/progress.json")):
        try:progress.append(json.loads(p.read_text()))
        except Exception:pass
    completed=sum(x.get("completed",0) for x in progress); config=json.loads((run/"resolved_config.json").read_text()); total=config.get("extraction_rows",config.get("missing_rows",1284))
    elapsed=max([x.get("elapsed_seconds",0) for x in progress] or [0]); rate=sum(x.get("throughput",0) for x in progress)
    snap={"phase":"extracting","completed":completed,"total":total,"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(total-completed)/rate if rate else None,"run_id":run_id,"workers_reporting":len(progress),"worker_progress":progress,"error_count":sum(x.get("oom_count",0) for x in progress),"retry_count":sum(x.get("retry_count",0) for x in progress)}
    atomic_json(run/"progress.json",snap); render_dashboard(run,snap)


def extract_shard(run_dir: str, worker: int, commit=lambda:None) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    run=Path(run_dir); config=json.loads((run/"resolved_config.json").read_text()); checkpoint=json.loads((run/"checkpoint.json").read_text())
    if checkpoint.get("config_fingerprint") != config.get("config_fingerprint"): raise ValueError("resume config fingerprint mismatch")
    manifest_name=config.get("extraction_manifest","missing_manifest.jsonl")
    rows=[json.loads(x) for x in (run/manifest_name).read_text().splitlines()]; rows=[r for r in rows if r["worker"]==worker]
    outdir=run/"activations"/f"worker-{worker:02d}"; outdir.mkdir(parents=True,exist_ok=True)
    singleton=config.get("forward_batch_size")==1
    if singleton:
        rejected=outdir/"rejected_batched"; rejected.mkdir(exist_ok=True)
        for meta_path in list(outdir.glob("batch-*.json")):
            if len(json.loads(meta_path.read_text()).get("example_ids",[])) != 1:
                npy=meta_path.with_suffix(".npy")
                if npy.exists(): shutil.move(str(npy),rejected/npy.name)
                shutil.move(str(meta_path),rejected/meta_path.name)
    done=set()
    for p in outdir.glob("batch-*.json"):
        done.update(json.loads(p.read_text())["example_ids"])
    rows=[r for r in rows if r["example_id"] not in done]
    model_id=config.get("model",MODEL); revision=config.get("model_revision")
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision); tokenizer.padding_side="left"; tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda").eval()
    core=model.model.language_model if hasattr(model.model,"language_model") else model.model; captured=[None]*LAYERS; positions=None
    hooks=[]
    for layer_idx,block in enumerate(core.layers):
        def hook(_m,_i,o,idx=layer_idx):
            h=o[0] if isinstance(o,tuple) else o; captured[idx]=h[torch.arange(h.shape[0],device=h.device),positions].detach().cpu().to(torch.float16)
        hooks.append(block.register_forward_hook(hook))
    prepared=[]
    for row in rows:
        data=json.loads(Path(row["path"]).read_text()); ids,pos,audit=tokenize_with_boundary(tokenizer,trajectory_messages(data)); prepared.append((len(ids),row,ids,pos,audit))
    prepared.sort(key=lambda x:x[0]); cursor=0; batch_no=len(list(outdir.glob("batch-*.npy"))); started=time.time(); token_budget=32768; attention_budget=32768**2; max_batch=1 if singleton else 32; successes=0; oom=0; validation=None
    try:
        while cursor<len(prepared):
            batch=[]
            while cursor+len(batch)<len(prepared) and len(batch)<max_batch:
                cand=prepared[cursor+len(batch)]; n=len(batch)+1; maxlen=cand[0]
                if batch and (maxlen*n>token_budget or maxlen*maxlen*n>attention_budget):break
                batch.append(cand)
            maxlen=max(x[0] for x in batch); ids_t=torch.full((len(batch),maxlen),tokenizer.pad_token_id,dtype=torch.long); mask=torch.zeros_like(ids_t); pos=[]
            for b,(_,_,ids,p,_) in enumerate(batch):
                off=maxlen-len(ids); ids_t[b,off:]=torch.tensor(ids); mask[b,off:]=1; pos.append(off+p)
            ids_t=ids_t.cuda(); mask=mask.cuda(); positions=torch.tensor(pos,device="cuda")
            try:
                with torch.inference_mode():core(input_ids=ids_t,attention_mask=mask,use_cache=False)
                values=torch.stack(captured,dim=1).numpy()
                if not np.isfinite(values).all(): raise FloatingPointError("non-finite activation detected")
                if validation is None:
                    first=values[0].copy(); one_ids=ids_t[:1]; one_mask=mask[:1]; positions=positions[:1]
                    # Repeat the identical padded batch shape. Comparing a sample
                    # batched versus alone selects different SDPA kernels and is
                    # not expected to be bit-identical in bfloat16.
                    positions=torch.tensor(pos,device="cuda")
                    with torch.inference_mode():core(input_ids=ids_t,attention_mask=mask,use_cache=False)
                    again=torch.stack(captured,dim=1).numpy()[0]; diff=float(np.max(np.abs(first.astype(np.float32)-again.astype(np.float32))))
                    validation={"example_id":batch[0][1]["example_id"],"repeat_max_abs_diff":diff,"validation_shape":"identical_padded_batch"}; atomic_json(outdir/"repeat_validation.json",validation)
                    if diff!=0:raise AssertionError(validation)
            except torch.OutOfMemoryError:
                oom+=1; torch.cuda.empty_cache(); token_budget=max(maxlen,token_budget//2); attention_budget=max(maxlen*maxlen,attention_budget//2); successes=0; continue
            meta={"example_ids":[x[1]["example_id"] for x in batch],"audits":[{"example_id":x[1]["example_id"],**x[4]} for x in batch],"shape":list(values.shape)}
            np.save(outdir/f"batch-{batch_no:05d}.npy",values,allow_pickle=False); atomic_json(outdir/f"batch-{batch_no:05d}.json",meta); batch_no+=1; cursor+=len(batch); successes+=1
            if not singleton and successes%3==0:token_budget=min(65536,int(token_budget*1.25)); attention_budget=min(2*32768**2,int(attention_budget*1.25))
            completed=len(done)+cursor; elapsed=time.time()-started; rate=cursor/max(elapsed,1e-6)
            atomic_json(outdir/"progress.json",{"worker":worker,"phase":"extracting","completed":completed,"total":len(done)+len(prepared),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(prepared)-cursor)/rate,"token_budget":token_budget,"attention_budget":attention_budget,"max_batch_size":max_batch,"oom_count":oom,"retry_count":0}); commit(); aggregate_dashboard(run,run.name); commit()
    finally:
        for h in hooks:h.remove()
    return {"worker":worker,"completed":len(done)+len(prepared),"repeat_validation":validation,"oom_count":oom}


def configure_singleton(run_dir: str) -> dict:
    run=Path(run_dir); config=json.loads((run/"resolved_config.json").read_text())
    config["forward_batch_size"]=1; config["parallel_workers"]=WORKERS
    config["batching_decision"]="disabled after strict batched-versus-singleton validation failure"
    prior=config.pop("config_fingerprint",None); fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); config["config_fingerprint"]=fp
    atomic_json(run/"resolved_config.json",config); (run/"config.yaml").write_text(json.dumps(config,indent=2,sort_keys=True))
    atomic_json(run/"checkpoint.json",{"status":"running","phase":"configured_singleton","next_step":"extract","config_fingerprint":fp,"prior_config_fingerprint":prior})
    return config


def prepare_full_singleton(run_dir: str, prepared_run_id: str, model_id: str=MODEL, model_revision: str|None=None) -> dict:
    """Create a fresh singleton extraction manifest from the repaired source index."""
    run=Path(run_dir); run.mkdir(parents=True,exist_ok=False); (run/"activations").mkdir()
    prepared=run.parent/prepared_run_id
    rows=[json.loads(x) for x in (prepared/"canonical_manifest.jsonl").read_text().splitlines()]
    if len(rows)!=4360 or len({r["example_id"] for r in rows})!=4360:raise AssertionError("canonical source is incomplete")
    loads=[0]*WORKERS
    for row in sorted(rows,key=lambda r:Path(r["path"]).stat().st_size,reverse=True):
        worker=min(range(WORKERS),key=lambda i:loads[i]); size=Path(row["path"]).stat().st_size; row["worker"]=worker; row["estimated_bytes"]=size; loads[worker]+=size
    rows=sorted(rows,key=lambda r:(r["worker"],r["example_id"]))
    with (run/"extraction_manifest.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    canonical=sorted(rows,key=lambda r:r["example_id"])
    for i,row in enumerate(canonical):row["canonical_row"]=i
    with (run/"canonical_manifest.jsonl").open("w") as f:
        for row in canonical:f.write(json.dumps(row,sort_keys=True)+"\n")
    config={"run_id":run.name,"trial_type":"fresh singleton activation extraction","construction":"fresh_singleton","prepared_source_run":prepared_run_id,"model":model_id,"model_revision":model_revision,"dataset_revision":DATA_REV,"fixed_tasks":194,"extraction_rows":4360,"complete_rows":4360,"extraction_manifest":"extraction_manifest.jsonl","workers":WORKERS,"parallel_workers":WORKERS,"forward_batch_size":1,"gpu":"A100-80GB","layers":LAYERS,"hidden_size":HIDDEN,"max_length":MAX_LEN,"activation_position":"last assistant content token immediately before <|im_end|>","storage":"separate; no activation values reused from prior cache"}
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); config["config_fingerprint"]=fp
    atomic_json(run/"resolved_config.json",config); (run/"config.yaml").write_text(json.dumps(config,indent=2,sort_keys=True)); atomic_json(run/"checkpoint.json",{"status":"running","phase":"prepared","next_step":"extract","config_fingerprint":fp})
    snap={"phase":"prepared","completed":0,"total":4360,"elapsed_seconds":0,"eta_seconds":None,"run_id":run.name,"config_fingerprint":fp,"error_count":0,"retry_count":0}; atomic_json(run/"progress.json",snap); render_dashboard(run,snap)
    return config


def finalize_full_singleton(run_dir: str, commit=lambda:None) -> dict:
    run=Path(run_dir); canonical=[json.loads(x) for x in (run/"canonical_manifest.jsonl").read_text().splitlines()]; n=len(canonical); index={r["example_id"]:r["canonical_row"] for r in canonical}
    target=run/"activations"/"all_layers.float16.npy"; out=np.lib.format.open_memmap(target,mode="w+",dtype=np.float16,shape=(n,LAYERS,HIDDEN)); filled=np.zeros(n,dtype=bool); audits=[]; count=0
    compact_meta=sorted((run/"activations").glob("worker-*/compact.json"))
    if len(compact_meta)!=WORKERS:raise AssertionError(f"expected {WORKERS} compact shards, found {len(compact_meta)}")
    for meta_path in compact_meta:
        meta=json.loads(meta_path.read_text()); vals=np.load(meta_path.with_suffix(".npy"),mmap_mode="r")
        if vals.shape!=(len(meta["example_ids"]),LAYERS,HIDDEN):raise AssertionError((meta_path,vals.shape))
        for i,eid in enumerate(meta["example_ids"]):out[index[eid]]=vals[i]; filled[index[eid]]=True; count+=1
        audits.extend(meta["audits"])
    out.flush()
    if count!=n or not filled.all() or len(index)!=n:raise AssertionError((count,int(filled.sum()),len(index),n))
    storage_checks=[]
    rng=random.Random(42)
    compact_lookup={}
    for p in compact_meta:
        meta=json.loads(p.read_text())
        for i,eid in enumerate(meta["example_ids"]):compact_lookup[eid]=(p.with_suffix(".npy"),i)
    for row in rng.sample(canonical,20):
        p,i=compact_lookup[row["example_id"]]; v=np.load(p,mmap_mode="r")[i]; same=bool(np.array_equal(v,out[row["canonical_row"]])); storage_checks.append({"example_id":row["example_id"],"exact":same})
    repeat=[json.loads(p.read_text()) for p in sorted((run/"activations").glob("worker-*/repeat_validation.json"))]
    pairs=defaultdict(set)
    for r in canonical:pairs[(r["task_id"],r["source_model"])].add(r["label"])
    result={"status":"complete_pending_fresh_forward_audit","shape":[n,LAYERS,HIDDEN],"dtype":"float16","rows":n,"unique_ids":len(index),"singleton_batch_files":count,"boundary_audits":len(audits),"task_model_pairs":len(pairs),"all_pairs_have_both_labels":len(pairs)==582 and all(v=={0,1} for v in pairs.values()),"repeat_forward_checks":repeat,"storage_checks":storage_checks,"remote_activation_path":str(target)}
    if len(repeat)!=WORKERS or not all(x.get("repeat_max_abs_diff")==0 for x in repeat) or not all(x["exact"] for x in storage_checks):raise AssertionError("singleton validation failed")
    atomic_json(run/"results.json",result); atomic_json(run/"checkpoint.json",{"status":"running","phase":"pending_fresh_forward_audit","next_step":"fresh_forward_audit"}); commit(); return result


def compact_worker(run_dir: str, worker: int, commit=lambda:None) -> dict:
    run=Path(run_dir); d=run/"activations"/f"worker-{worker:02d}"; metas=sorted(d.glob("batch-*.json")); ids=[]; audits=[]; arrays=[]
    for p in metas:
        meta=json.loads(p.read_text()); value=np.load(p.with_suffix(".npy"))
        if len(meta["example_ids"])!=1 or value.shape!=(1,LAYERS,HIDDEN):raise AssertionError((p,value.shape))
        ids.extend(meta["example_ids"]); audits.extend(meta["audits"]); arrays.append(value[0])
    manifest=json.loads((run/"resolved_config.json").read_text()); expected=sum(1 for line in (run/manifest["extraction_manifest"]).read_text().splitlines() if json.loads(line)["worker"]==worker)
    if len(ids)!=expected or len(set(ids))!=expected:raise AssertionError((worker,len(ids),expected))
    values=np.stack(arrays,axis=0).astype(np.float16,copy=False); np.save(d/"compact.npy",values,allow_pickle=False); atomic_json(d/"compact.json",{"worker":worker,"example_ids":ids,"audits":audits,"shape":list(values.shape)}); commit(); return {"worker":worker,"rows":len(ids),"shape":list(values.shape)}


def compact_all_workers(run_dir: str) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda worker: compact_worker(run_dir,worker),range(WORKERS)))


def finalize_repair(run_dir: str, commit=lambda:None) -> dict:
    run=Path(run_dir); old_run=run.parent/OLD_RUN; old_rows=[json.loads(x) for x in (old_run/"manifest.jsonl").read_text().splitlines()]; canonical=[json.loads(x) for x in (run/"canonical_manifest.jsonl").read_text().splitlines()]
    index={r["example_id"]:r["canonical_row"] for r in canonical}; target=run/"activations"/"all_layers.float16.npy"; out=np.lib.format.open_memmap(target,mode="w+",dtype=np.float16,shape=(4360,LAYERS,HIDDEN)); filled=np.zeros(4360,dtype=bool)
    old=np.load(old_run/"activations"/"all_layers.float16.npy",mmap_mode="r")
    for i,row in enumerate(old_rows):out[index[row["example_id"]]]=old[i]; filled[index[row["example_id"]]]=True
    new_count=0; audits=[]
    for meta_path in sorted((run/"activations").glob("worker-*/batch-*.json")):
        meta=json.loads(meta_path.read_text()); vals=np.load(meta_path.with_suffix(".npy"),mmap_mode="r")
        if vals.shape!=(len(meta["example_ids"]),LAYERS,HIDDEN):raise AssertionError((meta_path,vals.shape))
        for i,eid in enumerate(meta["example_ids"]):out[index[eid]]=vals[i]; filled[index[eid]]=True; new_count+=1
        audits.extend(meta["audits"])
    out.flush()
    if new_count!=1284 or not filled.all():raise AssertionError((new_count,int(filled.sum())))
    # Byte-identical preservation of sampled old rows.
    rng=random.Random(42); checks=[]
    for i in rng.sample(range(len(old_rows)),10):
        same=bool(np.array_equal(old[i],out[index[old_rows[i]["example_id"]]])); checks.append({"example_id":old_rows[i]["example_id"],"byte_identical":same})
    if not all(x["byte_identical"] for x in checks):raise AssertionError(checks)
    pairs=defaultdict(set)
    for r in canonical:pairs[(r["task_id"],r["source_model"])].add(r["label"])
    repeat=[json.loads(p.read_text()) for p in sorted((run/"activations").glob("worker-*/repeat_validation.json"))]
    result={"status":"complete","shape":[4360,LAYERS,HIDDEN],"dtype":"float16","existing_rows":3076,"new_rows":new_count,"unique_ids":len(index),"task_model_pairs":len(pairs),"all_pairs_have_both_labels":len(pairs)==582 and all(v=={0,1} for v in pairs.values()),"boundary_audits":len(audits),"repeat_forward_checks":repeat,"old_row_preservation_checks":checks,"remote_activation_path":str(target)}
    atomic_json(run/"results.json",result); (run/"RESULTS.md").write_text("# Activation cache repair\n\n"+"\n".join(f"- **{k}:** `{v}`" for k,v in result.items() if k not in {"repeat_forward_checks","old_row_preservation_checks"})+"\n")
    atomic_json(run/"checkpoint.json",{"status":"complete","phase":"complete","next_step":None}); snap={"phase":"complete","completed":1284,"total":1284,"elapsed_seconds":0,"eta_seconds":0,"run_id":run.name,"latest_objective":1284,"best_metric":1284,"error_count":0,"retry_count":0,**result}; atomic_json(run/"progress.json",snap); render_dashboard(run,snap); commit(); return result


def validate_batch_equivalence(run_dir: str, commit=lambda:None) -> dict:
    """Quantify padded-batch versus singleton numerical equivalence."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    run=Path(run_dir); rows=[json.loads(x) for x in (run/"missing_manifest.jsonl").read_text().splitlines() if json.loads(x)["worker"]==0]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); tokenizer.padding_side="left"; tokenizer.pad_token=tokenizer.eos_token
    prepared=[]
    for row in rows:
        data=json.loads(Path(row["path"]).read_text()); ids,pos,audit=tokenize_with_boundary(tokenizer,trajectory_messages(data)); prepared.append((len(ids),row,ids,pos,audit))
    prepared=sorted(prepared,key=lambda x:x[0])[:4]
    model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",local_files_only=True).eval()
    core=model.model.language_model if hasattr(model.model,"language_model") else model.model; captured=[None]*LAYERS; positions=None; hooks=[]
    for layer_idx,block in enumerate(core.layers):
        def hook(_m,_i,o,idx=layer_idx):
            h=o[0] if isinstance(o,tuple) else o; captured[idx]=h[torch.arange(h.shape[0],device=h.device),positions].detach().float().cpu()
        hooks.append(block.register_forward_hook(hook))
    try:
        maxlen=max(x[0] for x in prepared); ids_t=torch.full((len(prepared),maxlen),tokenizer.pad_token_id,dtype=torch.long); mask=torch.zeros_like(ids_t); pos=[]
        for b,(_,_,ids,p,_) in enumerate(prepared):
            off=maxlen-len(ids); ids_t[b,off:]=torch.tensor(ids); mask[b,off:]=1; pos.append(off+p)
        ids_t=ids_t.cuda(); mask=mask.cuda(); positions=torch.tensor(pos,device="cuda")
        with torch.inference_mode():core(input_ids=ids_t,attention_mask=mask,use_cache=False)
        batched=torch.stack(captured,dim=1)
        examples=[]
        for b,(_,row,ids,p,audit) in enumerate(prepared):
            one=torch.tensor(ids,dtype=torch.long,device="cuda")[None,:]; one_mask=torch.ones_like(one); positions=torch.tensor([p],device="cuda")
            with torch.inference_mode():core(input_ids=one,attention_mask=one_mask,use_cache=False)
            single=torch.stack(captured,dim=1)[0]; batch=batched[b]
            diff=batch-single; rel=(torch.linalg.vector_norm(diff,dim=1)/torch.linalg.vector_norm(single,dim=1).clamp_min(1e-12)); cos=torch.nn.functional.cosine_similarity(batch,single,dim=1)
            examples.append({"example_id":row["example_id"],"token_id":audit["token_id"],"token":audit["token"],"max_abs_diff":float(diff.abs().max()),"max_relative_l2":float(rel.max()),"min_cosine_similarity":float(cos.min()),"per_layer_relative_l2":rel.tolist(),"per_layer_cosine_similarity":cos.tolist()})
        result={"status":"pass" if min(x["min_cosine_similarity"] for x in examples)>=0.9999 and max(x["max_relative_l2"] for x in examples)<=0.01 else "fail","criteria":{"minimum_cosine_similarity":0.9999,"maximum_relative_l2":0.01},"examples":examples,"minimum_cosine_similarity":min(x["min_cosine_similarity"] for x in examples),"maximum_relative_l2":max(x["max_relative_l2"] for x in examples),"maximum_absolute_difference":max(x["max_abs_diff"] for x in examples)}
        atomic_json(run/"batch_singleton_validation.json",result); commit(); return result
    finally:
        for h in hooks:h.remove()


def audit_saved_singletons(run_dir: str, commit=lambda:None) -> dict:
    """Freshly re-forward deterministic rows and compare to the merged store."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    run=Path(run_dir); config=json.loads((run/"resolved_config.json").read_text()); rows=[json.loads(x) for x in (run/"canonical_manifest.jsonl").read_text().splitlines()]; sample=random.Random(2026).sample(rows,5)
    saved=np.load(run/"activations"/"all_layers.float16.npy",mmap_mode="r")
    model_id=config.get("model",MODEL); revision=config.get("model_revision")
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision); model=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda").eval(); core=model.model.language_model if hasattr(model.model,"language_model") else model.model
    captured=[None]*LAYERS; position=None; hooks=[]
    for idx,block in enumerate(core.layers):
        def hook(_m,_i,o,layer=idx):
            h=o[0] if isinstance(o,tuple) else o; captured[layer]=h[0,position].detach().cpu().to(torch.float16)
        hooks.append(block.register_forward_hook(hook))
    checks=[]
    try:
        for row in sample:
            data=json.loads(Path(row["path"]).read_text()); ids,pos,audit=tokenize_with_boundary(tokenizer,trajectory_messages(data)); position=pos; ids_t=torch.tensor(ids,dtype=torch.long,device="cuda")[None,:]; mask=torch.ones_like(ids_t)
            with torch.inference_mode():core(input_ids=ids_t,attention_mask=mask,use_cache=False)
            fresh=torch.stack(captured,dim=0).numpy(); prior=saved[row["canonical_row"]]; diff=np.abs(fresh.astype(np.float32)-prior.astype(np.float32)); checks.append({"example_id":row["example_id"],"token_id":audit["token_id"],"exact":bool(np.array_equal(fresh,prior)),"max_abs_diff":float(diff.max())})
    finally:
        for h in hooks:h.remove()
    passed=all(x["exact"] for x in checks); audit={"status":"pass" if passed else "fail","sample_size":len(checks),"checks":checks}; atomic_json(run/"fresh_forward_audit.json",audit)
    result=json.loads((run/"results.json").read_text()); result["fresh_forward_audit"]=audit; result["status"]="complete" if passed else "failed_fresh_forward_audit"; atomic_json(run/"results.json",result)
    if passed:
        atomic_json(run/"checkpoint.json",{"status":"complete","phase":"complete","next_step":None}); (run/"RESULTS.md").write_text(f"# Full singleton activation cache\n\n- Rows: **{result['rows']}**\n- Shape: `{result['shape']}`\n- Unique IDs: **{result['unique_ids']}**\n- Boundary audits: **{result['boundary_audits']}**\n- Fresh singleton re-forward audit: **pass ({len(checks)}/{len(checks)})**\n- Remote tensor: `{result['remote_activation_path']}`\n")
        snap={"phase":"complete","completed":result["rows"],"total":result["rows"],"elapsed_seconds":0,"eta_seconds":0,"run_id":run.name,"latest_objective":1.0,"best_metric":1.0,"error_count":0,"retry_count":0,**result}; atomic_json(run/"progress.json",snap); render_dashboard(run,snap)
    commit()
    if not passed:raise AssertionError(audit)
    return audit
