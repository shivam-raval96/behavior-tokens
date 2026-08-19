#!/usr/bin/env python3
"""Repeated layer-20 probes on the fully paired singleton activation cache."""

from __future__ import annotations

import hashlib, json, time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES=("claude-opus-4.6","gemini-3.1-pro","gpt-5.4")
DISPLAY={"claude-opus-4.6":"Claude","gemini-3.1-pro":"Gemini","gpt-5.4":"GPT"}
SEEDS=tuple(range(42,52)); LAYER=20

def atomic_json(path:Path,value:object)->None:
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)

def checkpoint(run:Path,started:float,payload:dict)->None:
    elapsed=time.time()-started; snap={"run_id":run.name,"elapsed_seconds":round(elapsed,3),"throughput_fits_per_second":round(payload.get("completed",0)/max(elapsed,1e-9),4),"error_count":0,"retry_count":0,**payload}
    atomic_json(run/"progress.json",snap)
    with (run/"dashboard_history.jsonl").open("a") as f:f.write(json.dumps(snap,sort_keys=True)+"\n")
    (run/"dashboard.html").write_text(f"""<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=3><title>Singleton paired repeated holdout</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;background:#101418;color:#e8eef2}}progress{{width:100%}}pre{{background:#182028;padding:18px;overflow:auto}}</style><h1>Layer-20 paired repeated holdout</h1><h2>{snap['phase']}</h2><progress value="{snap.get('completed',0)}" max="{snap.get('total',30)}"></progress><pre>{json.dumps(snap,indent=2,sort_keys=True)}</pre>""")

def summarize(values:list[float])->dict:
    a=np.asarray(values,float); sd=float(a.std(ddof=1)); half=float(1.96*sd/np.sqrt(len(a))); mean=float(a.mean()); return {"mean":mean,"std":sd,"ci95_low":mean-half,"ci95_high":mean+half,"values":list(map(float,values))}

def run_experiment(source_run:str,output_run:str)->dict:
    source=Path(source_run); run=Path(output_run); run.mkdir(parents=True,exist_ok=False); (run/"probes").mkdir(); started=time.time()
    config={"run_id":run.name,"trial_type":"repeated paired task-level holdout","activation_source_run":source.name,"activation_layer":LAYER,"activation_construction":"singleton forwards only","tasks":194,"train_tasks":136,"test_tasks":58,"repeats":10,"seeds":list(SEEDS),"sources":list(SOURCES),"example_selection":"one SHA-256-ranked baseline and one sanitized trajectory per task/model, fixed across repeats","probe":{"classifier":"StandardScaler + L2 LogisticRegression","C":1.0,"class_weight":"balanced","max_iter":5000},"primary_metric":"held-out accuracy","total_fits":30}
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); config["config_fingerprint"]=fp; atomic_json(run/"resolved_config.json",config); (run/"config.yaml").write_text(json.dumps(config,indent=2,sort_keys=True)); atomic_json(run/"checkpoint.json",{"status":"running","phase":"initialized","next_fit":0,"config_fingerprint":fp})
    checkpoint(run,started,{"phase":"initialized","completed":0,"total":30,"latest_objective":None,"best_metric":None,"config_fingerprint":fp})
    rows=[json.loads(x) for x in (source/"canonical_manifest.jsonl").read_text().splitlines()]; full=np.load(source/"activations"/"all_layers.float16.npy",mmap_mode="r")
    if full.shape!=(4360,32,4096) or len(rows)!=4360:raise AssertionError((full.shape,len(rows)))
    by_key={}
    for i,row in enumerate(rows):by_key.setdefault((row["task_id"],row["source_model"],row["label"]),[]).append(i)
    tasks=sorted({r["task_id"] for r in rows}); selected=[]
    for task in tasks:
        for source_model in SOURCES:
            for label in (0,1):
                candidates=by_key.get((task,source_model,label),[])
                if not candidates:raise AssertionError((task,source_model,label))
                pick=min(candidates,key=lambda i:hashlib.sha256(f"42|{rows[i]['source_path']}".encode()).hexdigest()); selected.append(pick)
    if len(tasks)!=194 or len(selected)!=194*3*2 or len(set(selected))!=len(selected):raise AssertionError("paired selection invalid")
    chosen=[rows[i] for i in selected]; selection=[]
    for i,row in zip(selected,chosen):selection.append({"activation_row":i,"example_id":row["example_id"],"task_id":row["task_id"],"source_model":row["source_model"],"label":row["label"],"class":row["class"],"source_path":row["source_path"]})
    with (run/"selection_manifest.jsonl").open("w") as f:
        for row in selection:f.write(json.dumps(row,sort_keys=True)+"\n")
    idx=np.asarray(selected,dtype=np.int64); x=full[idx,LAYER,:]; labels=np.asarray([r["label"] for r in chosen],dtype=np.int8); models=np.asarray([r["source_model"] for r in chosen]); task_ids=np.asarray([r["task_id"] for r in chosen]); ids=np.asarray([r["example_id"] for r in chosen])
    fits=[]; split_records=[]; completed=0
    for repeat,seed in enumerate(SEEDS):
        rng=np.random.default_rng(seed); shuffled=np.asarray(tasks)[rng.permutation(len(tasks))]; train_tasks=set(shuffled[:136]); test_tasks=set(shuffled[136:])
        if len(train_tasks)!=136 or len(test_tasks)!=58 or train_tasks&test_tasks:raise AssertionError("invalid split")
        split={"repeat":repeat,"seed":seed,"train_tasks":sorted(train_tasks),"test_tasks":sorted(test_tasks)}; split_records.append(split)
        for train_source in SOURCES:
            train_mask=(models==train_source)&np.isin(task_ids,list(train_tasks)); train_idx=np.flatnonzero(train_mask)
            if len(train_idx)!=272 or np.bincount(labels[train_idx],minlength=2).tolist()!=[136,136]:raise AssertionError("unbalanced train")
            probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=seed))
            probe.fit(x[train_idx].astype(np.float32),labels[train_idx]); evaluations={}
            for eval_source in SOURCES:
                for partition,task_set in (("train",train_tasks),("test",test_tasks)):
                    mask=(models==eval_source)&np.isin(task_ids,list(task_set)); ev=np.flatnonzero(mask); expected=272 if partition=="train" else 116
                    if len(ev)!=expected or np.bincount(labels[ev],minlength=2).tolist()!=[expected//2,expected//2]:raise AssertionError("evaluation balance")
                    pred=probe.predict(x[ev].astype(np.float32)); prob=probe.predict_proba(x[ev].astype(np.float32))[:,1]
                    evaluations[f"{eval_source}|{partition}"]={"n":len(ev),"accuracy":float(accuracy_score(labels[ev],pred)),"balanced_accuracy":float(balanced_accuracy_score(labels[ev],pred)),"precision":float(precision_score(labels[ev],pred,zero_division=0)),"recall":float(recall_score(labels[ev],pred,zero_division=0)),"f1":float(f1_score(labels[ev],pred,zero_division=0)),"roc_auc":float(roc_auc_score(labels[ev],prob)),"confusion_matrix":confusion_matrix(labels[ev],pred,labels=[0,1]).tolist()}
            joblib.dump(probe,run/"probes"/f"repeat_{repeat:02d}_{DISPLAY[train_source].lower()}.joblib"); fits.append({"repeat":repeat,"seed":seed,"train_source":train_source,"train_n":len(train_idx),"evaluations":evaluations}); completed+=1
            latest=evaluations[f"{train_source}|test"]["accuracy"]; checkpoint(run,started,{"phase":"fitting","completed":completed,"total":30,"repeat":repeat,"seed":seed,"train_source":train_source,"latest_objective":latest,"best_metric":max(f["evaluations"][f"{f['train_source']}|test"]["accuracy"] for f in fits),"config_fingerprint":fp}); atomic_json(run/"partial_results.json",fits); atomic_json(run/"checkpoint.json",{"status":"running","phase":"fitting","next_fit":completed,"config_fingerprint":fp})
    atomic_json(run/"splits.json",split_records); atomic_json(run/"per_repeat_results.json",fits)
    summary={}
    for train_source in SOURCES:
        sf=[f for f in fits if f["train_source"]==train_source]; evaluations={}
        for eval_source in SOURCES:
            for partition in ("train","test"):
                key=f"{eval_source}|{partition}"; evaluations[key]={metric:summarize([f["evaluations"][key][metric] for f in sf]) for metric in ("accuracy","balanced_accuracy","precision","recall","f1","roc_auc")}
        summary[train_source]={"evaluations":evaluations}
        pos=np.arange(3); width=.36; tr=[evaluations[f"{s}|train"]["accuracy"]["mean"] for s in SOURCES]; te=[evaluations[f"{s}|test"]["accuracy"]["mean"] for s in SOURCES]; trerr=[(evaluations[f"{s}|train"]["accuracy"]["ci95_high"]-evaluations[f"{s}|train"]["accuracy"]["ci95_low"])/2 for s in SOURCES]; teerr=[(evaluations[f"{s}|test"]["accuracy"]["ci95_high"]-evaluations[f"{s}|test"]["accuracy"]["ci95_low"])/2 for s in SOURCES]
        fig,ax=plt.subplots(figsize=(8,5)); a=ax.bar(pos-width/2,tr,width,yerr=trerr,capsize=4,label="Train tasks"); b=ax.bar(pos+width/2,te,width,yerr=teerr,capsize=4,label="Held-out tasks"); ax.bar_label(a,fmt="%.3f",padding=7); ax.bar_label(b,fmt="%.3f",padding=7); ax.set_xticks(pos,[DISPLAY[s] for s in SOURCES]); ax.set_ylabel("Mean accuracy"); ax.set_ylim(0,1.1); ax.set_title(f"{DISPLAY[train_source]}-trained layer-20 probe\n10 repeated 70/30 task splits"); ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(run/f"{DISPLAY[train_source].lower()}_trained_layer20_repeated_holdout.png",dpi=180); plt.close(fig)
    means=np.array([[summary[t]["evaluations"][f"{e}|test"]["accuracy"]["mean"] for e in SOURCES] for t in SOURCES]); stds=np.array([[summary[t]["evaluations"][f"{e}|test"]["accuracy"]["std"] for e in SOURCES] for t in SOURCES]); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(means,cmap="Blues",vmin=.5,vmax=1); ax.set_xticks(range(3),[DISPLAY[s] for s in SOURCES]); ax.set_yticks(range(3),[DISPLAY[s] for s in SOURCES]); ax.set_xlabel("Evaluation source"); ax.set_ylabel("Probe training source"); ax.set_title("Layer-20 held-out accuracy");
    for i in range(3):
        for j in range(3):ax.text(j,i,f"{means[i,j]:.3f}\n± {stds[i,j]:.3f}",ha="center",va="center",color="white" if means[i,j]>=.78 else "#172033",fontweight="semibold")
    fig.colorbar(im,ax=ax,label="Mean accuracy"); fig.tight_layout(); fig.savefig(run/"layer20_cross_model_test_accuracy_heatmap.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,train_source in zip(axes,SOURCES):
        vals=[summary[train_source]["evaluations"][f"{e}|test"]["accuracy"]["values"] for e in SOURCES]; ax.boxplot(vals,tick_labels=[DISPLAY[e] for e in SOURCES],showmeans=True); ax.set_title(f"{DISPLAY[train_source]}-trained"); ax.grid(axis="y",alpha=.25); ax.set_ylim(0,1.02)
    axes[0].set_ylabel("Held-out accuracy"); fig.suptitle("Per-split layer-20 transfer accuracy"); fig.tight_layout(); fig.savefig(run/"layer20_per_repeat_test_accuracy_distributions.png",dpi=180); plt.close(fig)
    result={"status":"complete","layer":LAYER,"repeats":10,"selected_examples":len(selected),"examples_per_model":388,"config_fingerprint":fp,"summary":summary}; atomic_json(run/"results.json",result)
    lines=["# Singleton layer-20 repeated holdout","","Ten exact 136/58 task splits; one baseline and one sanitized trajectory per task/model.",""]
    for train_source in SOURCES:
        lines += [f"## {DISPLAY[train_source]}-trained","","| Evaluation source | Train accuracy | Held-out accuracy |","|---|---:|---:|"]
        for eval_source in SOURCES:
            tr=summary[train_source]["evaluations"][f"{eval_source}|train"]["accuracy"]; te=summary[train_source]["evaluations"][f"{eval_source}|test"]["accuracy"]; lines.append(f"| {DISPLAY[eval_source]} | {tr['mean']:.4f} ± {tr['std']:.4f} | {te['mean']:.4f} ± {te['std']:.4f} |")
        lines.append("")
    (run/"RESULTS.md").write_text("\n".join(lines)); atomic_json(run/"checkpoint.json",{"status":"complete","phase":"complete","next_fit":None,"config_fingerprint":fp}); checkpoint(run,started,{"phase":"complete","completed":30,"total":30,"latest_objective":None,"best_metric":None,"config_fingerprint":fp}); return result
