from __future__ import annotations
import argparse, hashlib, json, random, time, urllib.error, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from terminal_wrench_probe import DATA_REV, atomic_json, render_dashboard, split_for

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",required=True); ap.add_argument("--output-root",type=Path,default=Path("/outputs/rough/runs")); a=ap.parse_args()
    run=a.output_root/a.run_id; run.mkdir(parents=True,exist_ok=True); (run/"activations").mkdir(exist_ok=True)
    render_dashboard(run,{"phase":"preparing","completed":0,"total":5984,"elapsed_seconds":0,"eta_seconds":None,"run_id":a.run_id,"error_count":0,"retry_count":0})
    base=f"https://raw.githubusercontent.com/few-sh/terminal-wrench/{DATA_REV}"
    with urllib.request.urlopen(base+"/index/tasks.json") as response: index=json.load(response)
    source=run/"source"/"trajectories"; source.mkdir(parents=True,exist_ok=True); rows=[]
    for entry in index:
        task=str(entry["task_id"]); model=entry["model"]
        for item in entry["trajectories"]:
            if item["tree_name"] != "hack_trajectories": continue
            rel=f"tasks/{task}/{model}/hack_trajectories/{item['trajectory_label']}/trial/agent/trajectory.json"
            eid=hashlib.sha256(rel.encode()).hexdigest()[:20]; rows.append({"example_id":eid,"task_id":task,"source_model":model,"trajectory_label":item["trajectory_label"],"label":1,"split":split_for(task),"path":str(source/f"{eid}.json"),"source_path":rel})
        for item in entry["baselines"]:
            label_name=str(item["label"]); rel=f"tasks/{task}/{model}/baseline_trajectories/{label_name}/trial/agent/trajectory.json"
            eid=hashlib.sha256(rel.encode()).hexdigest()[:20]; rows.append({"example_id":eid,"task_id":task,"source_model":model,"trajectory_label":label_name,"label":0,"split":split_for(task),"path":str(source/f"{eid}.json"),"source_path":rel})
    started=time.time()
    def download(row):
        target=Path(row["path"])
        if target.exists() and target.stat().st_size: return
        for attempt in range(9):
            try:
                request=urllib.request.Request(base+"/"+row["source_path"],headers={"User-Agent":"behavior-tokens-research"})
                with urllib.request.urlopen(request,timeout=120) as response: target.write_bytes(response.read())
                return
            except urllib.error.HTTPError as exc:
                if exc.code not in {429,500,502,503,504} or attempt == 8: raise
                time.sleep(min(60,2**attempt)+random.random())
    with ThreadPoolExecutor(max_workers=16) as pool:
        for completed,_ in enumerate(pool.map(download,rows),1):
            if completed%100==0 or completed==len(rows):
                elapsed=time.time()-started; rate=completed/max(elapsed,1e-6)
                render_dashboard(run,{"phase":"downloading_dataset","completed":completed,"total":len(rows),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(rows)-completed)/rate,"run_id":a.run_id,"error_count":0,"retry_count":0})
    with (run/"manifest.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    counts=Counter((r["split"],r["label"]) for r in rows)
    config={"run_id":a.run_id,"dataset_revision":DATA_REV,"model":"Qwen/Qwen3.5-9B","seed":42,"max_length":16384,"workers":4,"layers":32,"hidden_size":4096,"counts":{f"{k[0]}_label_{k[1]}":v for k,v in counts.items()},"examples":len(rows)}
    atomic_json(run/"resolved_config.json",config); atomic_json(run/"checkpoint.json",{"phase":"prepared",**config})
    render_dashboard(run,{"phase":"prepared","completed":0,"total":len(rows),"elapsed_seconds":0,"eta_seconds":None,"run_id":a.run_id,"class_counts":config["counts"],"error_count":0,"retry_count":0})
    partial=run/"terminal-wrench.tar.gz"
    if partial.exists(): partial.unlink()
    print(json.dumps(config,indent=2))
if __name__=="__main__":main()
