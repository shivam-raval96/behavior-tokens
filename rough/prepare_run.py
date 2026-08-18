from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, tarfile, urllib.request
from collections import Counter
from pathlib import Path
from terminal_wrench_probe import DATA_REV, atomic_json, render_dashboard, split_for

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",required=True); ap.add_argument("--output-root",type=Path,default=Path("/outputs/rough/runs")); a=ap.parse_args()
    run=a.output_root/a.run_id; run.mkdir(parents=True,exist_ok=False); (run/"activations").mkdir()
    render_dashboard(run,{"phase":"preparing","completed":0,"total":5984,"elapsed_seconds":0,"eta_seconds":None,"run_id":a.run_id,"error_count":0,"retry_count":0})
    archive=run/"terminal-wrench.tar.gz"; urllib.request.urlretrieve(f"https://codeload.github.com/few-sh/terminal-wrench/tar.gz/{DATA_REV}",archive)
    with tarfile.open(archive) as tf: tf.extractall(run/"source",filter="data")
    root=next((run/"source").iterdir()); rows=[]
    for path in sorted(root.glob("tasks/*/*/hack_trajectories/*/trial/agent/trajectory.json")):
        parts=path.parts; task=parts[parts.index("tasks")+1]; model=parts[parts.index("tasks")+2]; label_name=parts[parts.index("hack_trajectories")+1]
        eid=hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:20]; rows.append({"example_id":eid,"task_id":task,"source_model":model,"trajectory_label":label_name,"label":1,"split":split_for(task),"path":str(path)})
    for path in sorted(root.glob("tasks/*/*/baseline_trajectories/*/trial/agent/trajectory.json")):
        parts=path.parts; task=parts[parts.index("tasks")+1]; model=parts[parts.index("tasks")+2]; label_name=parts[parts.index("baseline_trajectories")+1]
        eid=hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:20]; rows.append({"example_id":eid,"task_id":task,"source_model":model,"trajectory_label":label_name,"label":0,"split":split_for(task),"path":str(path)})
    with (run/"manifest.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    counts=Counter((r["split"],r["label"]) for r in rows)
    config={"run_id":a.run_id,"dataset_revision":DATA_REV,"model":"Qwen/Qwen3.5-9B","seed":42,"max_length":16384,"workers":4,"layers":32,"hidden_size":4096,"counts":{f"{k[0]}_label_{k[1]}":v for k,v in counts.items()},"examples":len(rows)}
    atomic_json(run/"resolved_config.json",config); atomic_json(run/"checkpoint.json",{"phase":"prepared",**config})
    render_dashboard(run,{"phase":"prepared","completed":0,"total":len(rows),"elapsed_seconds":0,"eta_seconds":None,"run_id":a.run_id,"class_counts":config["counts"],"error_count":0,"retry_count":0})
    archive.unlink(); print(json.dumps(config,indent=2))
if __name__=="__main__":main()
