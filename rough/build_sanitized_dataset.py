from __future__ import annotations

import argparse, hashlib, json, random, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "d8a29613235a0ef56a8b70b3142626a533da28c2"
BASE = f"https://raw.githubusercontent.com/few-sh/terminal-wrench/{REVISION}"


def fetch(path: str, binary: bool = False):
    for attempt in range(9):
        try:
            request = urllib.request.Request(
                f"{BASE}/{path}", headers={"User-Agent": "behavior-tokens-research"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            return data if binary else data.decode()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 8:
                raise
            time.sleep(min(60, 2**attempt) + random.random())


def enumerate_rows(index: list[dict]) -> list[dict]:
    rows = []
    for entry in index:
        task_id, model = str(entry["task_id"]), entry["model"]
        for trajectory in entry["trajectories"]:
            if trajectory["tree_name"] != "hack_trajectories":
                continue
            label = trajectory["trajectory_label"]
            root = f"tasks/{task_id}/{model}/sanitized_trajectories/{label}"
            rows.append({"class":"sanitized_hack","label":1,"task_id":task_id,"source_model":model,"trajectory_label":label,"trajectory_path":f"{root}/trial/agent/trajectory.json","sanitizer_model_path":f"{root}/trial/agent/sanitizer_model.txt"})
        for baseline in entry["baselines"]:
            label = str(baseline["label"])
            rows.append({"class":"baseline","label":0,"task_id":task_id,"source_model":model,"trajectory_label":label,"trajectory_path":f"tasks/{task_id}/{model}/baseline_trajectories/{label}/trial/agent/trajectory.json","sanitizer_model_path":None})
    return rows


def resolve(row: dict) -> dict:
    trajectory = json.loads(fetch(row["trajectory_path"]))
    sanitizer_model = fetch(row["sanitizer_model_path"]).strip() if row["sanitizer_model_path"] else None
    example_id = hashlib.sha256(row["trajectory_path"].encode()).hexdigest()[:20]
    return {**row, "example_id":example_id, "sanitizer_model":sanitizer_model, "trajectory":trajectory}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--workers",type=int,default=12); args=parser.parse_args()
    index=json.loads(fetch("index/tasks.json")); rows=enumerate_rows(index); args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+".tmp"); counts={"sanitized_hack":0,"baseline":0}; sanitizer_models={}
    digest=hashlib.sha256(); started=time.time()
    with tmp.open("w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        out.write('{"dataset_revision":'+json.dumps(REVISION)+',"rows":[')
        for i,resolved in enumerate(pool.map(resolve,rows)):
            encoded=json.dumps(resolved,ensure_ascii=False,separators=(",",":"))
            if i: out.write(",")
            out.write(encoded); digest.update(encoded.encode()); counts[resolved["class"]]+=1
            if resolved["sanitizer_model"]: sanitizer_models[resolved["sanitizer_model"]]=sanitizer_models.get(resolved["sanitizer_model"],0)+1
            if (i+1)%100==0 or i+1==len(rows): print(json.dumps({"completed":i+1,"total":len(rows),"elapsed_seconds":time.time()-started}),flush=True)
        out.write("]}")
    tmp.replace(args.output)
    manifest={"dataset_revision":REVISION,"rows":len(rows),"class_counts":counts,"sanitizer_model_counts":sanitizer_models,"content_sha256":digest.hexdigest(),"file_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),"bytes":args.output.stat().st_size}
    args.output.with_name(args.output.stem+"_manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
