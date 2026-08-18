from __future__ import annotations

import argparse, hashlib, json, random, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from terminal_wrench_probe import DATA_REV, MODEL, atomic_json, render_dashboard, split_for

MODELS = ["claude-opus-4.6", "gemini-3.1-pro", "gpt-5.4"]
CLASSES = ["baseline", "sanitized_hack"]


def stable_rank(row: dict) -> str:
    return hashlib.sha256(f"42|{row['source_path']}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-root", type=Path, default=Path("/outputs/rough/runs"))
    a = ap.parse_args()
    run = a.output_root / a.run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "activations").mkdir()
    started = time.time()
    render_dashboard(run, {"phase":"preparing","completed":0,"total":1,"elapsed_seconds":0,"eta_seconds":None,"run_id":a.run_id,"error_count":0,"retry_count":0})
    base = f"https://raw.githubusercontent.com/few-sh/terminal-wrench/{DATA_REV}"
    with urllib.request.urlopen(base + "/index/tasks.json") as response:
        index = json.load(response)

    entries = {(str(e["task_id"]), e["model"]): e for e in index}
    availability = defaultdict(set)
    for (task, model), entry in entries.items():
        if entry.get("baselines"):
            availability[task].add((model, "baseline"))
        if any(x["tree_name"] == "hack_trajectories" for x in entry["trajectories"]):
            availability[task].add((model, "sanitized_hack"))
    required = {(m, c) for m in MODELS for c in CLASSES}
    tasks = sorted(task for task, present in availability.items() if required <= present)

    candidates = []
    source = run / "source" / "trajectories"
    source.mkdir(parents=True)
    for task in tasks:
        for model in MODELS:
            entry = entries[(task, model)]
            for item in entry["baselines"]:
                label = str(item["label"])
                rel = f"tasks/{task}/{model}/baseline_trajectories/{label}/trial/agent/trajectory.json"
                candidates.append((task, model, "baseline", label, rel))
            for item in entry["trajectories"]:
                if item["tree_name"] != "hack_trajectories":
                    continue
                label = item["trajectory_label"]
                rel = f"tasks/{task}/{model}/sanitized_trajectories/{label}/trial/agent/trajectory.json"
                candidates.append((task, model, "sanitized_hack", label, rel))

    rows = []
    for task, model, class_name, label, rel in candidates:
        eid = hashlib.sha256(rel.encode()).hexdigest()[:20]
        rows.append({"example_id":eid,"task_id":task,"source_model":model,"class":class_name,"trajectory_label":label,"label":int(class_name=="sanitized_hack"),"split":split_for(task),"path":str(source/f"{eid}.json"),"source_path":rel})

    balanced = []
    for model in MODELS:
        for split in ("train", "test"):
            subset = [r for r in rows if r["source_model"] == model and r["split"] == split]
            by_class = {c: sorted((r for r in subset if r["class"] == c), key=stable_rank) for c in CLASSES}
            n = min(map(len, by_class.values()))
            for c in CLASSES:
                balanced.extend(by_class[c][:n])
    rows = sorted(balanced, key=lambda r: (MODELS.index(r["source_model"]), r["split"], r["class"], stable_rank(r)))

    def download(row):
        target = Path(row["path"])
        for attempt in range(9):
            try:
                request = urllib.request.Request(base + "/" + row["source_path"], headers={"User-Agent":"behavior-tokens-research"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    target.write_bytes(response.read())
                return
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 8:
                    raise
                time.sleep(min(60, 2**attempt) + random.random())

    with ThreadPoolExecutor(max_workers=16) as pool:
        for completed, _ in enumerate(pool.map(download, rows), 1):
            if completed % 100 == 0 or completed == len(rows):
                elapsed = time.time() - started; rate = completed / max(elapsed, 1e-6)
                render_dashboard(run, {"phase":"downloading_dataset","completed":completed,"total":len(rows),"elapsed_seconds":elapsed,"throughput":rate,"eta_seconds":(len(rows)-completed)/rate,"run_id":a.run_id,"error_count":0,"retry_count":0})

    with (run / "manifest.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter((r["source_model"], r["split"], r["class"]) for r in rows)
    count_map = {"|".join(k): v for k, v in sorted(counts.items())}
    config = {"run_id":a.run_id,"trial_type":"new","dataset_revision":DATA_REV,"model":MODEL,"source_models":MODELS,"training_source":"claude-opus-4.6","classes":CLASSES,"task_selection":"194 tasks with both classes for all source models","task_count":len(tasks),"split":"sha256(seed|task_id) modulo 100; <70 train","seed":42,"max_length":16384,"workers":4,"layers":32,"hidden_size":4096,"probe":{"classifier":"StandardScaler + L2 LogisticRegression","C":1.0,"class_weight":"balanced","max_iter":5000},"counts":count_map,"examples":len(rows)}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_fingerprint"] = fingerprint
    atomic_json(run / "resolved_config.json", config)
    (run / "config.yaml").write_text(json.dumps(config, indent=2, sort_keys=True))
    atomic_json(run / "checkpoint.json", {"status":"running","phase":"prepared","next_step":"extract","config_fingerprint":fingerprint})
    atomic_json(run / "progress.json", {"phase":"prepared","completed":0,"total":len(rows),"run_id":a.run_id,"config_fingerprint":fingerprint,"class_counts":count_map,"error_count":0,"retry_count":0})
    render_dashboard(run, {"phase":"prepared","completed":0,"total":len(rows),"elapsed_seconds":time.time()-started,"eta_seconds":None,"run_id":a.run_id,"config_fingerprint":fingerprint,"class_counts":count_map,"error_count":0,"retry_count":0})
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
