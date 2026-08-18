from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "d8a29613235a0ef56a8b70b3142626a533da28c2"
BASE = f"https://raw.githubusercontent.com/few-sh/terminal-wrench/{REVISION}"
TASK_IDS = ["1056", "1063", "1144", "117", "1183"]
MODELS = ["claude-opus-4.6", "gemini-3.1-pro", "gpt-5.4"]


def fetch(path: str) -> str:
    request = urllib.request.Request(
        f"{BASE}/{path}", headers={"User-Agent": "behavior-tokens-research"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def main() -> None:
    index = json.loads(fetch("index/tasks.json"))
    entries = {(str(row["task_id"]), row["model"]): row for row in index}
    specs = []
    prompts = {}

    for task_id in TASK_IDS:
        prompt_path = f"tasks/{task_id}/{MODELS[0]}/original_task/instruction.md"
        prompts[task_id] = prompt_path
        for model in MODELS:
            entry = entries[(task_id, model)]
            baseline_label = str(entry["baselines"][0]["label"])
            hack_label = next(
                row["trajectory_label"]
                for row in entry["trajectories"]
                if row["tree_name"] == "hack_trajectories"
            )
            non_hack_label = next(
                row["trajectory_label"]
                for row in entry["trajectories"]
                if row["tree_name"] == "non_hack_trajectories"
            )
            labels = {
                "baseline": ("baseline_trajectories", baseline_label),
                "hacked": ("hack_trajectories", hack_label),
                "non_hacked": ("non_hack_trajectories", non_hack_label),
                "sanitized": ("sanitized_trajectories", hack_label),
                "stripped": ("stripped_trajectories", hack_label),
            }
            for style, (tree, label) in labels.items():
                path = f"tasks/{task_id}/{model}/{tree}/{label}/trial/agent/trajectory.json"
                specs.append((task_id, model, style, label, path))

    paths = list(prompts.values()) + [spec[4] for spec in specs]
    with ThreadPoolExecutor(max_workers=16) as pool:
        contents = dict(zip(paths, pool.map(fetch, paths)))

    tasks = []
    for task_id in TASK_IDS:
        task = {
            "task_id": task_id,
            "prompt": contents[prompts[task_id]],
            "models": {model: {} for model in MODELS},
        }
        for spec_task, model, style, label, path in specs:
            if spec_task == task_id:
                task["models"][model][style] = {
                    "label": label,
                    "source_path": path,
                    "trajectory": json.loads(contents[path]),
                }
        tasks.append(task)

    payload = {
        "dataset_revision": REVISION,
        "task_ids": TASK_IDS,
        "models": MODELS,
        "styles": ["baseline", "hacked", "non_hacked", "sanitized", "stripped"],
        "tasks": tasks,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output = Path(__file__).with_name("terminal_wrench_comparison_5_tasks.json")
    output.write_text(encoded)
    output.with_suffix(".js").write_text("window.COMPARISON_DATA=" + encoded + ";")
    print(f"wrote {output} ({output.stat().st_size:,} bytes, {len(specs)} traces)")


if __name__ == "__main__":
    main()
