"""Load three HuggingFace jailbreak datasets → uniform [{goal, target}] records.

Each dataset supplies harmful *requests* (goals). GCG also needs an affirmative
*target* string; datasets that ship one (AdvBench, JBB) use it, others synthesize
the standard AdvBench-style target: "Sure, here is <goal>".

Datasets (all public, ungated research red-team benchmarks):
  advbench  mlabonne/harmful_behaviors        — AdvBench (Zou et al.) mirror; goal only
  jbb       JailbreakBench/JBB-Behaviors      — 100 curated behaviors (Goal, Target)
  forbidden TrustAIRLab/forbidden_question_set — 390 forbidden questions across 13 policies

walledai/{AdvBench,HarmBench} are gated on the Hub; the AdvBench goals are identical
in the mlabonne mirror. Pass a HF token + swap `path` if you have gated access.
"""
from __future__ import annotations

# key candidates for auto-detecting the goal / target column across schemas
_GOAL_KEYS = ["goal", "Goal", "prompt", "Prompt", "behavior", "Behavior",
              "query", "question", "text"]
_TARGET_KEYS = ["target", "Target"]

DATASETS = {
    "advbench":  dict(path="mlabonne/harmful_behaviors",        split="train"),
    "jbb":       dict(path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful"),
    "forbidden": dict(path="TrustAIRLab/forbidden_question_set", split="train"),
}


def _pick(row: dict, keys: list[str]):
    for k in keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None


def _synth_target(goal: str) -> str:
    g = goal.rstrip(".?! ")
    return "Sure, here is " + g[0].lower() + g[1:] if g else "Sure, here is the information"


def load(name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    ds = load_dataset(spec["path"], spec.get("name"), split=spec["split"])
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))

    out = []
    for row in ds:
        goal = _pick(row, _GOAL_KEYS)
        if not goal:
            continue
        target = _pick(row, _TARGET_KEYS) or _synth_target(goal)
        out.append({"goal": goal, "target": target})
    return out


if __name__ == "__main__":
    for nm in DATASETS:
        recs = load(nm, n=3)
        print(f"\n=== {nm} ({len(recs)} shown) ===")
        for r in recs:
            print(f"  goal:   {r['goal'][:70]}")
            print(f"  target: {r['target'][:70]}")
