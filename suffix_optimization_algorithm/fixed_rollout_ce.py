from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config: dict[str, Any]) -> str:
    scientific = {k: v for k, v in config.items() if k != "run_mode"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def splice_suffix(prompt: str, suffix: str) -> str:
    return prompt if not suffix else f"{prompt}\n\n{suffix}"


def chat_prefix_ids(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return list(ids)


def continuation_nll(
    logits: torch.Tensor, full_ids: list[int], prefix_length: int
) -> tuple[torch.Tensor, int]:
    if prefix_length < 1 or prefix_length >= len(full_ids):
        raise ValueError("input must contain a nonempty prefix and continuation")
    if logits.shape[1] != len(full_ids):
        raise ValueError("logit sequence length does not match token ids")
    labels = torch.tensor(full_ids, device=logits.device, dtype=torch.long)
    # Label y[0] (at full_ids[prefix_length]) is predicted by the final prefix
    # logit (at prefix_length - 1). No prompt or suffix label enters this slice.
    target_logits = logits[0, prefix_length - 1 : -1].float()
    target_labels = labels[prefix_length:]
    losses = torch.nn.functional.cross_entropy(
        target_logits, target_labels, reduction="none"
    )
    return losses.sum(), int(losses.numel())


def transformer_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected model.model.layers")
    return layers


@contextmanager
def steering(model, module_index: int, vector: torch.Tensor, coefficient: float):
    addition = coefficient * vector

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden + addition.to(hidden.device, hidden.dtype).view(1, 1, -1)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = transformer_layers(model)[module_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tokens = sum(int(row["token_count"]) for row in rows)
    if not rows or not tokens:
        raise ValueError("cannot aggregate empty score records")
    return {
        "records": len(rows),
        "tokens": tokens,
        "nll_sum": float(sum(float(row["nll_sum"]) for row in rows)),
        "token_weighted_ce": float(
            sum(float(row["nll_sum"]) for row in rows) / tokens
        ),
        "record_mean_ce": float(
            np.mean(
                [float(row["nll_sum"]) / int(row["token_count"]) for row in rows]
            )
        ),
    }


class RunWriter:
    def __init__(self, output: Path, commit=None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.commit = commit or (lambda: None)

    def json(self, name: str, payload: Any) -> None:
        atomic_json(self.output / name, payload)
        self.commit()

    def progress(self, payload: dict[str, Any]) -> None:
        self.json("progress.json", payload)
        append_jsonl(self.output / "dashboard_history.jsonl", payload)
        self.dashboard(payload)
        self.commit()

    def dashboard(self, latest: dict[str, Any]) -> None:
        history = jsonl_rows(self.output / "dashboard_history.jsonl")
        embedded = json.dumps(history, separators=(",", ":")).replace("</", "<\\/")
        pretty = json.dumps(latest, indent=2, sort_keys=True)
        html = f"""<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5>
<title>{self.output.name}</title><style>body{{font:14px system-ui;background:#0b1020;color:#edf2ff;max-width:1000px;margin:30px auto}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}section{{background:#151c31;border:1px solid #2a3553;border-radius:10px;padding:16px;margin:12px 0}}b{{font-size:22px}}pre{{white-space:pre-wrap;color:#b8c3dd}}</style>
<h1>{self.output.name}</h1><div class=grid id=g></div><section><canvas id=c width=960 height=280></canvas></section><section><pre>{pretty}</pre></section>
<script>const H={embedded},L=H.at(-1)||{{}},G=[['Phase',L.phase],['Progress',(L.completed||0)+' / '+(L.total||0)],['Current CE',L.latest_ce],['Best gap',L.best_gap]];g.innerHTML=G.map(x=>`<section>${{x[0]}}<br><b>${{x[1]??'—'}}</b></section>`).join('');const x=c.getContext('2d'),W=c.width,Ht=c.height;x.strokeStyle='#2a3553';for(let i=0;i<5;i++){{x.beginPath();x.moveTo(45,20+i*55);x.lineTo(940,20+i*55);x.stroke()}}const vals=H.map(r=>r.completed_fraction||0);x.strokeStyle='#67e8f9';x.lineWidth=3;x.beginPath();vals.forEach((v,i)=>{{const X=45+i/Math.max(1,vals.length-1)*895,Y=240-v*220;i?x.lineTo(X,Y):x.moveTo(X,Y)}});x.stroke()</script>"""
        (self.output / "dashboard.html").write_text(html)


def load_selection(dataset: Path, rows: list[int]) -> list[dict[str, Any]]:
    with dataset.open(newline="") as stream:
        source = list(csv.DictReader(stream))
    return [
        {
            "dataset_row": index,
            "behavior_id": hashlib.sha256(source[index]["goal"].encode()).hexdigest()[
                :12
            ],
            "prompt": source[index]["goal"],
        }
        for index in rows
    ]


@torch.inference_mode()
def score_record(model, prefix: list[int], continuation: list[int], hook=None):
    full = prefix + continuation
    ids = torch.tensor([full], device=model.device)
    context = hook if hook is not None else _null_context()
    with context:
        logits = model(input_ids=ids, use_cache=False).logits
    nll, count = continuation_nll(logits, full, len(prefix))
    return float(nll), count


@contextmanager
def _null_context():
    yield


def run(config_path: Path, output: Path, mode: str = "fresh", commit=None):
    config = yaml.safe_load(config_path.read_text())
    config["run_mode"] = mode
    config_hash = fingerprint(config)
    writer = RunWriter(output, commit)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    if mode == "fresh" and checkpoint:
        raise FileExistsError("fresh run refuses an existing checkpoint")
    if mode == "resume" and checkpoint.get("config_fingerprint") != config_hash:
        raise ValueError("resume config fingerprint mismatch")

    dataset, vector_path = Path(config["dataset_csv"]), Path(config["vector_path"])
    if sha256(dataset) != config["dataset_sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    if sha256(vector_path) != config["vector_sha256"]:
        raise ValueError("vector SHA-256 mismatch")
    selection = load_selection(dataset, config["dataset_rows"])
    writer.json("resolved_config.json", config)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    writer.json("selection.json", {"dataset_sha256": sha256(dataset), "rows": selection})
    writer.json(
        "source_artifacts.json",
        {
            "config_fingerprint": config_hash,
            "vector_sha256": sha256(vector_path),
            "vector_metadata": json.loads(Path(config["vector_metadata_path"]).read_text()),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    vector = torch.tensor(np.load(vector_path), device=model.device, dtype=torch.float32)
    stopped = False

    def terminate(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    cache_path = output / "teacher_rollouts.jsonl"
    cached = jsonl_rows(cache_path)
    total_rollouts = len(selection) * config["rollouts_per_prompt"]
    started = time.monotonic()
    for prompt_index, item in enumerate(tqdm(selection, desc="teacher_cache")):
        for rollout_index, seed in enumerate(config["rollout_seeds"]):
            record_index = prompt_index * config["rollouts_per_prompt"] + rollout_index
            if record_index < len(cached):
                continue
            prefix = chat_prefix_ids(tokenizer, item["prompt"])
            generator = torch.Generator(device=model.device).manual_seed(seed)
            with steering(model, config["module_index"], vector, config["coefficient"]):
                generated = model.generate(
                    input_ids=torch.tensor([prefix], device=model.device),
                    do_sample=True,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    max_new_tokens=config["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generator=generator,
                    use_cache=True,
                )[0, len(prefix):].tolist()
            row = {
                **item,
                "rollout_index": rollout_index,
                "seed": seed,
                "prefix_ids": prefix,
                "continuation_ids": generated,
                "continuation_text": tokenizer.decode(generated, skip_special_tokens=True),
                "hit_eos": bool(generated and generated[-1] == tokenizer.eos_token_id),
            }
            append_jsonl(cache_path, row)
            cached.append(row)
        progress = {
            "run_id": output.name, "config_fingerprint": config_hash, "phase": "cache_teacher",
            "completed": len(cached), "total": total_rollouts, "completed_fraction": len(cached) / total_rollouts,
            "elapsed_seconds": time.monotonic() - started, "latest_ce": None, "best_gap": None,
            "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0)),
        }
        writer.progress(progress)
        writer.json("checkpoint.json", {"status": "stopped" if stopped else "running", "phase": "cache_teacher", "cached_records": len(cached), "config_fingerprint": config_hash, "retry_count": progress["retry_count"]})
        if stopped:
            return {"status": "stopped", "phase": "cache_teacher", "completed": len(cached)}

    cache_hash = sha256(cache_path)
    writer.json("teacher_cache_manifest.json", {"records": len(cached), "sha256": cache_hash, "immutable": True})
    floor_path = output / "floor_scores.jsonl"
    floors = jsonl_rows(floor_path)
    for index, record in enumerate(tqdm(cached[len(floors):], initial=len(floors), total=len(cached), desc="floor"), start=len(floors)):
        nll, count = score_record(model, record["prefix_ids"], record["continuation_ids"], steering(model, config["module_index"], vector, config["coefficient"]))
        row = {"record_index": index, "behavior_id": record["behavior_id"], "rollout_index": record["rollout_index"], "nll_sum": nll, "token_count": count}
        append_jsonl(floor_path, row); floors.append(row)
        if len(floors) % config["score_checkpoint_every_records"] == 0 or len(floors) == len(cached):
            current = aggregate(floors)
            writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "score_floor", "completed": len(floors), "total": len(cached), "completed_fraction": len(floors)/len(cached), "elapsed_seconds": time.monotonic()-started, "latest_ce": current["token_weighted_ce"], "best_gap": None, "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
            writer.json("checkpoint.json", {"status": "running", "phase": "score_floor", "floor_records": len(floors), "config_fingerprint": config_hash})
        if stopped: return {"status": "stopped", "phase": "score_floor", "completed": len(floors)}
    floor = aggregate(floors)

    suffix_results = []
    for suffix in config["suffixes"]:
        score_path = output / f"suffix_scores_{suffix['name']}.jsonl"
        scores = jsonl_rows(score_path)
        for index, record in enumerate(tqdm(cached[len(scores):], initial=len(scores), total=len(cached), desc=f"suffix_{suffix['name']}"), start=len(scores)):
            prompt = next(row["prompt"] for row in selection if row["behavior_id"] == record["behavior_id"])
            prefix = chat_prefix_ids(tokenizer, splice_suffix(prompt, suffix["text"]))
            nll, count = score_record(model, prefix, record["continuation_ids"])
            row = {"record_index": index, "behavior_id": record["behavior_id"], "rollout_index": record["rollout_index"], "nll_sum": nll, "token_count": count}
            append_jsonl(score_path, row); scores.append(row)
            if len(scores) % config["score_checkpoint_every_records"] == 0 or len(scores) == len(cached):
                current = aggregate(scores); gap = current["token_weighted_ce"] - floor["token_weighted_ce"]
                writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": f"score_suffix_{suffix['name']}", "completed": len(scores), "total": len(cached), "completed_fraction": len(scores)/len(cached), "elapsed_seconds": time.monotonic()-started, "latest_ce": current["token_weighted_ce"], "best_gap": gap, "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
                writer.json("checkpoint.json", {"status": "running", "phase": f"score_suffix_{suffix['name']}", "suffix_records": len(scores), "config_fingerprint": config_hash})
            if stopped: return {"status": "stopped", "phase": f"score_suffix_{suffix['name']}", "completed": len(scores)}
        summary = aggregate(scores)
        suffix_results.append({"name": suffix["name"], "text": suffix["text"], **summary, "student_ce_minus_floor": summary["token_weighted_ce"] - floor["token_weighted_ce"]})

    if sha256(cache_path) != cache_hash:
        raise RuntimeError("teacher cache mutated during scoring")
    results = {"run_id": output.name, "config_fingerprint": config_hash, "teacher_cache_sha256": cache_hash, "floor": floor, "suffixes": suffix_results}
    writer.json("results.json", results)
    lines = ["# Fixed-rollout continuation CE results", "", f"- Teacher cache: `{cache_hash}` ({len(cached)} records)", f"- Steered floor CE: **{floor['token_weighted_ce']:.6f} nats/token**"]
    for row in suffix_results:
        lines += [f"- Student `{row['name']}` CE: **{row['token_weighted_ce']:.6f} nats/token**", f"- `student_CE - floor`: **{row['student_ce_minus_floor']:.6f} nats/token**"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    writer.json("checkpoint.json", {"status": "complete", "phase": "complete", "config_fingerprint": config_hash, "teacher_cache_sha256": cache_hash})
    writer.progress({"run_id": output.name, "config_fingerprint": config_hash, "phase": "complete", "completed": len(cached), "total": len(cached), "completed_fraction": 1.0, "elapsed_seconds": time.monotonic()-started, "latest_ce": suffix_results[-1]["token_weighted_ce"], "best_gap": min(row["student_ce_minus_floor"] for row in suffix_results), "error_count": 0, "retry_count": int(checkpoint.get("retry_count", 0))})
    return results
