"""One-step token-boundary audit for the Llama-3 text-control GCG path."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from jailbreaks.gcg_small_scale import load_model, run_paths
from jailbreaks.gcg_text_reference import ReferenceTextGCG, disallowed_control_tokens


def run(config_path: Path, output_base: Path, commit=None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("stage") != "llama32_gcg_text_boundary_probe":
        raise ValueError("invalid text-boundary probe config")
    paths = run_paths(cfg, output_base)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["directory"].joinpath("config.yaml").write_text(config_path.read_text())
    with Path(cfg["dataset_csv"]).open(newline="") as handle:
        row = list(csv.DictReader(handle))[cfg["diagnostic_behavior_index"]]
    goal = row["goal"].strip()
    target = cfg["target_prefix"] + goal[:1].lower() + goal[1:]
    model, tokenizer, device = load_model(cfg)
    attack = ReferenceTextGCG(model, tokenizer, cfg["suffix_length"], cfg["top_k"],
                              cfg["candidate_batch_size"], cfg["candidate_batch_size"])
    control = attack.initial_control()
    gradient, loss, layout = attack.token_gradient(goal, control, target)
    _, control_start, control_stop, _, _ = attack.manager._render(goal, control, target)
    standalone = attack.manager.control_ids(control)
    slice_ids = layout.input_ids[layout.control_slice]
    blocked = disallowed_control_tokens(tokenizer, cfg["allow_non_ascii"]).to(device)
    ranked = gradient.clone(); ranked[:, blocked] = float("inf")
    top_ids = (-ranked).topk(cfg["top_k"], dim=1).indices
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    positions = torch.arange(0, cfg["suffix_length"], cfg["suffix_length"] / cfg["candidate_batch_size"], device=device).long()
    choices = torch.randint(0, cfg["top_k"], (cfg["candidate_batch_size"],), device=device, generator=generator)
    raw = slice_ids.unsqueeze(0).repeat(cfg["candidate_batch_size"], 1)
    raw[torch.arange(cfg["candidate_batch_size"], device=device), positions] = top_ids[positions, choices]
    lengths, examples, valid = Counter(), [], []
    for candidate in raw:
        text = tokenizer.decode(candidate.tolist(), skip_special_tokens=True)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        lengths[len(ids)] += 1
        if len(examples) < 12:
            examples.append({"decoded": text, "raw_ids": candidate.tolist(), "retokenized_ids": ids,
                             "retokenized_length": len(ids)})
        if len(ids) == cfg["suffix_length"] and text != control:
            valid.append(text)
    result = {
        "status": "complete", "objective_loss": loss,
        "control": {"text": control, "character_start": control_start, "character_stop": control_stop,
                    "standalone_ids": standalone.tolist(), "full_prompt_slice_ids": slice_ids.tolist(),
                    "same_ids": standalone.tolist() == slice_ids.tolist()},
        "prompt": layout.prompt, "control_slice": [layout.control_slice.start, layout.control_slice.stop],
        "candidate_length_histogram": dict(sorted(lengths.items())), "valid_candidate_count": len(valid),
        "candidate_examples": examples,
    }
    paths["checkpoint"].write_text(json.dumps(result, indent=2)); paths["progress"].write_text(json.dumps(result, indent=2)); paths["result"].write_text(json.dumps(result, indent=2))
    paths["summary"].write_text(f"# Llama-3.2 GCG text-boundary probe\n\n- Full/suffix IDs equal: {result['control']['same_ids']}\n- Valid candidates: {len(valid)}/{cfg['candidate_batch_size']}\n- Candidate lengths: {result['candidate_length_histogram']}\n")
    if commit:
        commit()
    return result
