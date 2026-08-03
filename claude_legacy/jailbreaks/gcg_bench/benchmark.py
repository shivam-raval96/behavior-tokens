"""Benchmark Zou GCG on Llama-3.2-1B across 3 jailbreak datasets → ASR + plot + json.

For each dataset, for each behavior:
  1. run GCG to find an adversarial suffix (affirmative-target objective),
  2. generate WITH the suffix (attacked) and WITHOUT (baseline),
  3. score success by refusal-string matching.

ASR_gcg vs ASR_baseline per dataset quantifies the attack's lift. Results written
to outputs/gcg_zou/{results.json, asr.png}.

Run:
  # quick local smoke (MPS/CPU)
  .venv/bin/python -m jailbreaks.gcg_bench.benchmark --quick
  # full run (prefer a GPU / Modal)
  .venv/bin/python -m jailbreaks.gcg_bench.benchmark \
      --datasets advbench jbb harmbench --n_behaviors 25 --steps 500
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .asr import asr, is_success
from .datasets_hf import DATASETS, load
from .gcg_zou import GCG, GCGConfig

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Canonical Llama-2 chat template — some mirrors (NousResearch) ship the tokenizer
# without one, and transformers >=4.44 no longer falls back to a default. Injected in
# load_model when tokenizer.chat_template is missing so prompt surgery works uniformly.
LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}{% else %}"
    "{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:                     # e.g. NousResearch Llama-2 mirror
        tok.chat_template = LLAMA2_CHAT_TEMPLATE
        print(f"[setup] no chat_template on {model_name}; injected Llama-2 default")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=_DTYPES[dtype]).to(device)
    model.eval()
    return model, tok


def run(args):
    device = resolve_device(args.device)
    print(f"[setup] model={args.model} device={device} dtype={args.dtype}")
    model, tok = load_model(args.model, device, args.dtype)
    gcfg = GCGConfig(suffix_len=args.suffix_len, steps=args.steps, topk=args.topk,
                     search_batch=args.search_batch, eval_chunk=args.eval_chunk,
                     seed=args.seed)
    attack = GCG(model, tok, gcfg)

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "results.json")
    plot_path = os.path.join(args.out, "asr.png")
    summary, per_behavior = {}, {}
    t0 = time.time()

    def checkpoint():
        """Flush partial results after each dataset so a crash leaves usable data."""
        payload = {
            "config": {
                "model": args.model, "device": device, "dtype": args.dtype,
                "metric": "refusal-string-match ASR (Zou et al. 2023)",
                "gcg": vars(gcfg), "n_behaviors": args.n_behaviors,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            },
            "runtime_sec": round(time.time() - t0, 1),
            "datasets_done": list(summary),
            "summary": summary,
            "per_behavior": per_behavior,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        make_plot(summary, plot_path, args.model)
        return payload

    for ds_name in args.datasets:
        recs = load(ds_name, n=args.n_behaviors, seed=args.seed)
        print(f"\n=== {ds_name}: {len(recs)} behaviors ===", flush=True)
        rows, atk_gens, base_gens = [], [], []
        for i, r in enumerate(recs):
            print(f"  [{ds_name} {i+1}/{len(recs)}] {r['goal'][:60]}", flush=True)
            res = attack.optimize(r["goal"], r["target"], log=print)
            atk = attack.generate(r["goal"], res.suffix_ids, args.max_new_tokens)
            base = attack.generate(r["goal"], None, args.max_new_tokens)
            atk_gens.append(atk)
            base_gens.append(base)
            rows.append({
                "goal": r["goal"], "target": r["target"], "suffix": res.suffix,
                "final_loss": res.loss,
                "baseline_gen": base[:400], "attacked_gen": atk[:400],
                "baseline_success": is_success(base), "attacked_success": is_success(atk),
            })
            per_behavior[ds_name] = rows          # partial rows visible mid-dataset
            summary[ds_name] = {
                "n": i + 1,
                "asr_baseline": round(asr(base_gens), 4),
                "asr_gcg": round(asr(atk_gens), 4),
            }
            if (i + 1) % 5 == 0:                  # periodic flush within a dataset
                checkpoint()
                print(f"    [checkpoint] {ds_name} {i+1}/{len(recs)}  "
                      f"ASR_gcg={summary[ds_name]['asr_gcg']}", flush=True)
        checkpoint()
        print(f"  -> ASR baseline={summary[ds_name]['asr_baseline']} "
              f"GCG={summary[ds_name]['asr_gcg']}", flush=True)

    payload = checkpoint()
    print(f"\n[write] {json_path}")
    print(f"[write] {plot_path}")
    print(f"[done] {payload['runtime_sec']}s  " +
          "  ".join(f"{k}:{v['asr_gcg']}" for k, v in summary.items()))
    return payload


def make_plot(summary: dict, path: str, model: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(summary)
    base = [summary[n]["asr_baseline"] for n in names]
    gcg = [summary[n]["asr_gcg"] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 4.2))
    b1 = ax.bar(x - w / 2, base, w, label="baseline (no suffix)", color="#5b8a72")
    b2 = ax.bar(x + w / 2, gcg, w, label="GCG suffix", color="#d1495b")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Zou GCG ASR — {model.split('/')[-1]}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_argparser():
    p = argparse.ArgumentParser(description="Zou GCG jailbreak benchmark")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--n_behaviors", type=int, default=25)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--suffix_len", type=int, default=20)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--search_batch", type=int, default=512)
    p.add_argument("--eval_chunk", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="jailbreaks/outputs/gcg_zou")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke: advbench only, 2 behaviors, 25 steps, small search")
    return p


def apply_quick(args):
    args.datasets = ["advbench"]
    args.n_behaviors = 2
    args.steps = 25
    args.suffix_len = 8
    args.search_batch = 64
    args.eval_chunk = 32
    args.max_new_tokens = 48
    return args


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.quick:
        args = apply_quick(args)
    run(args)
