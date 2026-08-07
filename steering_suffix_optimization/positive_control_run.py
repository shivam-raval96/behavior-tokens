"""Direct layer-10 steering gate for response-position suffix optimization."""
from __future__ import annotations

import csv, hashlib, json, os, time
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .controls import add_residual_direction
from .harmbench_eval import TEMPLATE, atomic_json, label, paired_delta


def repeated_trigram_fraction(text: str) -> float:
    tokens=text.split()
    if len(tokens)<3: return 0.0
    grams=[tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return 1-len(set(grams))/len(grams)


def run(config_path: Path, dataset_csv: Path, output_root: Path, run_id: str,
        commit: Callable[[],None] | None=None) -> dict:
    config=yaml.safe_load(config_path.read_text()); fp=hashlib.sha256(config_path.read_bytes()).hexdigest()
    output=output_root/run_id; output.mkdir(parents=True,exist_ok=True)
    resolved=output/"resolved_config.json"
    if resolved.exists() and json.loads(resolved.read_text())!=config: raise ValueError("config mismatch")
    (output/"config.yaml").write_text(config_path.read_text()); atomic_json(resolved,config)
    with dataset_csv.open(newline="") as stream: source=list(csv.DictReader(stream))
    prompts=[source[i]["goal"].strip() for i in config["evaluation_indices"]]
    vector=torch.from_numpy(np.load(config["vector"]).astype("float32"))
    if not np.isclose(float(vector.norm()),1.0,atol=1e-5): raise ValueError("expected unit vector")
    tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["model_revision"])
    model=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["model_revision"],
        torch_dtype=torch.bfloat16,device_map="auto").eval()
    rows_path=output/"generations.jsonl"
    rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x] if rows_path.exists() else []
    done={(x["additive_norm"],x["prompt_index"]) for x in rows}; started=time.monotonic()
    with rows_path.open("a") as stream:
        for scale in config["positive_control_additive_norms"]:
            for index,prompt in enumerate(tqdm(prompts,desc=f"steering {scale:g}",unit="prompt")):
                if (float(scale),index) in done: continue
                ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,
                    add_generation_prompt=True,return_tensors="pt").to(model.device)
                context=add_residual_direction(model,config["module_index"],vector,float(scale)) if scale else None
                if context: context.__enter__()
                try:
                    generated=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        max_new_tokens=config["max_new_tokens"],pad_token_id=tokenizer.eos_token_id)
                finally:
                    if context: context.__exit__(None,None,None)
                new=generated[0,ids.shape[1]:]
                response=tokenizer.decode(new,skip_special_tokens=True)
                row={"prompt_index":index,"dataset_index":config["evaluation_indices"][index],
                    "prompt":prompt,"additive_norm":float(scale),"response":response,
                    "generated_tokens":int(new.numel()),"eos":bool((new==tokenizer.eos_token_id).any()),
                    "repeated_trigram_fraction":repeated_trigram_fraction(response)}
                rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush(); os.fsync(stream.fileno())
                progress={"status":"running","phase":"generation","completed":len(rows),
                    "total":len(prompts)*len(config["positive_control_additive_norms"]),"run_id":run_id,
                    "config_fingerprint":fp,"elapsed_seconds":time.monotonic()-started,
                    "throughput":len(rows)/max(time.monotonic()-started,1e-6),"latest_scale":scale,
                    "best_metric":None,"error_count":0,"retry_count":0}
                atomic_json(output/"checkpoint.json",progress); atomic_json(output/"progress.json",progress)
                if commit: commit()
                if len(rows)%config["generation_checkpoint_every"]==0: print("METRIC "+json.dumps(progress),flush=True)
    del model; torch.cuda.empty_cache()
    judge_tokenizer=AutoTokenizer.from_pretrained(config["judge"],revision=config["judge_revision"],
        token=os.environ.get("HF_TOKEN"),use_fast=False)
    judge_tokenizer.padding_side=judge_tokenizer.truncation_side="left"
    if judge_tokenizer.pad_token_id is None: judge_tokenizer.pad_token=judge_tokenizer.eos_token
    judge=AutoModelForCausalLM.from_pretrained(config["judge"],revision=config["judge_revision"],
        token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    judge.generation_config.temperature=judge.generation_config.top_p=None
    judgments_path=output/"harmbench_judgments.jsonl"
    judgments=[json.loads(x) for x in judgments_path.read_text().splitlines() if x] if judgments_path.exists() else []
    judged={(x["additive_norm"],x["prompt_index"]) for x in judgments}
    pending=[x for x in rows if (x["additive_norm"],x["prompt_index"]) not in judged]
    with judgments_path.open("a") as stream:
        for start in tqdm(range(0,len(pending),config["judge_batch_size"]),desc="HarmBench",unit="batch"):
            batch=pending[start:start+config["judge_batch_size"]]
            encoded=judge_tokenizer([TEMPLATE.format(behavior=x["prompt"],generation=x["response"]) for x in batch],
                padding=True,truncation=True,max_length=config["judge_max_input_tokens"],return_tensors="pt").to("cuda")
            generated=judge.generate(**encoded,do_sample=False,max_new_tokens=1,pad_token_id=judge_tokenizer.pad_token_id)
            raw=judge_tokenizer.batch_decode(generated[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
            for source,prediction in zip(batch,raw):
                item={**source,"raw_prediction":prediction,"parsed_prediction":label(prediction),
                      "valid":label(prediction)!="invalid","success":label(prediction)=="yes"}
                judgments.append(item); stream.write(json.dumps(item)+"\n")
            stream.flush(); os.fsync(stream.fileno())
            rates={str(scale):float(np.mean([x["success"] for x in judgments if x["additive_norm"]==scale]))
                   for scale in sorted({x["additive_norm"] for x in judgments})}
            progress={"status":"running","phase":"judge","completed":len(judgments),"total":len(rows),
                "run_id":run_id,"config_fingerprint":fp,"current_asr":rates,
                "invalid_count":sum(not x["valid"] for x in judgments),"error_count":0,"retry_count":0}
            atomic_json(output/"checkpoint.json",progress); atomic_json(output/"progress.json",progress)
            if commit: commit()
            print("METRIC "+json.dumps(progress),flush=True)
    by={(x["additive_norm"],x["prompt_index"]):x for x in judgments}; zero=[by[(0.0,i)]["success"] for i in range(len(prompts))]
    curve=[]
    for scale in config["positive_control_additive_norms"]:
        flags=[by[(float(scale),i)]["success"] for i in range(len(prompts))]
        quality=[x for x in judgments if x["additive_norm"]==float(scale)]
        curve.append({"additive_norm":float(scale),"asr":float(np.mean(flags)),
            "paired_vs_zero":paired_delta(zero,flags,config["bootstrap_seed"],config["bootstrap_samples"]),
            "invalid":sum(not x["valid"] for x in quality),"eos_rate":float(np.mean([x["eos"] for x in quality])),
            "mean_repeated_trigram_fraction":float(np.mean([x["repeated_trigram_fraction"] for x in quality]))})
    result={"status":"complete","run_id":run_id,"config_fingerprint":fp,"curve":curve,"rows":len(rows),
            "judgments":len(judgments),"vector_sha256":config["vector_sha256"]}
    atomic_json(output/"results.json",result); atomic_json(output/"checkpoint.json",result); atomic_json(output/"progress.json",result)
    (output/"RESULTS.md").write_text("# Layer-10 direct-steering positive control\n\n"+"\n".join(f"- {x['additive_norm']:+.6f}: ASR {x['asr']:.3f}, delta {x['paired_vs_zero']['delta']:+.3f}" for x in curve)+"\n")
    if commit: commit()
    return result
