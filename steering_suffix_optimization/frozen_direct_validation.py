from __future__ import annotations

import csv, hashlib, json, signal, time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import beta
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import file_sha256
from .direct_steering_calibration import generate, judge, repeated_trigram_fraction
from .io_utils import ArtifactWriter
from .layout import build_layout
from .live_dashboard import update_dashboard


def fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k:v for k,v in config.items() if k != "run_mode"}, sort_keys=True).encode()).hexdigest()


def exact_ci(successes: int, n: int, alpha=.05):
    low = 0.0 if successes == 0 else float(beta.ppf(alpha/2, successes, n-successes+1))
    high = 1.0 if successes == n else float(beta.ppf(1-alpha/2, successes+1, n-successes))
    return [low, high]


def summarize(rows, bootstrap_seed=42, bootstrap_samples=1000):
    judged=[r for r in rows if len(r.get("judgments",{}))==2]
    result={"n":len(judged)}
    if not judged: return result
    b=np.array([int(r["judgments"]["baseline"]["success"]) for r in judged]); d=np.array([int(r["judgments"]["direct"]["success"]) for r in judged]); delta=d-b
    rng=np.random.default_rng(bootstrap_seed); boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(bootstrap_samples)])
    result.update(baseline_successes=int(b.sum()),baseline_asr=float(b.mean()),baseline_asr_95ci=exact_ci(int(b.sum()),len(b)),
        direct_successes=int(d.sum()),direct_asr=float(d.mean()),direct_asr_95ci=exact_ci(int(d.sum()),len(d)),paired_delta=float(delta.mean()),
        paired_delta_95ci=[float(x) for x in np.quantile(boot,[.025,.975])],failure_to_success=int(((b==0)&(d==1)).sum()),success_to_failure=int(((b==1)&(d==0)).sum()),
        incoherent_rate=float(np.mean([not r["judgments"]["direct"]["coherent"] for r in judged])),
        mean_repeated_trigram_fraction=float(np.mean([r["quality"]["direct"]["repeated_trigram_fraction"] for r in judged])),
        eos_rate=float(np.mean([r["quality"]["direct"]["hit_eos"] for r in judged])))
    return result


def run(config_path: Path, output: Path, mode="fresh", commit=None):
    config=yaml.safe_load(config_path.read_text()); config["run_mode"]=mode; config_hash=fingerprint(config); writer=ArtifactWriter(output,commit)
    cp=output/"checkpoint.json"
    if mode=="fresh" and cp.exists(): raise FileExistsError("fresh run refuses existing checkpoint")
    checkpoint=json.loads(cp.read_text()) if cp.exists() else {}
    if mode=="resume" and checkpoint.get("config_fingerprint")!=config_hash: raise ValueError("resume fingerprint mismatch")
    dataset,vector_path=Path(config["dataset_csv"]),Path(config["vector_path"])
    if file_sha256(vector_path)!=config["vector_sha256"]: raise ValueError("vector checksum mismatch")
    with dataset.open(newline="") as stream: source=list(csv.DictReader(stream))
    selection=[{"dataset_row":i,"behavior_id":hashlib.sha256(source[i]["goal"].encode()).hexdigest()[:12],"prompt":source[i]["goal"]} for i in config["evaluation_rows"]]
    output.mkdir(parents=True,exist_ok=True); (output/"config.yaml").write_text(config_path.read_text()); config.update(run_id=output.name,config_fingerprint=config_hash)
    writer.json("resolved_config.json",config); writer.json("selection.json",{"dataset_sha256":file_sha256(dataset),"rows":selection}); writer.json("source_artifacts.json",{"vector_sha256":config["vector_sha256"],"calibration_run_id":config["source_calibration_run_id"],"calibration_fingerprint":config["source_calibration_fingerprint"],"implementation_commit":config["implementation_commit"]})
    update_dashboard(output,{"run_id":output.name,"config_fingerprint":config_hash,"phase":"initializing","completed":0,"total":25,"completed_fraction":0.0,"elapsed_seconds":0.0,"throughput_per_second":0.0,"eta_seconds":None,"error_count":0,"retry_count":0})
    writer.commit()
    tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["model_revision"]); tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["model_revision"],torch_dtype=torch.bfloat16,device_map="auto").eval()
    unit=torch.tensor(np.load(vector_path),device=model.device,dtype=torch.float32); unit/=unit.norm()
    rows=list(checkpoint.get("rows",[])); phase=checkpoint.get("phase","generate"); errors=int(checkpoint.get("error_count",0)); retries=int(checkpoint.get("retry_count",0)); stopped=False; started=time.monotonic()
    def terminate(_sig,_frame):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGTERM,terminate)
    def save(active):
        elapsed=time.monotonic()-started; completed=len(rows) if active=="generate" else sum(len(r.get("judgments",{})) for r in rows); total=len(selection) if active=="generate" else 2*len(selection)
        partial=summarize(rows,config["bootstrap_seed"],min(1000,config["bootstrap_samples"])); rate=completed/max(elapsed,1e-9); snap={"run_id":output.name,"config_fingerprint":config_hash,"phase":active,"completed":completed,"total":total,"completed_fraction":completed/total,"elapsed_seconds":elapsed,"throughput_per_second":rate,"eta_seconds":(total-completed)/rate if rate else None,"error_count":errors,"retry_count":retries,**partial}
        writer.json("progress.json",snap); writer.json("checkpoint.json",{"config_fingerprint":config_hash,"phase":active,"rows":rows,"error_count":errors,"retry_count":retries}); update_dashboard(output,snap); writer.commit(); print(json.dumps(snap),flush=True)
    save(phase)
    if phase=="generate":
        for item in tqdm(selection[len(rows):],initial=len(rows),total=len(selection),desc="paired_generation"):
            layout=build_layout(tokenizer,item["prompt"],1); baseline,b_eos=generate(model,tokenizer,layout,config["max_new_tokens"]); direct,d_eos=generate(model,tokenizer,layout,config["max_new_tokens"],config["module_index"],unit,config["additive_norm"])
            row={**item,"responses":{"baseline":baseline,"direct":direct},"quality":{"baseline":{"hit_eos":b_eos,"repeated_trigram_fraction":repeated_trigram_fraction(baseline)},"direct":{"hit_eos":d_eos,"repeated_trigram_fraction":repeated_trigram_fraction(direct)}},"judgments":{}}
            rows.append(row); writer.jsonl("paired_generations.jsonl",row); save("generate")
            if stopped:return {"status":"stopped","phase":"generate"}
        phase="judge"; save(phase)
    pending=[(r,c) for r in rows for c in ("baseline","direct") if c not in r["judgments"]]
    for i,(row,condition) in enumerate(tqdm(pending,desc="openai_judge"),1):
        try: row["judgments"][condition]=judge(row["prompt"],row["responses"][condition],config["judge"])
        except Exception:
            errors+=1; save("judge"); raise
        writer.jsonl("openai_judgments.jsonl",{"behavior_id":row["behavior_id"],"condition":condition,**row["judgments"][condition]})
        if i%config["judge"]["checkpoint_every"]==0 or i==len(pending):save("judge")
        if stopped:save("judge");return {"status":"stopped","phase":"judge"}
    summary=summarize(rows,config["bootstrap_seed"],config["bootstrap_samples"]); gate=config["validation_gate"]
    passed=summary["direct_successes"]>=gate["min_direct_successes"] and summary["failure_to_success"]-summary["success_to_failure"]>=gate["min_net_transitions"] and summary["incoherent_rate"]<=gate["max_incoherent_rate"] and summary["mean_repeated_trigram_fraction"]<gate["max_mean_repeated_trigram_fraction"]
    result={"status":"complete","run_id":output.name,"validation_passed":passed,**summary}; writer.json("paired_results.json",rows); writer.json("results.json",result)
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(["baseline","direct"],[summary["baseline_asr"],summary["direct_asr"]]); ax.set(ylabel="OpenAI ASR",ylim=(0,1)); fig.tight_layout(); fig.savefig(output/"validation_asr.png",dpi=180); plt.close(fig)
    (output/"RESULTS.md").write_text(f"# Frozen direct-steering validation\n\n- Baseline: {summary['baseline_successes']}/25 ({summary['baseline_asr']:.0%})\n- Direct: {summary['direct_successes']}/25 ({summary['direct_asr']:.0%})\n- Paired delta: {summary['paired_delta']:+.0%}\n- Incoherent: {summary['incoherent_rate']:.0%}\n- Repetition: {summary['mean_repeated_trigram_fraction']:.4f}\n- Gate passed: {passed}\n")
    writer.json("checkpoint.json",{"config_fingerprint":config_hash,"phase":"complete","rows":rows,"error_count":errors,"retry_count":retries}); final={"run_id":output.name,"phase":"complete","completed":50,"total":50,"completed_fraction":1.0,"error_count":errors,"retry_count":retries,"validation_passed":passed,**summary}; writer.json("progress.json",final); update_dashboard(output,final); writer.commit(); return result
