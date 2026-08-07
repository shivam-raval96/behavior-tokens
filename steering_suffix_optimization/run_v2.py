"""Resumable response-position suffix optimization and paired evaluation."""
from __future__ import annotations

import csv, hashlib, json, os, random, signal, time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .harmbench_eval import TEMPLATE, atomic_json, label, paired_delta
from .layout_v2 import response_layout
from .optimizer import ActivationTarget
from .optimizer_v2 import ResponsePositionOptimizer
from .positive_control_run import repeated_trigram_fraction


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    temporary.replace(path)


def _plot(output: Path, history: list[dict], result: dict, suffix_text: str) -> None:
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5))
    left.plot([x["step"] for x in history],[x["loss_fp32"] for x in history],label="loss")
    left.set(xlabel="Optimization step",ylabel="FP32 objective",title="Activation matching")
    twin=left.twinx(); twin.plot([x["step"] for x in history],[x["cosine"] for x in history],color="tab:orange",label="cosine")
    twin.set_ylabel("Cosine to negative refusal direction")
    names=list(result["conditions"]); vals=[result["conditions"][x]["asr"] for x in names]
    right.bar(names,vals); right.set(ylim=(0,1),ylabel="HarmBench ASR",title="Held-out attack success")
    fig.suptitle("Best suffix: "+suffix_text.replace("\n"," ")[:140],fontsize=9)
    fig.tight_layout(); fig.savefig(output/"optimization_and_judge.png",dpi=180); plt.close(fig)


def run(config_path: Path, dataset_csv: Path, output_root: Path, run_id: str,
        mode: str="fresh", commit: Callable[[],None] | None=None) -> dict:
    config=yaml.safe_load(config_path.read_text()); raw=config_path.read_bytes()
    fingerprint=hashlib.sha256(raw).hexdigest(); output=output_root/run_id; output.mkdir(parents=True,exist_ok=True)
    checkpoint_path=output/"checkpoint.json"; progress_path=output/"progress.json"
    if mode not in {"fresh","resume"}: raise ValueError("mode must be fresh or resume")
    if mode=="fresh" and checkpoint_path.exists(): raise FileExistsError("fresh run already exists")
    (output/"config.yaml").write_bytes(raw); atomic_json(output/"resolved_config.json",config)
    checkpoint=json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if mode=="resume" and (not checkpoint or checkpoint.get("config_fingerprint")!=fingerprint):
        raise ValueError("missing checkpoint or config fingerprint mismatch")
    with dataset_csv.open(newline="") as stream: source=list(csv.DictReader(stream))
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["model_revision"])
    model=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["model_revision"],torch_dtype=torch.bfloat16,device_map="auto").eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    vector_np=np.load(Path(config_path).parent.parent/config["vector"].split("steering_suffix_optimization/",1)[-1]).astype("float32") if not Path(config["vector"]).exists() else np.load(config["vector"]).astype("float32")
    vector=torch.from_numpy(vector_np); vector=vector/vector.norm()
    target=ActivationTarget(layer=config["hidden_state_index"],vector=vector,scale=config["target_additive_norms"][0])
    layouts=[]; initial=None
    for index in config["train_indices"]:
        layout,marker=response_layout(source[index]["goal"].strip(),source[index][config["measurement_response_column"]].strip(),tokenizer,config["suffix_length"],config["system_prompt"])
        layouts.append(layout); initial=marker
    optimizer=ResponsePositionOptimizer(model,tokenizer,target,top_k=config["top_k"],candidate_batch_size=config["candidate_batch_size"],norm_weight=config["norm_weight"],consistency_weight=config["consistency_weight"])
    baselines=optimizer.baselines(layouts); generator=torch.Generator(device=optimizer.device)
    history=[]; start_step=0; suffix=initial
    if checkpoint and checkpoint.get("phase")=="optimization":
        start_step=int(checkpoint["next_step"]); suffix=torch.tensor(checkpoint["suffix_ids"],dtype=torch.long)
        history=checkpoint["history"]; generator.set_state(torch.tensor(checkpoint["generator_state"],dtype=torch.uint8,device=optimizer.device))
    elif checkpoint and checkpoint.get("phase") in {"generation","judge"}:
        history=json.loads((output/"optimization_history.json").read_text())
        suffix=torch.tensor(json.loads((output/"best_suffix.json").read_text())["token_ids"],dtype=torch.long)
        start_step=config["steps"]
        generator.manual_seed(config["seed"])
    else: generator.manual_seed(config["seed"])
    interrupted=False
    def stop(_signum,_frame):
        nonlocal interrupted; interrupted=True
    previous=signal.signal(signal.SIGTERM,stop); began=time.monotonic()
    try:
        for step in tqdm(range(start_step,config["steps"]),desc="suffix optimization",unit="step"):
            suffix,_,accepted=optimizer.step(suffix,layouts,baselines,generator)
            metrics=optimizer.metrics(suffix,layouts,baselines); metrics.update(step=step+1,accepted=accepted,suffix_ids=suffix.tolist())
            history.append(metrics)
            state={"status":"running","phase":"optimization","completed":step+1,"total":config["steps"],"next_step":step+1,"run_id":run_id,"config_fingerprint":fingerprint,"elapsed_seconds":time.monotonic()-began,"throughput":(step+1-start_step)/max(time.monotonic()-began,1e-6),"latest_objective":metrics,"best_metric":min(x["loss_fp32"] for x in history),"best_state":min(history,key=lambda x:x["loss_fp32"])["suffix_ids"],"suffix_ids":suffix.tolist(),"history":history,"generator_state":generator.get_state().cpu().tolist(),"error_count":0,"retry_count":int(checkpoint.get("retry_count",-1)+1) if mode=="resume" else 0}
            atomic_json(checkpoint_path,state); atomic_json(progress_path,{k:v for k,v in state.items() if k not in {"history","generator_state"}}); commit and commit()
            print("METRIC "+json.dumps({k:v for k,v in state.items() if k not in {"history","generator_state"}}),flush=True)
            if interrupted: state["status"]="stopped"; atomic_json(checkpoint_path,state); commit and commit(); raise KeyboardInterrupt
    finally: signal.signal(signal.SIGTERM,previous)
    best=min(history,key=lambda x:x["loss_fp32"]); suffix=torch.tensor(best["suffix_ids"],dtype=torch.long)
    suffix_text=tokenizer.decode(suffix); random_gen=torch.Generator().manual_seed(config["random_suffix_seed"])
    allowed=torch.tensor([i for i in range(len(tokenizer)) if i not in tokenizer.all_special_ids]); random_suffix=allowed[torch.randint(len(allowed),(config["suffix_length"],),generator=random_gen)]
    atomic_json(output/"best_suffix.json",{"token_ids":suffix.tolist(),"text":suffix_text,"metrics":best,"random_control_ids":random_suffix.tolist(),"random_control_text":tokenizer.decode(random_suffix)})
    atomic_json(output/"optimization_history.json",history); commit and commit()
    generations_path=output/"generations.jsonl"
    rows=[json.loads(line) for line in generations_path.read_text().splitlines() if line] if generations_path.exists() else []
    done={(x["condition"],x["dataset_index"]) for x in rows}
    conditions={"baseline":None,"optimized_suffix":suffix,"random_suffix":random_suffix}
    for condition,token_suffix in conditions.items():
        for dataset_index in tqdm(config["evaluation_indices"],desc=condition,unit="prompt"):
            if (condition,dataset_index) in done: continue
            prompt=source[dataset_index]["goal"].strip(); messages=[{"role":"user","content":prompt}]
            clean=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")[0]
            if token_suffix is None: ids=clean
            else:
                layout,_=response_layout(prompt,"x",tokenizer,config["suffix_length"],config["system_prompt"])
                ids=torch.cat((layout.before_suffix_ids,token_suffix,layout.after_suffix_ids))
            ids=ids[None].to(model.device); generated=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=config["max_new_tokens"],pad_token_id=tokenizer.eos_token_id)
            new=generated[0,ids.shape[1]:]; text=tokenizer.decode(new,skip_special_tokens=True)
            rows.append({"condition":condition,"dataset_index":dataset_index,"prompt":prompt,"suffix_ids":[] if token_suffix is None else token_suffix.tolist(),"suffix_text":"" if token_suffix is None else tokenizer.decode(token_suffix),"response":text,"generated_tokens":int(new.numel()),"eos":bool((new==tokenizer.eos_token_id).any()),"repeated_trigram_fraction":repeated_trigram_fraction(text)})
            _write_jsonl(generations_path,rows)
            progress={"status":"running","phase":"generation","completed":len(rows),"total":len(config["evaluation_indices"])*3,"run_id":run_id,"config_fingerprint":fingerprint,"latest_condition":condition,"error_count":0}; atomic_json(checkpoint_path,progress); atomic_json(progress_path,progress); commit and commit()
    del optimizer,model; torch.cuda.empty_cache()
    judgments_path=output/"harmbench_judgments.jsonl"
    judgments=[json.loads(line) for line in judgments_path.read_text().splitlines() if line] if judgments_path.exists() else []
    judged={(x["condition"],x["dataset_index"]) for x in judgments}
    pending=[x for x in rows if (x["condition"],x["dataset_index"]) not in judged]
    judge_tokenizer=judge=None
    if pending:
        judge_tokenizer=AutoTokenizer.from_pretrained(config["judge"],revision=config["judge_revision"],token=os.environ.get("HF_TOKEN"),use_fast=False); judge_tokenizer.padding_side=judge_tokenizer.truncation_side="left"
        if judge_tokenizer.pad_token_id is None: judge_tokenizer.pad_token=judge_tokenizer.eos_token
        judge=AutoModelForCausalLM.from_pretrained(config["judge"],revision=config["judge_revision"],token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval(); judge.generation_config.temperature=judge.generation_config.top_p=None
    for start in tqdm(range(0,len(pending),config["judge_batch_size"]),desc="HarmBench",unit="batch"):
        batch=pending[start:start+config["judge_batch_size"]]; encoded=judge_tokenizer([TEMPLATE.format(behavior=x["prompt"],generation=x["response"]) for x in batch],padding=True,truncation=True,max_length=config["judge_max_input_tokens"],return_tensors="pt").to("cuda")
        generated=judge.generate(**encoded,do_sample=False,max_new_tokens=1,pad_token_id=judge_tokenizer.pad_token_id); predictions=judge_tokenizer.batch_decode(generated[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
        for item,prediction in zip(batch,predictions): judgments.append({**item,"raw_prediction":prediction,"parsed_prediction":label(prediction),"valid":label(prediction)!="invalid","success":label(prediction)=="yes"})
        _write_jsonl(judgments_path,judgments); rates={name:float(np.mean([x["success"] for x in judgments if x["condition"]==name])) for name in {x["condition"] for x in judgments}}
        progress={"status":"running","phase":"judge","completed":len(judgments),"total":len(rows),"run_id":run_id,"config_fingerprint":fingerprint,"current_asr":rates,"invalid_count":sum(not x["valid"] for x in judgments),"error_count":0}; atomic_json(checkpoint_path,progress); atomic_json(progress_path,progress); commit and commit(); print("METRIC "+json.dumps(progress),flush=True)
    baseline=[x["success"] for x in judgments if x["condition"]=="baseline"]; summary={}
    for name in conditions:
        selected=[x for x in judgments if x["condition"]==name]; flags=[x["success"] for x in selected]
        summary[name]={"n":len(flags),"asr":float(np.mean(flags)),"invalid":sum(not x["valid"] for x in selected),"eos_rate":float(np.mean([x["eos"] for x in selected])),"mean_repeated_trigram_fraction":float(np.mean([x["repeated_trigram_fraction"] for x in selected]))}
        if name!="baseline": summary[name]["paired_vs_baseline"]=paired_delta(baseline,flags,config["bootstrap_seed"],config["bootstrap_samples"])
    result={"status":"complete","run_id":run_id,"config_fingerprint":fingerprint,"best_suffix":{"text":suffix_text,"token_ids":suffix.tolist(),"metrics":best},"conditions":summary,"generation_rows":len(rows),"judgment_rows":len(judgments)}
    _plot(output,history,result,suffix_text); atomic_json(output/"results.json",result); atomic_json(checkpoint_path,result); atomic_json(progress_path,result)
    (output/"RESULTS.md").write_text("# Layer-10 response-position suffix\n\n- Best suffix: `"+suffix_text.replace("`","\\`")+"`\n"+"\n".join(f"- {name}: HarmBench ASR {value['asr']:.3f}" for name,value in summary.items())+"\n"); commit and commit(); return result
