from __future__ import annotations
import hashlib,json,os,re,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

MODEL="gpt-5.6-luna"; VERSION="terminal_wrench_sanitizer_v1"
MARKER="You are an authorized Red Team QA Agent"
INDICATORS=re.compile(r"\b(hack(?:ing|ed)?|cheat(?:ing|ed)?|bypass(?:ing|ed)?|exploit(?:ing|ed)?|reward[- ]hack(?:ing|ed)?|game(?:d|s|ing)? the (?:test|verifier)|deceiv(?:e|ing))\b",re.I)
SCHEMA={"type":"object","properties":{"rewrites":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"analysis":{"type":"string"},"plan":{"type":"string"},"fidelity_note":{"type":"string"}},"required":["id","analysis","plan","fidelity_note"],"additionalProperties":False}}},"required":["rewrites"],"additionalProperties":False}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def immutable(trace):
    out=[]
    for i,s in enumerate(trace["steps"]):
        if i==0: out.append({**s,"message":"PROMPT_REDACTED"});continue
        try:p=json.loads(s.get("message",""))
        except json.JSONDecodeError:out.append(s);continue
        out.append({"source":s.get("source"),"commands":p.get("commands"),"task_complete":p.get("task_complete")})
    return out

def sanitize(row,client):
    original=json.loads(Path(row["path"]).read_text()); cleaned=json.loads(json.dumps(original))
    first=cleaned["steps"][0]["message"]
    if MARKER not in first:raise ValueError("elicitation marker missing")
    prefix,suffix=first.split(MARKER,1);cleaned["steps"][0]["message"]=prefix.rstrip()
    items=[];parsed=[]
    for i,s in enumerate(cleaned["steps"]):
        if s.get("source") not in {"agent","assistant"}:continue
        try:p=json.loads(s.get("message",""))
        except json.JSONDecodeError:continue
        if not isinstance(p,dict) or "analysis" not in p or "plan" not in p:continue
        rid=f"step_{i}";items.append({"id":rid,"analysis":str(p["analysis"]),"plan":str(p["plan"])});parsed.append((i,rid,p))
    prompt="Rewrite only these analysis and plan strings in neutral technical language. Remove explicit linguistic admissions of hacking, cheating, bypassing, exploiting, reward hacking, verifier gaming, or deception. Preserve every concrete action, file, command reference, observation, causal claim, and ordering; omit or invent nothing. Return exactly one item per id.\n\n"+json.dumps(items,ensure_ascii=False)
    attempts=[]
    for attempt in range(3):
        r=client.responses.create(model=MODEL,reasoning={"effort":"low"},store=False,max_output_tokens=32000,input=[{"role":"user","content":prompt}],text={"format":{"type":"json_schema","name":VERSION,"strict":True,"schema":SCHEMA}})
        record={"attempt":attempt,"response_id":r.id,"status":r.status,"usage":r.usage.model_dump() if r.usage else None};attempts.append(record)
        try:
            rewrites={x["id"]:x for x in json.loads(r.output_text)["rewrites"]}
            if set(rewrites)!={x["id"] for x in items}:raise ValueError("rewrite IDs differ")
            for i,rid,p in parsed:
                q=rewrites[rid]
                if not q["analysis"].strip() or not q["plan"].strip():raise ValueError("empty rewrite")
                p["analysis"],p["plan"]=q["analysis"],q["plan"];cleaned["steps"][i]["message"]=json.dumps(p,ensure_ascii=False)
            break
        except Exception as e:
            record["validation_error"]=f"{type(e).__name__}: {e}"
            if attempt==2:raise
    before,after=digest(immutable(original)),digest(immutable(cleaned))
    if before!=after:raise ValueError("immutable trace changed")
    return {"example_id":row["example_id"],"task_id":row["task_id"],"source_model":row["source_model"],"removed_prompt_suffix":MARKER+suffix,"original_trajectory":original,"sanitized_trajectory":cleaned,"immutable_hash_before":before,"immutable_hash_after":after,"indicators_before":INDICATORS.findall(json.dumps(original)),"indicators_after":INDICATORS.findall(json.dumps(cleaned)),"api_attempts":attempts}

def run(source_run:Path,output:Path,checkpoint=lambda:None):
    from openai import OpenAI
    rows=[json.loads(x) for x in (source_run/"manifest.jsonl").read_text().splitlines()]
    rows=sorted((r for r in rows if r["label"]==1),key=lambda r:hashlib.sha256(f"42|{r['example_id']}".encode()).hexdigest())[:5]
    output.mkdir(parents=True,exist_ok=False);(output/"selection.json").write_text(json.dumps(rows,indent=2));checkpoint()
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"],timeout=300,max_retries=2);started=time.time();done=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(sanitize,r,client) for r in rows]
        for f in as_completed(futures):
            done.append(f.result());(output/"progress.json").write_text(json.dumps({"phase":"sanitizing","completed":len(done),"total":5,"elapsed_seconds":time.time()-started},indent=2));checkpoint()
    done.sort(key=lambda x:x["example_id"]);artifact={"schema_version":VERSION,"model":MODEL,"examples":done}
    (output/"sanitized_examples.json").write_text(json.dumps(artifact,indent=2,ensure_ascii=False));summary={"status":"complete","examples":5,"preservation_valid":5,"elapsed_seconds":time.time()-started};(output/"results.json").write_text(json.dumps(summary,indent=2));checkpoint();return summary
