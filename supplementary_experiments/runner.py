"""Bridge, baseline and repeatability experiments isolated from original results."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, traceback
from pathlib import Path
import pandas as pd
from q1_experiments.runner import (BASE_POLICY_PROMPTS, DEFAULT_TICKERS, decision_indices,
    evaluate_one_call, load_data)
from q1_experiments.reference_policies import reference_weights
from engine.strategies import project_capped_simplex
from .baselines import rule_parse, tfidf_match
from .analysis import projection_ablation, p5_audit
from .checkpoint import TaskLedger

def sha(x): return hashlib.sha256(str(x).encode()).hexdigest()
def cfg(a,out):
    return {"run_id":a.experiment_id,"ollama_url":a.ollama_url,"rebalance_days":a.rebalance,"tcost":a.tcost,
      "max_weight":a.maxw,"turnover_cap":a.turncap,"prompt_cap_pct":a.prompt_cap,"seed":a.seed,
      "temperature":a.temperature,"top_p":a.top_p,"ollama_connect_timeout":30,"ollama_read_timeout":a.timeout,
      "max_retries":a.max_retries,"parse_retries":a.parse_retries,"timeout_retries":1,"num_predict":a.num_predict,
      "ollama_keep_alive":"30m","result_root":str(out)}

def manifest(a,out,features):
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit="unavailable"
    stable={k:v for k,v in vars(a).items() if k not in {"retry_failed","checkpoint_every"}}
    config_hash=sha(json.dumps(stable,sort_keys=True,ensure_ascii=False))
    existing=out/"manifest.json"
    if existing.exists():
      old=json.loads(existing.read_text(encoding="utf-8"))
      if old.get("run_config_hash") not in (None,config_hash):
        raise SystemExit("Experiment ID already exists with different settings. Use a new --experiment-id.")
    m=vars(a).copy(); m.update({"python":platform.python_version(),"platform":platform.platform(),"git_commit":commit,"run_config_hash":config_hash,
      "feature_snapshot_hash":sha(features.to_csv()),"separation_guard":"supplementary_results_only"})
    (out/"manifest.json").write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding="utf-8")

def baseline_rows(features,tickers,idx,a):
    rows=[]
    for i in idx:
      row=features.iloc[i]
      for expected,text in BASE_POLICY_PROMPTS.items():
        for method in ("rule","tfidf"):
          pred,score=(rule_parse(text),1.0) if method=="rule" else tfidf_match(text)
          w=reference_weights(pred,row,tickers,a.prompt_cap/100); w=project_capped_simplex(w,a.maxw)
          ref=project_capped_simplex(reference_weights(expected,row,tickers,a.prompt_cap/100),a.maxw)
          rows.append({"experiment_id":"nl_baseline","method":method,"policy_id":expected,"predicted_policy":pred,
            "retrieval_score":score,"decision_date":str(features.index[i].date()),"policy_accuracy":int(pred==expected),
            "allocation_l1_to_reference":float((w-ref).abs().sum()),"projected_weights":w.to_json(),"reference_weights":ref.to_json()})
    return rows

def task_id(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]

def write_csv(ledger,out):
    rows=list(ledger.results())
    if not rows: return
    target=out/"supplementary_calls.csv"; tmp=target.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp,index=False); tmp.replace(target)

def make_tasks(features,idx,a,experiments):
    tasks=[]
    if "baseline" in experiments:
      for i in idx:
       for expected,text in BASE_POLICY_PROMPTS.items():
        for method in ("rule","tfidf"):
         tasks.append({"kind":"baseline","i":int(i),"policy_id":expected,"prompt_text":text,"method":method})
    if "bridge" in experiments:
      for i in idx:
       for model in a.models:
        for pid in a.policies:
         for protocol,reminder in (("A_original",False),("B_canonical",True)):
          tasks.append({"kind":"bridge","i":int(i),"model":model,"policy_id":pid,"protocol":protocol,"reminder":reminder})
    if "repeatability" in experiments:
      for i in idx:
       for model in a.models:
        for pid in a.policies:
         for rep in range(a.repeats):
          tasks.append({"kind":"repeatability","i":int(i),"model":model,"policy_id":pid,"rep":rep})
    return tasks

def execute_task(t,features,tickers,a,c):
    i=t["i"]; row=features.iloc[i]; date=features.index[i]
    if t["kind"]=="baseline":
      expected=t["policy_id"]; method=t["method"]
      pred,score=(rule_parse(t["prompt_text"]),1.0) if method=="rule" else tfidf_match(t["prompt_text"])
      w=project_capped_simplex(reference_weights(pred,row,tickers,a.prompt_cap/100),a.maxw)
      ref=project_capped_simplex(reference_weights(expected,row,tickers,a.prompt_cap/100),a.maxw)
      return {"experiment_id":"nl_baseline","method":method,"policy_id":expected,"predicted_policy":pred,
        "retrieval_score":score,"decision_date":str(date.date()),"policy_accuracy":int(pred==expected),
        "allocation_l1_to_reference":float((w-ref).abs().sum()),"projected_weights":w.to_json(),"reference_weights":ref.to_json()}
    if t["kind"]=="bridge":
      return evaluate_one_call(experiment_id="bridge_validation",condition_id=t["protocol"],model=t["model"],policy_id=t["policy_id"],prompt_text=BASE_POLICY_PROMPTS[t["policy_id"]],row=row,tickers=tickers,cfg=c,date=date,dry_run=a.dry_run,constraint_reminder=t["reminder"])
    cc=dict(c); cc["seed"]=a.seed+t["rep"]
    r=evaluate_one_call(experiment_id="repeatability",condition_id=f"repeat_{t['rep']:02d}",model=t["model"],policy_id=t["policy_id"],prompt_text=BASE_POLICY_PROMPTS[t["policy_id"]],row=row,tickers=tickers,cfg=cc,date=date,dry_run=a.dry_run,constraint_reminder=True)
    r["repeat_no"]=t["rep"]; r["generation_seed"]=cc["seed"]; return r

def parse_args(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--experiments",nargs="+",default=["bridge","baseline","repeatability"],choices=["bridge","baseline","repeatability","projection_ablation","p5_audit","all"])
 p.add_argument("--models",nargs="+",default=["llama3.1:8b"]); p.add_argument("--policies",nargs="+",default=["P1","P2","P3","P4","P5","P6"])
 p.add_argument("--tickers",nargs="+",default=DEFAULT_TICKERS); p.add_argument("--start",default="2010-01-01"); p.add_argument("--end",default="2025-12-29")
 p.add_argument("--max-dates",type=int,default=7); p.add_argument("--repeats",type=int,default=5); p.add_argument("--seed",type=int,default=42)
 p.add_argument("--temperature",type=float,default=0.0); p.add_argument("--top-p",type=float,default=.9); p.add_argument("--rebalance",type=int,default=42)
 p.add_argument("--tcost",type=float,default=.001); p.add_argument("--maxw",type=float,default=.60); p.add_argument("--turncap",type=float,default=.25); p.add_argument("--prompt-cap",type=float,default=60)
 p.add_argument("--ollama-url",default="http://localhost:11434"); p.add_argument("--timeout",type=float,default=900); p.add_argument("--max-retries",type=int,default=2); p.add_argument("--parse-retries",type=int,default=1); p.add_argument("--num-predict",type=int,default=512)
 p.add_argument("--out-root",default="results/supplementary"); p.add_argument("--experiment-id",default="supplementary_v1"); p.add_argument("--source-log",default=""); p.add_argument("--dry-run",action="store_true"); p.add_argument("--synthetic-data",action="store_true")
 p.add_argument("--retry-failed",action="store_true",help="Retry tasks previously marked failed")
 p.add_argument("--checkpoint-every",type=int,default=1,help="Rebuild summary CSV every N completed tasks")
 return p.parse_args(argv)

def main(argv=None):
 a=parse_args(argv); out=Path(a.out_root)/a.experiment_id
 if out.resolve().is_relative_to((Path("outputs").resolve())): raise SystemExit("Supplementary output must not be under outputs/ (original study).")
 out.mkdir(parents=True,exist_ok=True); prices,features,tickers=load_data(a); manifest(a,out,features)
 idx=decision_indices(features,a.rebalance,"stratified",10,a.seed,a.max_dates); ex=set(a.experiments); ex={"bridge","baseline","repeatability","projection_ablation","p5_audit"} if "all" in ex else ex; c=cfg(a,out)
 ledger=TaskLedger(out); recovered=ledger.recover_interrupted()
 tasks=make_tasks(features,idx,a,ex)
 for t in tasks: ledger.add(task_id(t),t["kind"],t)
 ledger.commit(); print(f"[RESUME] recovered={recovered} status={ledger.counts()}",flush=True)
 done_since_csv=0
 try:
  for tid,kind,t in ledger.pending(a.retry_failed):
   ledger.start(tid)
   try:
    result=execute_task(t,features,tickers,a,c); result["task_id"]=tid
    ledger.complete(tid,result); done_since_csv+=1
    if done_since_csv >= max(1,a.checkpoint_every): write_csv(ledger,out); done_since_csv=0
    print(f"[CHECKPOINT] {tid} {kind} status={ledger.counts()}",flush=True)
   except KeyboardInterrupt: raise
   except Exception as exc:
    ledger.fail(tid,f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    print(f"[FAILED] {tid}: {exc}",flush=True)
 finally:
  write_csv(ledger,out); print(f"[STATUS] {ledger.counts()}",flush=True); ledger.close()
 src=Path(a.source_log) if a.source_log else out/"supplementary_calls.csv"
 if "projection_ablation" in ex and src.exists(): projection_ablation(src,out/"analysis")
 if "p5_audit" in ex and src.exists(): p5_audit(src,out/"analysis")
 print(f"[DONE] {out}")
if __name__=="__main__": main()
