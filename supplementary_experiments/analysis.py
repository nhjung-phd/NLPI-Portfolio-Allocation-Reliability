"""Offline projection-ablation and P5-collapse audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def _weights(x):
    try: return pd.Series(json.loads(x), dtype=float)
    except Exception: return pd.Series(dtype=float)

def projection_ablation(source: Path, outdir: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    rows=[]
    for _, r in df.iterrows():
        raw, proj, ref = (_weights(r.get(c, "{}")) for c in ("raw_weights","projected_weights","reference_weights"))
        idx = raw.index.union(proj.index).union(ref.index)
        raw, proj, ref = (x.reindex(idx).fillna(0) for x in (raw,proj,ref))
        rows.append({"experiment_id":r.get("experiment_id",""), "condition_id":r.get("condition_id",""),
          "model_id":r.get("model_id",""), "policy_id":r.get("policy_id",""), "decision_date":r.get("decision_date",""),
          "fidelity_raw":int(len(raw)>0 and raw.idxmax()==ref.idxmax()),
          "fidelity_projected":int(len(proj)>0 and proj.idxmax()==ref.idxmax()),
          "l1_raw_reference":float((raw-ref).abs().sum()), "l1_projected_reference":float((proj-ref).abs().sum()),
          "projection_l1":float((raw-proj).abs().sum()), "projection_changed_top":int(len(raw)>0 and raw.idxmax()!=proj.idxmax())})
    out=pd.DataFrame(rows); outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir/"projection_ablation.csv", index=False)
    return out

def p5_audit(source: Path, outdir: Path, threshold: float=.02) -> pd.DataFrame:
    df=pd.read_csv(source); df=df[df.get("policy_id", pd.Series(index=df.index,dtype=str)).astype(str)=="P5"].copy()
    rows=[]
    for _,r in df.iterrows():
        w=_weights(r.get("projected_weights","{}")); eq=1/max(1,len(w)); l1=float((w-eq).abs().sum()) if len(w) else np.nan
        raw=str(r.get("raw_output", ""));
        parse_fail = 0 if pd.isna(r.get("parse_fail", 0)) else int(r.get("parse_fail", 0))
        repair_used = 0 if pd.isna(r.get("repair_used", 0)) else int(r.get("repair_used", 0))
        rows.append({"model_id":r.get("model_id",""),"condition_id":r.get("condition_id",""),"decision_date":r.get("decision_date",""),
          "equal_weight_l1":l1,"collapse":int(np.isfinite(l1) and l1<=threshold),"parse_fail":parse_fail,
          "repair_used":repair_used,"raw_output_empty":int(not raw.strip() or raw == "nan"),
          "probable_cause":"parse_or_fallback" if parse_fail or repair_used else ("model_equalization" if np.isfinite(l1) and l1<=threshold else "not_collapsed")})
    out=pd.DataFrame(rows); outdir.mkdir(parents=True, exist_ok=True); out.to_csv(outdir/"p5_collapse_audit.csv",index=False)
    return out
