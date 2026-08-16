#!/usr/bin/env python3
"""Reproducible post-hoc statistics for the NLPI paper (no LLM calls)."""
from __future__ import annotations

import argparse, json, math, re, shutil, sys, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
SEED = 20260801
BOOT_N = 2000


def locate_one(root: Path, pattern: str, prefer: str | None = None) -> Path:
    xs = [p for p in root.glob(pattern) if p.is_file() and "reliability_smoke" not in str(p) and "reliability_retry" not in str(p)]
    if prefer:
        ys = [p for p in xs if prefer in str(p)]
        if ys: xs = ys
    if not xs:
        raise FileNotFoundError(f"필수 파일을 찾지 못했습니다: {pattern}")
    return max(xs, key=lambda p: p.stat().st_size)


def bh_holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); out = np.empty(len(p))
    running = 0.0; m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*p[idx]); out[idx] = min(1.0, running)
    return out


def bootstrap_mean_ci(x, rng, n=BOOT_N):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return (np.nan, np.nan)
    vals = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return tuple(np.quantile(vals, [0.025, .975]))


def load_master(root: Path):
    compact_base = root / "results/reliability_primary/q1_decision_log.csv"
    compact_qwen = root / "results/reliability_qwen/q1_decision_log.csv"
    base = compact_base if compact_base.is_file() else locate_one(root, "outputs/**/reliability_resumable_20260727/logs/q1_decision_log.csv")
    qwen = compact_qwen if compact_qwen.is_file() else locate_one(root, "outputs/**/qwen_model_generalization_108/logs/q1_decision_log.csv")
    a = pd.read_csv(base, low_memory=False); b = pd.read_csv(qwen, low_memory=False)
    a["source_package"]="reliability_resumable"; b["source_package"]="qwen_108"
    m = pd.concat([a,b], ignore_index=True, sort=False)
    m["decision_date"] = pd.to_datetime(m["decision_date"], errors="coerce")
    for c in ["json_valid","parse_fail","repair_used","raw_fidelity","projected_fidelity","post_feasible_budget","post_feasible_cap","post_feasible_longonly"]:
        if c in m: m[c]=pd.to_numeric(m[c], errors="coerce")
    return m, base, qwen


def inventory(root, paths):
    rows=[]
    for p in paths:
        try:
            d=pd.read_csv(p, low_memory=False); rows.append({"file":str(p.relative_to(root)),"rows":len(d),"columns":len(d.columns),"bytes":p.stat().st_size})
        except Exception: rows.append({"file":str(p.relative_to(root)),"rows":np.nan,"columns":np.nan,"bytes":p.stat().st_size})
    return pd.DataFrame(rows)


def cross_family(master, out, rng):
    d=master[master.experiment_id.eq("model_family_generalization")].copy()
    models=sorted(d.model_id.dropna().unique()); metrics=[x for x in ["raw_fidelity","projected_fidelity","top3_overlap","allocation_l1_to_reference","projection_l1","ew_l1_distance"] if x in d]
    desc=[]; omni=[]; post=[]
    for policy in sorted(d.policy_id.dropna().unique()):
      z=d[d.policy_id.eq(policy)]
      for metric in metrics:
        pv=z.pivot_table(index="decision_date",columns="model_id",values=metric,aggfunc="first").dropna()
        for model in models:
          x=pd.to_numeric(z.loc[z.model_id.eq(model),metric],errors="coerce").dropna().to_numpy(); lo,hi=bootstrap_mean_ci(x,rng)
          desc.append({"policy":policy,"metric":metric,"model":model,"n":len(x),"mean":np.mean(x) if len(x) else np.nan,"sd":np.std(x,ddof=1) if len(x)>1 else np.nan,"median":np.median(x) if len(x) else np.nan,"ci_low":lo,"ci_high":hi})
        if len(pv)>=2 and pv.shape[1]>=3 and not np.allclose(pv.to_numpy(),pv.to_numpy()[0,0],equal_nan=True):
          try: stat,p=stats.friedmanchisquare(*[pv[c].to_numpy() for c in pv.columns]); W=stat/(len(pv)*(len(pv.columns)-1))
          except ValueError: stat=p=W=np.nan
        else: stat=p=W=np.nan
        omni.append({"policy":policy,"metric":metric,"n_blocks":len(pv),"n_models":pv.shape[1],"friedman_chi2":stat,"p_value":p,"kendalls_w":W,"constant_or_untestable":not np.isfinite(p)})
        pairs=[]
        if np.isfinite(p) and p<ALPHA:
          for a,b in combinations(pv.columns,2):
            delta=pv[a]-pv[b]
            all_zero=bool(np.allclose(delta.to_numpy(),0.0,equal_nan=False))
            if all_zero:
              ws=wp=np.nan; status="not_testable_all_zero"
            else:
              try: ws,wp=stats.wilcoxon(pv[a],pv[b],zero_method="pratt",alternative="two-sided"); status="tested"
              except ValueError: ws=wp=np.nan; status="not_testable"
            pairs.append({"pair_id":f"{policy}|{metric}|{a}|{b}","model_a":a,"model_b":b,"wilcoxon_w":ws,"p_raw":wp,"mean_paired_difference":float(delta.mean()),"median_paired_difference":float(delta.median()),"test_status":status})
          valid=[x for x in pairs if np.isfinite(x["p_raw"])]
          adjusted=bh_holm([x["p_raw"] for x in valid]) if valid else []
          adj_by_id={x["pair_id"]:ap for x,ap in zip(valid,adjusted)}
          for x in pairs:
            post.append({"policy":policy,"metric":metric,**{k:v for k,v in x.items() if k!="pair_id"},"p_holm":adj_by_id.get(x["pair_id"],np.nan)})
    pd.DataFrame(omni).to_csv(out/"05_cross_family_omnibus.csv",index=False)
    pd.DataFrame(post).to_csv(out/"06_cross_family_posthoc.csv",index=False)
    pd.DataFrame(desc).to_csv(out/"07_cross_family_descriptives.csv",index=False)
    return pd.DataFrame(omni), pd.DataFrame(post), pd.DataFrame(desc)


def reliability_gee(master, out):
    rows=[]
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    specs={
      "prompt_robustness":["model_id","policy_id","paraphrase_id"],
      "ticker_masking":["model_id","policy_id","mask_condition"],
      "policy_complexity":["model_id","complexity_level"],
      "constraint_conflict_stress":["model_id","conflict_type"],
      "model_family_generalization":["model_id","policy_id"],
    }
    for experiment,predictors in specs.items():
      d=master[master.experiment_id.eq(experiment)].dropna(subset=["decision_date","projected_fidelity","model_id"]).copy()
      d["cluster"]=d["decision_date"].dt.strftime("%Y-%m-%d"); d["success"]=(d["projected_fidelity"]>=0.5).astype(int)
      # A policy/complexity cell with no outcome variation supplies no finite
      # within-cell logistic information and can cause complete separation.
      cell="policy_id" if "policy_id" in predictors else "complexity_level"
      excluded=[]
      if cell in d:
        varying=d.groupby(cell,dropna=False).success.nunique()
        excluded=varying[varying<2].index.astype(str).tolist()
        d=d[d[cell].astype(str).isin(varying[varying>=2].index.astype(str))].copy()
      active=[x for x in predictors if x in d and d[x].nunique(dropna=True)>1]
      if len(d)==0 or d.success.nunique()<2 or not active:
        rows.append({"experiment":experiment,"outcome":"projected_fidelity>=0.5","term":"DESCRIPTIVE_ONLY","fit_status":"not_estimable_after_constant_cell_exclusion","excluded_constant_cells":";".join(excluded),"n":len(d),"clusters":d.cluster.nunique() if len(d) else 0})
        continue
      formula="success ~ "+" + ".join(f"C({x})" for x in active)
      try:
        family_name="binomial_logit"
        with warnings.catch_warnings():
          warnings.simplefilter("ignore",RuntimeWarning)
          fit=smf.gee(formula,groups="cluster",data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
        probe=np.r_[fit.params.to_numpy(),fit.bse.to_numpy(),fit.pvalues.to_numpy()]
        if not np.isfinite(probe).all():
          # Explicit separation fallback: cluster-robust linear probability GEE.
          family_name="gaussian_identity_fallback"
          fit=smf.gee(formula,groups="cluster",data=d,family=sm.families.Gaussian(),cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
        ci=fit.conf_int()
        for term in fit.params.index:
          vals=[fit.params[term],fit.bse[term],fit.pvalues[term],ci.loc[term,0],ci.loc[term,1]]
          finite=all(np.isfinite(vals)); clip=lambda x: math.exp(float(np.clip(x,-700,700))) if np.isfinite(x) else np.nan
          is_logit=family_name=="binomial_logit"
          rows.append({"experiment":experiment,"outcome":"projected_fidelity>=0.5","term":term,"fit_status":"ok" if finite else "nonfinite_do_not_use","model_family":family_name,"formula":formula,"coefficient":fit.params[term],"odds_ratio":clip(fit.params[term]) if is_logit else np.nan,"se":fit.bse[term],"p_value":fit.pvalues[term],"ci_low":ci.loc[term,0],"ci_high":ci.loc[term,1],"n":len(d),"clusters":d.cluster.nunique(),"excluded_constant_cells":";".join(excluded)})
      except Exception as e:
        rows.append({"experiment":experiment,"outcome":"projected_fidelity>=0.5","term":type(e).__name__,"fit_status":"failed_do_not_use","n":len(d),"clusters":d.cluster.nunique(),"excluded_constant_cells":";".join(excluded),"message":str(e)})
    pd.DataFrame(rows).to_csv(out/"08_reliability_gee.csv",index=False)
    return pd.DataFrame(rows)


def perf_stats(r):
    r=np.asarray(r,float); r=r[np.isfinite(r)]
    if len(r)<2:return dict(n=len(r),ann_return=np.nan,ann_vol=np.nan,sharpe=np.nan,mdd=np.nan)
    ar=(np.prod(1+r)**(252/len(r))-1) if np.all(r>-1) else np.nan
    av=np.std(r,ddof=1)*np.sqrt(252); sh=np.mean(r)/np.std(r,ddof=1)*np.sqrt(252) if np.std(r,ddof=1)>0 else np.nan
    eq=np.cumprod(1+r); mdd=np.min(eq/np.maximum.accumulate(eq)-1)
    return dict(n=len(r),ann_return=ar,ann_vol=av,sharpe=sh,mdd=mdd)


def circular_block_sample(x, block, rng):
    n=len(x); starts=rng.integers(0,n,size=math.ceil(n/block)); idx=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]; return x[idx]


def portfolio_analysis(root,out,rng):
    compact = root / "results/wfcv/oos_tidy.csv"
    p=compact if compact.is_file() else locate_one(root,"outputs/**/main/oos_tidy.csv"); d=pd.read_csv(p,low_memory=False); d["date"]=pd.to_datetime(d.date,errors="coerce")
    # The file contains both fold overlays and already-stitched curves.  Mixing
    # them duplicates every date.  NLPI also contains 15 distinct model-persona
    # paths under the same strategy name, so preserve that identity.
    d=d[(d.is_oos.eq(1)) & (d.segment.eq("stitched_oos"))].copy()
    d["series_id"]=np.where(d.strategy.eq("NLPI"),"NLPI["+d.model.astype(str)+"|"+d.persona.astype(str)+"]",d.strategy.astype(str))
    key=["series_id"]
    if d.duplicated(key+["date"]).any():
      raise ValueError("oos_tidy contains duplicate stitched observations within a strategy/model/persona series")
    d=d.sort_values(key+["date"]); d["return"]=d.groupby(key,dropna=False).equity.pct_change(fill_method=None)
    r=d.dropna(subset=["return"])[["date","series_id","return"]].rename(columns={"series_id":"strategy"})
    perf=pd.DataFrame([{"strategy":s,**perf_stats(g["return"])} for s,g in r.groupby("strategy")]).sort_values("sharpe",ascending=False)
    perf.to_csv(out/"09_portfolio_performance.csv",index=False)
    bench="EQUAL" if "EQUAL" in set(r.strategy) else perf.iloc[-1].strategy
    wide=r.pivot(index="date",columns="strategy",values="return").dropna(subset=[bench]); hac=[]; boots=[]
    try:
      import statsmodels.api as sm
      for s in wide.columns:
        if s==bench:continue
        x=(wide[s]-wide[bench]).dropna(); fit=sm.OLS(x.to_numpy(),np.ones((len(x),1))).fit(cov_type="HAC",cov_kwds={"maxlags":5})
        hac.append({"strategy":s,"benchmark":bench,"n":len(x),"mean_daily_excess":x.mean(),"annualized_arithmetic_excess":252*x.mean(),"hac_t":fit.tvalues[0],"hac_p":fit.pvalues[0]})
        vals=[]
        for _ in range(BOOT_N):
          xb=circular_block_sample(x.to_numpy(),20,rng); vals.append(perf_stats(xb)["sharpe"])
        lo,hi=np.quantile(vals,[.025,.975]); boots.append({"strategy":s,"benchmark":bench,"metric":"Sharpe of paired excess returns","block_length":20,"boot_n":BOOT_N,"estimate":perf_stats(x)["sharpe"],"ci_low":lo,"ci_high":hi})
    except Exception as e: hac=[{"strategy":"ERROR","benchmark":bench,"message":str(e)}]
    pd.DataFrame(hac).to_csv(out/"10_portfolio_hac.csv",index=False); pd.DataFrame(boots).to_csv(out/"11_portfolio_bootstrap.csv",index=False)
    return perf,pd.DataFrame(hac),pd.DataFrame(boots),p


def p2_audit(root,master,out):
    compact = root / "results/wfcv/prompt_fidelity.csv"
    pf=compact if compact.is_file() else locate_one(root,"outputs/**/main/prompt_fidelity.csv"); a=pd.read_csv(pf,low_memory=False); a=a[a.persona.astype(str).eq("P2")]
    b=master[(master.experiment_id.eq("model_family_generalization")) & (master.policy_id.eq("P2"))]
    rows=[
      {"item":"sample size","study_a":len(a),"study_b":len(b),"same_definition":"No","interpretation":"Study A WFCV events versus Study B common decision dates"},
      {"item":"reference target implementation","study_a":"LLMStrategy._intended_core_asset: argmin *_vol3m","study_b":"reference_weights(P2): argmin *_vol3m","same_definition":"Yes","interpretation":"Both code paths use the same low-volatility target rule"},
      {"item":"LLM execution context","study_a":"WFCV backtest prompt generated inside LLMStrategy","study_b":"standalone reliability runner prompt and common dates","same_definition":"No","interpretation":"Prompt rendering, feature snapshot, call context, and sampling protocol differ"},
      {"item":"projected fidelity mean","study_a":pd.to_numeric(a.projected_fidelity,errors="coerce").mean(),"study_b":pd.to_numeric(b.projected_fidelity,errors="coerce").mean(),"same_definition":"Outcome differs","interpretation":"Do not pool before protocol reconciliation"},
      {"item":"raw fidelity mean","study_a":pd.to_numeric(a.raw_fidelity,errors="coerce").mean(),"study_b":pd.to_numeric(b.raw_fidelity,errors="coerce").mean(),"same_definition":"Outcome differs","interpretation":"Compare pre-projection semantics separately"},
      {"item":"projection stage available","study_a":bool(a.projected_fidelity.notna().any()),"study_b":bool(b.projected_fidelity.notna().any()),"same_definition":"Yes at field level","interpretation":"Implementation semantics still require source audit"},
      {"item":"decision-date structure","study_a":"fold/event based; date not stored in prompt_fidelity","study_b":f"{b.decision_date.nunique()} common dates","same_definition":"No","interpretation":"Sampling frame differs"},
    ]
    pd.DataFrame(rows).to_csv(out/"12_p2_protocol_audit.csv",index=False)
    cols=[c for c in ["condition_id","model_id","decision_date","prompt_hash","target_asset","top_asset_raw","top_asset_projected","raw_fidelity","projected_fidelity"] if c in b]
    b[cols].to_csv(out/"13_p2_prompt_hashes.csv",index=False)
    # Extract auditable source lines mentioning P2/fidelity without modifying sources.
    src=[]
    for rel in ["q1_experiments/prompt_library.py","q1_experiments/reference_policies.py","q1_experiments/runner.py","paper_canonical.py"]:
      p=root/rel
      if p.exists():
        for i,line in enumerate(p.read_text(encoding="utf-8",errors="ignore").splitlines(),1):
          if re.search(r"P2|fidelity|target_asset|projected_top",line,re.I): src.append({"file":rel,"line":i,"text":line.strip()[:500]})
    pd.DataFrame(src).to_csv(out/"14_p2_source_trace.csv",index=False)
    pd.DataFrame([
      {"component":"policy text","study_a":"DEFENSIVE LOW-VOLATILITY; minimum _vol3m","study_b":"DEFENSIVE LOW-VOLATILITY; minimum _vol3m","comparison":"aligned"},
      {"component":"target executor","study_a":"engine/strategies.py::_intended_core_asset","study_b":"q1_experiments/reference_policies.py::reference_weights","comparison":"independent implementations; same argmin _vol3m rule"},
      {"component":"sampling","study_a":"WFCV calls across five folds; decision date absent from exported diagnostic","study_b":"18 fixed common dates per model-policy","comparison":"not aligned"},
      {"component":"observed P2 target","study_a":"BIL in all exported P2 diagnostic rows","study_b":"BIL on all 18 common dates","comparison":"aligned in observed target"},
      {"component":"observed LLM top asset","study_a":"BIL in exported P2 calls","study_b":"mostly SPY; never BIL","comparison":"not aligned"},
      {"component":"conclusion","study_a":"fidelity 1.0","study_b":"fidelity 0.0","comparison":"not a reference-code discrepancy; execution-context/protocol discontinuity"},
    ]).to_csv(out/"15_p2_code_path_comparison.csv",index=False)
    pd.DataFrame([{
      "decision":"targeted_bridge_validation_recommended",
      "additional_calls":72,
      "reason":"P2 target logic and the observed BIL reference align, but model choices change across execution contexts.",
      "minimum_design":"Replay 18 Study-B dates for four models with byte-identical prompt/feature serialization and reference execution; retain raw response, parsed weights, prompt hash, seed, temperature, and projection.",
      "scope_condition":"Required only if Study A and Study B P2 results are compared or jointly interpreted."
    }]).to_csv(out/"16_bridge_validation_decision.csv",index=False)
    return pd.DataFrame(rows),pf


def figures(desc,perf,out):
    try:
      import matplotlib.pyplot as plt, seaborn as sns
      fdir=out/"figures"; fdir.mkdir(exist_ok=True)
      z=desc[desc.metric.eq("projected_fidelity")]
      if len(z):
        plt.figure(figsize=(10,5)); sns.barplot(data=z,x="policy",y="mean",hue="model"); plt.ylim(0,1); plt.ylabel("Projected fidelity"); plt.tight_layout(); plt.savefig(fdir/"cross_family_projected_fidelity.png",dpi=200); plt.close()
      if len(perf):
        plt.figure(figsize=(11,5)); q=perf.head(15); sns.barplot(data=q,x="sharpe",y="strategy"); plt.tight_layout(); plt.savefig(fdir/"portfolio_sharpe_top15.png",dpi=200); plt.close()
    except Exception: pass


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default=".."); ap.add_argument("--output",default=None); args=ap.parse_args()
    here=Path(__file__).resolve().parent; root=Path(args.project_root).expanduser().resolve(); out=Path(args.output).resolve() if args.output else here/"analysis_outputs"; out.mkdir(parents=True,exist_ok=True)
    if not (root/"outputs").exists() and not (root/"results").exists():
        raise SystemExit(f"Project root must contain outputs/ or compact results/: {root}")
    rng=np.random.default_rng(SEED); master,base,qwen=load_master(root); master.to_csv(out/"02_master_reliability.csv",index=False)
    keys=[c for c in ["source_package","experiment_id","model_id","policy_id"] if c in master]
    master.groupby(keys,dropna=False).size().rename("n").reset_index().to_csv(out/"03_sample_audit.csv",index=False)
    dkeys=[c for c in ["experiment_id","condition_id","model_id","policy_id","decision_date","prompt_hash"] if c in master]
    dup=master.duplicated(dkeys,keep=False); pd.DataFrame([{"rows":len(master),"expected_total":5670,"total_matches_expected":len(master)==5670,"duplicate_rows_by_analysis_key":int(dup.sum()),"unique_call_ids":master.call_id.nunique() if "call_id" in master else np.nan,"json_valid_rate":master.json_valid.mean(),"parse_fail_count":master.parse_fail.sum(),"repair_used_count":master.repair_used.sum()}]).to_csv(out/"04_duplicate_audit.csv",index=False)
    omni,post,desc=cross_family(master,out,rng); gee=reliability_gee(master,out); perf,hac,boots,oos=portfolio_analysis(root,out,rng); p2,pf=p2_audit(root,master,out); figures(desc,perf,out)
    inv=inventory(root,[base,qwen,oos,pf]); inv.to_csv(out/"01_data_inventory.csv",index=False)
    report=f"""# NLPI Q2 통계분석 자동 보고서\n\n- 실행 시각: {pd.Timestamp.now()}\n- 프로젝트: `{root}`\n- 통합 신뢰성 표본: **{len(master):,}건** (기대값 5,670건과 {'일치' if len(master)==5670 else '불일치'})\n- 최초 parse failure: **{master.parse_fail.sum():.0f}건**\n- repair 사용: **{master.repair_used.sum():.0f}건**\n- 최종 JSON validity: **{master.json_valid.mean()*100:.2f}%**\n- cross-family 정책 수: **{master.loc[master.experiment_id.eq('model_family_generalization'),'policy_id'].nunique()}개**\n- Study B P2 projected fidelity 평균: **{pd.to_numeric(master.loc[(master.experiment_id.eq('model_family_generalization'))&(master.policy_id.eq('P2')),'projected_fidelity'],errors='coerce').mean():.4f}**\n\n## 해석 원칙\n\n1. 5,670건 전체를 독립 관측치로 해석하지 않는다. Friedman 검정은 공통 날짜를 block으로 사용했다.\n2. 유의한 omnibus 결과에만 Wilcoxon signed-rank 사후검정을 적용하고 Holm 보정을 보고한다.\n3. all-zero 대응차는 검정하지 않고 Holm 보정에서도 제외한다.\n4. GEE는 실험별로 적합하며, 분리가 남는 경우 군집강건 선형확률 GEE를 명시적으로 사용한다.\n5. P2 기준정책은 Study A/B에서 동일하지만 실행 문맥의 단절이 남아 두 결과를 비교·통합하려면 72건 bridge validation을 권고한다.\n6. 포트폴리오 검정은 중복 fold overlay를 제외하고 stitched OOS equity에서 모델×페르소나별 일수익률을 복원한다.\n\n## 생성 결과\n\nCSV 표 {len(list(out.glob('*.csv')))}개와 그림 {len(list((out/'figures').glob('*.png'))) if (out/'figures').exists() else 0}개를 생성했다.\n"""
    (out/"00_analysis_report.md").write_text(report,encoding="utf-8")
    print(f"분석 완료: {out}")
    print(f"통합 표본: {len(master):,}; parse failures={master.parse_fail.sum():.0f}; repairs={master.repair_used.sum():.0f}")

if __name__=="__main__": main()
