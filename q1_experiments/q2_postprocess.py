# q1_experiments/q2_postprocess.py
# -*- coding: utf-8 -*-
"""Post-processing utilities for the NLPI Q2/Q1 robustness add-on.

This module deliberately does not call an LLM. It converts stored return series
or decision logs into manuscript-ready stress-period and robustness tables.

Supported returns formats:
1) Tidy: date,strategy,return
2) Wide: date,<strategy_1>,<strategy_2>,... where cells are returns
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

STRESS_PERIODS = {
    "COVID_crash_2020": ("2020-02-01", "2020-06-30"),
    "Inflation_rate_shock_2022": ("2022-01-01", "2022-12-31"),
    "High_rate_regime_2022_2023": ("2022-01-01", "2023-12-31"),
    "AI_tech_led_regime_2023_2025": ("2023-01-01", "2025-12-29"),
}


def _to_tidy_returns(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "date" not in df.columns:
        # Try common variants
        for c in ["Date", "timestamp", "time"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    if {"date", "strategy", "return"}.issubset(df.columns):
        out = df[["date", "strategy", "return"]].copy()
    else:
        if "date" not in df.columns:
            raise ValueError("Returns file must contain `date` or be tidy with date,strategy,return.")
        value_cols = [c for c in df.columns if c != "date"]
        out = df.melt(id_vars=["date"], value_vars=value_cols, var_name="strategy", value_name="return")
    out["date"] = pd.to_datetime(out["date"])
    out["return"] = pd.to_numeric(out["return"], errors="coerce").fillna(0.0)
    return out.sort_values(["strategy", "date"])


def _perf(x: pd.Series) -> dict:
    r = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(r) == 0:
        return {"n_days": 0, "sharpe": np.nan, "sortino": np.nan, "cagr": np.nan, "vol": np.nan, "mdd": np.nan, "cum_return": np.nan}
    equity = (1.0 + r).cumprod()
    ann = 252.0
    vol = float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 1 else 0.0
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(ann)) if len(r) > 1 and r.std(ddof=1) > 1e-12 else np.nan
    downside = r.where(r < 0, 0.0)
    sortino = float((r.mean() / downside.std(ddof=1)) * np.sqrt(ann)) if len(r) > 1 and downside.std(ddof=1) > 1e-12 else np.nan
    cagr = float(equity.iloc[-1] ** (ann / len(r)) - 1.0) if equity.iloc[-1] > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    return {"n_days": int(len(r)), "sharpe": sharpe, "sortino": sortino, "cagr": cagr, "vol": vol, "mdd": float(dd.min()), "cum_return": float(equity.iloc[-1] - 1.0)}


def stress_table(returns_path: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir); (outdir / "tables").mkdir(parents=True, exist_ok=True)
    df = _to_tidy_returns(returns_path)
    rows = []
    for name, (s, e) in STRESS_PERIODS.items():
        ss, ee = pd.Timestamp(s), pd.Timestamp(e)
        sub = df[(df["date"] >= ss) & (df["date"] <= ee)]
        for strategy, g in sub.groupby("strategy"):
            rows.append({"stress_period": name, "start": s, "end": e, "strategy": strategy, **_perf(g["return"])})
    out = pd.DataFrame(rows)
    path = outdir / "tables" / "q2_table_stress_period_performance.csv"
    out.to_csv(path, index=False)
    return path


def q2_minimum_manifest(outdir: str | Path) -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"module": "prompt_paraphrase_robustness", "minimum_scope": "P1-P6; 3-10 paraphrases; stratified decision dates", "main_metrics": "fidelity, collapse, missing ticker, latency, projection L1"},
        {"module": "constraint_conflict_ablation", "minimum_scope": "60/70/90 language-level target vs 60 hard cap", "main_metrics": "raw core weight, projected core weight, cap activation, projection L1"},
        {"module": "stress_period_table", "minimum_scope": "COVID 2020; 2022 inflation; 2022-2023 high-rate; 2023-2025 rally", "main_metrics": "Sharpe, CAGR, volatility, MDD, turnover if available"},
        {"module": "cost_rebalance_sensitivity", "minimum_scope": "cost 0/10/25/50bps; rebalance 21/42/63d", "main_metrics": "Sharpe, CAGR, MDD, turnover, rank stability"},
        {"module": "model_family_robustness", "minimum_scope": "base models plus at least one Qwen/Phi/Mistral family; focus P2/P5/P6", "main_metrics": "JSON validity, missing ticker, P2 fidelity, P5 collapse, P6 fidelity, latency"},
        {"module": "hybrid_policy_P6", "minimum_scope": "defensive-momentum policy with qualitative constraints", "main_metrics": "top3 overlap, allocation L1, projection L1, collapse"},
    ]
    path = outdir / "q2_minimum_robustness_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stress")
    s1.add_argument("--returns", required=True, help="CSV in tidy or wide return format")
    s1.add_argument("--outdir", required=True)
    s2 = sub.add_parser("manifest")
    s2.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if args.cmd == "stress":
        print(stress_table(args.returns, args.outdir))
    elif args.cmd == "manifest":
        print(q2_minimum_manifest(args.outdir))

if __name__ == "__main__":
    main()
