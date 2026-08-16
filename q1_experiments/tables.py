# q1_experiments/tables.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _mean(x):
    return float(pd.to_numeric(x, errors="coerce").mean())


def _std(x):
    return float(pd.to_numeric(x, errors="coerce").std())


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def generate_summary_tables(outdir: str | Path):
    outdir = Path(outdir)
    log_path = outdir / "logs" / "q1_decision_log.csv"
    tables_dir = outdir / "tables"
    _ensure(tables_dir)
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    if df.empty:
        return

    numeric_cols = [
        "projected_fidelity", "raw_fidelity", "top3_overlap", "allocation_l1_to_reference",
        "allocation_l2_to_reference", "projection_l1", "projection_l2", "ew_l1_distance",
        "collapse_flag", "json_valid", "parse_fail", "repair_used", "latency_sec",
        "missing_asset_count", "hallucinated_asset_count", "raw_cap_violation", "raw_longonly_violation",
        "raw_budget_violation", "post_feasible_budget", "post_feasible_cap", "post_feasible_longonly",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def agg(group_cols, subdf):
        cols = [c for c in group_cols if c in subdf.columns]
        if not cols or subdf.empty:
            return pd.DataFrame()
        out = subdf.groupby(cols).agg(
            n_calls=("projected_fidelity", "size"),
            mean_projected_fidelity=("projected_fidelity", "mean"),
            std_projected_fidelity=("projected_fidelity", "std"),
            mean_raw_fidelity=("raw_fidelity", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
            mean_allocation_l1=("allocation_l1_to_reference", "mean"),
            mean_projection_l1=("projection_l1", "mean"),
            collapse_rate=("collapse_flag", "mean"),
            json_valid_rate=("json_valid", "mean"),
            parse_fail_rate=("parse_fail", "mean"),
            repair_rate=("repair_used", "mean"),
            mean_missing_assets=("missing_asset_count", "mean"),
            mean_hallucinated_assets=("hallucinated_asset_count", "mean"),
            median_latency_sec=("latency_sec", "median"),
            mean_latency_sec=("latency_sec", "mean"),
        ).reset_index()
        return out

    tables = {
        "q1_table_prompt_robustness.csv": agg(["model_id", "policy_id"], df[df["experiment_id"] == "prompt_robustness"]),
        "q1_table_ticker_masking.csv": agg(["model_id", "policy_id", "mask_condition"], df[df["experiment_id"] == "ticker_masking"]),
        "q1_table_policy_complexity.csv": agg(["model_id", "complexity_level"], df[df["experiment_id"] == "policy_complexity"]),
        "q1_table_constraint_conflict.csv": agg(["model_id", "conflict_type", "condition_id"], df[df["experiment_id"] == "constraint_conflict_stress"]),
        "q1_table_model_generalization.csv": agg(["model_id", "policy_id"], df[df["experiment_id"] == "model_family_generalization"]),
    }
    for name, t in tables.items():
        if t is not None and not t.empty:
            t.to_csv(tables_dir / name, index=False)

    # Overall safety table for constraint-stress experiments.
    st = df[df["experiment_id"] == "constraint_conflict_stress"].copy()
    if not st.empty:
        safety = st.groupby(["model_id", "conflict_type"]).agg(
            n_calls=("json_valid", "size"),
            raw_cap_violation_rate=("raw_cap_violation", "mean"),
            raw_longonly_violation_rate=("raw_longonly_violation", "mean"),
            mean_budget_violation=("raw_budget_violation", "mean"),
            post_feasible_budget_rate=("post_feasible_budget", "mean"),
            post_feasible_cap_rate=("post_feasible_cap", "mean"),
            post_feasible_longonly_rate=("post_feasible_longonly", "mean"),
            mean_projection_l1=("projection_l1", "mean"),
            json_valid_rate=("json_valid", "mean"),
            repair_rate=("repair_used", "mean"),
            mean_missing_assets=("missing_asset_count", "mean"),
            mean_hallucinated_assets=("hallucinated_asset_count", "mean"),
        ).reset_index()
        safety.to_csv(tables_dir / "q1_table_constraint_safety_summary.csv", index=False)
