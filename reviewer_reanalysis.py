#!/usr/bin/env python3
"""Reviewer-requested reanalysis from retained NLPI CSV logs.

This program performs four analyses without calling Ollama or downloading data:
1. Fidelity decomposition by policy, model, and model-policy, including P4-excluded
   and policy-macro summaries with cluster-bootstrap intervals.
2. Decision-date block-bootstrap sensitivity analysis.
3. Audit of exact-zero feature values supplied to the LLM and their relationship
   to policy targets. Imputed zeros cannot be distinguished from observed zeros
   unless upstream imputation flags are separately available.
4. P5 semantic diagnostics using target fidelity, top-k overlap, allocation error,
   equal-weight distance, and rank correlation to the deterministic reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_MAIN = Path("results/reliability_primary/q1_decision_log.csv")
DEFAULT_QWEN = Path("results/reliability_qwen/q1_decision_log.csv")
DEFAULT_OUT = Path("results/reviewer_reanalysis")
POLICIES = ["P1", "P2", "P3", "P4", "P5", "P6", "L1", "L2", "L3", "L4", "L5", "L6"]
FEATURE_RE = re.compile(
    r"(?m)^-\s*([A-Za-z0-9]+)_(r1m|r3m|r12m|vol3m|mdd):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_logs(main_path: Path, qwen_path: Path) -> pd.DataFrame:
    frames = []
    for source, path in [("primary", main_path), ("qwen", qwen_path)]:
        if not path.is_file():
            raise FileNotFoundError(path)
        d = pd.read_csv(path, low_memory=False)
        d["source_log"] = source
        frames.append(d)
    data = pd.concat(frames, ignore_index=True, sort=False)
    if len(data) != 5670:
        raise AssertionError(f"Expected 5,670 retained calls, found {len(data):,}")
    required = {
        "model_id", "policy_id", "decision_date", "projected_fidelity",
        "projected_weights", "reference_weights", "target_asset", "prompt_text",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["decision_date"] = pd.to_datetime(data["decision_date"], errors="raise")
    data["projected_fidelity"] = pd.to_numeric(data["projected_fidelity"], errors="coerce")
    return data


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def cluster_bootstrap(
    data: pd.DataFrame,
    cluster_cols: list[str],
    metric: str,
    reps: int,
    seed: int,
) -> dict:
    usable = data.dropna(subset=[metric, *cluster_cols]).copy()
    g = usable.groupby(cluster_cols, observed=True, sort=True)[metric].agg(["sum", "count"])
    if len(g) < 2:
        return {"estimate": float(usable[metric].mean()), "ci95_lower": np.nan,
                "ci95_upper": np.nan, "records": len(usable), "clusters": len(g)}
    sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for start in range(0, reps, 500):
        stop = min(start + 500, reps)
        idx = rng.integers(0, len(g), size=(stop - start, len(g)))
        draws[start:stop] = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = percentile_ci(draws)
    return {"estimate": float(usable[metric].mean()), "ci95_lower": lo,
            "ci95_upper": hi, "records": len(usable), "clusters": len(g)}


def fidelity_decomposition(data: pd.DataFrame, out: Path, reps: int, seed: int) -> None:
    def grouped(cols: list[str]) -> pd.DataFrame:
        return (data.groupby(cols, observed=True, dropna=False)
                .agg(records=("projected_fidelity", "size"),
                     faithful=("projected_fidelity", "sum"),
                     projected_fidelity=("projected_fidelity", "mean"),
                     json_valid=("json_valid", "mean"),
                     allocation_l1=("allocation_l1_to_reference", "mean"),
                     top3_overlap=("top3_overlap", "mean"))
                .reset_index())

    grouped(["policy_id"]).to_csv(out / "01_fidelity_by_policy.csv", index=False)
    grouped(["model_id"]).to_csv(out / "02_fidelity_by_model.csv", index=False)
    grouped(["model_id", "policy_id"]).to_csv(out / "03_fidelity_by_model_policy.csv", index=False)

    defined = data[data["policy_id"].isin(["P1", "P2", "P3", "P4", "P5"])]
    no_p4 = data[data["policy_id"].ne("P4")]
    macro_source = grouped(["policy_id"])
    macro_defined = macro_source[macro_source["policy_id"].isin(["P1", "P2", "P3", "P4", "P5"])]
    macro_no_p4 = macro_source[macro_source["policy_id"].isin(["P1", "P2", "P3", "P5"])]
    rows = [
        {"scope": "all retained primary-audit policies", "aggregation": "call-weighted",
         "records": len(data), "fidelity": data.projected_fidelity.mean()},
        {"scope": "all policies excluding P4", "aggregation": "call-weighted",
         "records": len(no_p4), "fidelity": no_p4.projected_fidelity.mean()},
        {"scope": "defined P1-P5", "aggregation": "call-weighted",
         "records": len(defined), "fidelity": defined.projected_fidelity.mean()},
        {"scope": "defined P1-P5", "aggregation": "policy macro-average",
         "records": len(defined), "fidelity": macro_defined.projected_fidelity.mean()},
        {"scope": "defined P1-P3 and P5 (P4 excluded)", "aggregation": "policy macro-average",
         "records": len(data[data.policy_id.isin(["P1", "P2", "P3", "P5"])]),
         "fidelity": macro_no_p4.projected_fidelity.mean()},
    ]
    pd.DataFrame(rows).to_csv(out / "04_fidelity_aggregate_sensitivity.csv", index=False)

    ci_rows = []
    for i, (policy, subset) in enumerate(data.groupby("policy_id", observed=True, sort=True)):
        result = cluster_bootstrap(subset, ["model_id", "decision_date"],
                                   "projected_fidelity", reps, seed + i)
        ci_rows.append({"policy_id": policy, **result,
                        "cluster_definition": "model_id|decision_date",
                        "bootstrap_reps": reps, "seed": seed + i})
    pd.DataFrame(ci_rows).to_csv(out / "05_fidelity_policy_cluster_bootstrap.csv", index=False)


def date_block_bootstrap(data: pd.DataFrame, out: Path, reps: int, seed: int) -> None:
    scopes = [("overall", data), ("P4_excluded", data[data.policy_id.ne("P4")])]
    scopes += [(f"policy_{p}", d) for p, d in data.groupby("policy_id", observed=True, sort=True)]
    rows = []
    for i, (scope, subset) in enumerate(scopes):
        result = cluster_bootstrap(subset, ["decision_date"], "projected_fidelity", reps, seed + 100 + i)
        rows.append({"scope": scope, **result, "cluster_definition": "decision_date",
                     "bootstrap_reps": reps, "seed": seed + 100 + i})
    pd.DataFrame(rows).to_csv(out / "06_decision_date_block_bootstrap.csv", index=False)


def parse_features(text: str) -> dict[tuple[str, str], float]:
    if not isinstance(text, str):
        return {}
    return {(asset, feat): float(value) for asset, feat, value in FEATURE_RE.findall(text)}


def zero_feature_audit(data: pd.DataFrame, out: Path) -> None:
    feature_rows, call_rows = [], []
    relevant = {"P1": ("r12m", "max"), "P2": ("vol3m", "min"), "P3": ("r3m", "min")}
    for row in data.itertuples(index=False):
        features = parse_features(row.prompt_text)
        zero_count = sum(v == 0.0 for v in features.values())
        for (asset, feature), value in features.items():
            if value == 0.0:
                feature_rows.append({"call_id": getattr(row, "call_id", ""),
                                     "decision_date": row.decision_date.date(),
                                     "model_id": row.model_id, "policy_id": row.policy_id,
                                     "asset": asset, "feature": feature, "value": value})
        target_zero = False
        selected_from_zero = False
        relevant_feature = ""
        if row.policy_id in relevant:
            relevant_feature, direction = relevant[row.policy_id]
            vals = {a: v for (a, f), v in features.items() if f == relevant_feature}
            target_zero = bool(row.target_asset in vals and vals[row.target_asset] == 0.0)
            if vals:
                optimum = max(vals.values()) if direction == "max" else min(vals.values())
                selected_from_zero = bool(optimum == 0.0 and target_zero)
        call_rows.append({"call_id": getattr(row, "call_id", ""),
                          "decision_date": row.decision_date.date(),
                          "model_id": row.model_id, "policy_id": row.policy_id,
                          "parsed_feature_count": len(features), "exact_zero_feature_count": zero_count,
                          "has_exact_zero_feature": zero_count > 0,
                          "relevant_feature": relevant_feature,
                          "target_asset": row.target_asset,
                          "target_relevant_feature_is_zero": target_zero,
                          "policy_optimum_selected_from_zero": selected_from_zero})
    zero_detail = pd.DataFrame(feature_rows)
    calls = pd.DataFrame(call_rows)
    zero_detail.to_csv(out / "07_exact_zero_feature_detail.csv", index=False)
    calls.to_csv(out / "08_exact_zero_call_audit.csv", index=False)
    summary = (calls.groupby("policy_id", observed=True)
               .agg(calls=("call_id", "size"),
                    calls_with_any_exact_zero=("has_exact_zero_feature", "sum"),
                    mean_exact_zero_features=("exact_zero_feature_count", "mean"),
                    target_relevant_feature_zero=("target_relevant_feature_is_zero", "sum"),
                    optimum_selected_from_zero=("policy_optimum_selected_from_zero", "sum"))
               .reset_index())
    summary.to_csv(out / "09_exact_zero_feature_summary.csv", index=False)


def parse_weights(value) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    obj = json.loads(value)
    if isinstance(obj, dict) and "weights" in obj and isinstance(obj["weights"], dict):
        obj = obj["weights"]
    return {str(k): float(v) for k, v in obj.items()}


def p5_diagnostics(data: pd.DataFrame, out: Path) -> None:
    rows = []
    p5 = data[data.policy_id.eq("P5")].copy()
    for row in p5.itertuples(index=False):
        projected, reference = parse_weights(row.projected_weights), parse_weights(row.reference_weights)
        assets = sorted(set(projected) | set(reference))
        pvec = np.array([projected.get(a, 0.0) for a in assets], float)
        rvec = np.array([reference.get(a, 0.0) for a in assets], float)
        n = max(len(assets), 1)
        ew = np.full(n, 1.0 / n)
        if n > 1 and np.ptp(pvec) > 1e-12 and np.ptp(rvec) > 1e-12:
            rho = spearmanr(pvec, rvec).statistic
        else:
            rho = np.nan
        top3_p = set(sorted(assets, key=lambda a: projected.get(a, 0.0), reverse=True)[:3])
        top3_r = set(sorted(assets, key=lambda a: reference.get(a, 0.0), reverse=True)[:3])
        rows.append({"call_id": getattr(row, "call_id", ""),
                     "decision_date": row.decision_date.date(), "model_id": row.model_id,
                     "condition_id": getattr(row, "condition_id", ""),
                     "target_asset": row.target_asset,
                     "top_asset_projected": getattr(row, "top_asset_projected", ""),
                     "target_fidelity": float(row.projected_fidelity),
                     "top3_set_overlap_count": len(top3_p & top3_r),
                     "top3_jaccard": len(top3_p & top3_r) / max(len(top3_p | top3_r), 1),
                     "allocation_l1": float(np.abs(pvec - rvec).sum()),
                     "allocation_l2": float(np.linalg.norm(pvec - rvec)),
                     "equal_weight_l1": float(np.abs(pvec - ew).sum()),
                     "spearman_projected_vs_reference": float(rho) if np.isfinite(rho) else np.nan,
                     "asset_count": n})
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "10_p5_semantic_diagnostics_detail.csv", index=False)
    summary = (detail.groupby("model_id", observed=True)
               .agg(calls=("call_id", "size"), target_fidelity=("target_fidelity", "mean"),
                    mean_top3_jaccard=("top3_jaccard", "mean"),
                    mean_allocation_l1=("allocation_l1", "mean"),
                    mean_allocation_l2=("allocation_l2", "mean"),
                    mean_equal_weight_l1=("equal_weight_l1", "mean"),
                    mean_spearman=("spearman_projected_vs_reference", "mean"),
                    median_spearman=("spearman_projected_vs_reference", "median"))
               .reset_index())
    overall = pd.DataFrame([{"model_id": "ALL", "calls": len(detail),
                             "target_fidelity": detail.target_fidelity.mean(),
                             "mean_top3_jaccard": detail.top3_jaccard.mean(),
                             "mean_allocation_l1": detail.allocation_l1.mean(),
                             "mean_allocation_l2": detail.allocation_l2.mean(),
                             "mean_equal_weight_l1": detail.equal_weight_l1.mean(),
                             "mean_spearman": detail.spearman_projected_vs_reference.mean(),
                             "median_spearman": detail.spearman_projected_vs_reference.median()}])
    pd.concat([summary, overall], ignore_index=True).to_csv(out / "11_p5_semantic_diagnostics_summary.csv", index=False)


def write_report(out: Path, data: pd.DataFrame, reps: int, seed: int) -> None:
    agg = pd.read_csv(out / "04_fidelity_aggregate_sensitivity.csv")
    dates = pd.read_csv(out / "06_decision_date_block_bootstrap.csv")
    p5 = pd.read_csv(out / "11_p5_semantic_diagnostics_summary.csv")
    zero = pd.read_csv(out / "09_exact_zero_feature_summary.csv")
    get = lambda scope: agg.loc[agg.scope.eq(scope), "fidelity"].iloc[0]
    overall_date = dates.loc[dates.scope.eq("overall")].iloc[0]
    p5all = p5.loc[p5.model_id.eq("ALL")].iloc[0]
    text = f"""# Reviewer-requested CSV reanalysis

No LLM was called and no market data were downloaded.

## 1. Fidelity aggregation sensitivity

- Call-weighted overall fidelity: {get('all retained primary-audit policies'):.4f}
- Call-weighted fidelity excluding P4: {get('all policies excluding P4'):.4f}
- P1-P5 policy macro-average: {get('defined P1-P5'):.4f}
- P1-P3/P5 policy macro-average excluding P4: {get('defined P1-P3 and P5 (P4 excluded)'):.4f}

## 2. Decision-date block bootstrap

- Estimate: {overall_date.estimate:.4f}
- 95% percentile CI: [{overall_date.ci95_lower:.4f}, {overall_date.ci95_upper:.4f}]
- Unique decision-date blocks: {int(overall_date.clusters)}
- Replications: {reps:,}; seed family starts at {seed + 100}

## 3. Exact-zero feature audit

- Parsed calls: {int(zero.calls.sum()):,}
- Calls containing at least one exact-zero feature: {int(zero.calls_with_any_exact_zero.sum()):,}
- Important limitation: retained prompts identify exact zero values supplied to the LLM, but do not contain an upstream imputation flag. The outputs therefore cannot distinguish an observed economic zero from a residual value filled with zero.

## 4. P5 multidimensional semantic diagnostics

- Calls: {int(p5all.calls):,}
- Exact target fidelity: {p5all.target_fidelity:.4f}
- Mean top-3 Jaccard overlap: {p5all.mean_top3_jaccard:.4f}
- Mean allocation L1 error: {p5all.mean_allocation_l1:.4f}
- Mean projected/reference rank correlation: {p5all.mean_spearman:.4f}
- Mean distance from equal weight: {p5all.mean_equal_weight_l1:.4f}

The exact target measure remains useful, but P5 should be interpreted jointly with ranking, top-k, and allocation-distance diagnostics because it is a composite policy.
"""
    (out / "00_REANALYSIS_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-log", type=Path, default=DEFAULT_MAIN)
    ap.add_argument("--qwen-log", type=Path, default=DEFAULT_QWEN)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-reps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("Use at least 1,000 bootstrap replications")
    data = load_logs(args.main_log, args.qwen_log)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fidelity_decomposition(data, args.outdir, args.bootstrap_reps, args.seed)
    date_block_bootstrap(data, args.outdir, args.bootstrap_reps, args.seed)
    zero_feature_audit(data, args.outdir)
    p5_diagnostics(data, args.outdir)
    write_report(args.outdir, data, args.bootstrap_reps, args.seed)
    manifest = {
        "analysis": "reviewer_requested_csv_reanalysis",
        "records": len(data), "unique_decision_dates": int(data.decision_date.nunique()),
        "main_log": str(args.main_log), "main_log_sha256": sha256(args.main_log),
        "qwen_log": str(args.qwen_log), "qwen_log_sha256": sha256(args.qwen_log),
        "bootstrap_reps": args.bootstrap_reps, "seed": args.seed,
        "llm_calls": 0, "market_downloads": 0,
        "zero_audit_limitation": "Retained prompts have no upstream imputation flag; exact zero cannot be classified as observed versus imputed.",
    }
    (args.outdir / "12_reanalysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"REVIEWER_REANALYSIS_OK records={len(data)} dates={data.decision_date.nunique()} out={args.outdir}")


if __name__ == "__main__":
    main()
