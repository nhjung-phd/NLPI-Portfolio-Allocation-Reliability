from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

MAIN_MODELS = ["gemma3:270m", "gemma3:1b", "llama3.1:8b"]
MAIN_POLICIES = ["P1", "P2", "P3", "P4", "P5"]
MAIN_BENCHMARKS = ["EQUAL", "RiskParity", "MVP", "MOM6", "SHARPE"]


class FigureDataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_run_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    candidates = [p] + list(p.parents)
    for c in candidates:
        if (c / "run_manifest.json").exists() and (c / "main").exists():
            return c
    # Permit figure-only generation on a canonical-looking directory before finalization.
    if (p / "main").exists():
        return p
    raise FigureDataError(f"Could not locate canonical run root from: {path}")


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FigureDataError(f"Required CSV is missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise FigureDataError(f"Could not read {path}: {exc}") from exc


def load_oos_tidy(run_root: str | Path) -> pd.DataFrame:
    root = resolve_run_root(run_root)
    df = _read_csv(root / "main" / "oos_tidy.csv")
    required = {"date", "strategy", "model", "persona", "fold", "segment", "equity"}
    missing = required - set(df.columns)
    if missing:
        raise FigureDataError("oos_tidy.csv missing columns: " + ", ".join(sorted(missing)))
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").fillna(0).astype(int)
    for c in ["strategy", "model", "persona", "segment"]:
        df[c] = df[c].fillna("NA").astype(str)
    df = df.dropna(subset=["date", "equity"])
    df["series_key"] = df.apply(series_key_from_row, axis=1)
    return df.sort_values(["segment", "series_key", "fold", "date"]).reset_index(drop=True)


def series_key_from_row(row: pd.Series) -> str:
    strategy = str(row.get("strategy", ""))
    model = str(row.get("model", "NA"))
    persona = str(row.get("persona", "NA"))
    if strategy == "NLPI" and model not in {"", "NA", "nan"} and persona not in {"", "NA", "nan"}:
        return f"NLPI[{model}|{persona}]"
    return strategy


def load_performance(run_root: str | Path) -> pd.DataFrame:
    root = resolve_run_root(run_root)
    return _read_csv(root / "main" / "performance_main.csv", required=False)


def load_reliability_log(run_root: str | Path, required: bool = True) -> pd.DataFrame:
    root = resolve_run_root(run_root)
    df = _read_csv(root / "reliability" / "logs" / "q1_decision_log.csv", required=required)
    if df.empty:
        return df
    for c in [
        "projected_fidelity", "raw_fidelity", "allocation_l1_to_reference", "projection_l1",
        "collapse_flag", "json_valid", "parse_fail", "repair_used", "latency_sec",
        "missing_asset_count", "hallucinated_asset_count", "raw_max_weight",
        "post_max_weight", "post_feasible_cap", "post_feasible_budget", "post_feasible_longonly",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_json(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FigureDataError(f"Required JSON is missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FigureDataError(f"Could not read JSON {path}: {exc}") from exc


def _performance_key_column(df: pd.DataFrame) -> str | None:
    for c in ["Strategy", "name", "strategy", "strategy_key"]:
        if c in df.columns:
            return c
    return None


def _performance_cum_column(df: pd.DataFrame) -> str | None:
    for c in ["CUM", "Cumulative Return", "cumulative_return", "CumReturn"]:
        if c in df.columns:
            return c
    return None


def validate_canonical_inputs(run_root: str | Path, strict: bool = True) -> pd.DataFrame:
    root = resolve_run_root(run_root)
    rows: list[dict[str, str]] = []

    def add(check: str, status: str, detail: str, severity: str = "error") -> None:
        rows.append({"check": check, "status": status, "severity": severity, "detail": detail})

    oos = load_oos_tidy(root)
    stitched = oos[oos["segment"] == "stitched_oos"].copy()

    # 1. Required model/profile series.
    keys = set(stitched["series_key"].unique())
    missing = [f"NLPI[{m}|{p}]" for m in MAIN_MODELS for p in MAIN_POLICIES if f"NLPI[{m}|{p}]" not in keys]
    add("required_model_profile_series", "fail" if missing else "pass",
        "Missing: " + ", ".join(missing) if missing else "All main models contain P1-P5 stitched OOS series.")

    # 2. Common date support.
    ranges = stitched.groupby("series_key")["date"].agg(["min", "max", "count"])
    common = len(ranges[["min", "max"]].drop_duplicates()) == 1 if not ranges.empty else False
    add("common_stitched_date_range", "pass" if common else "fail",
        f"Unique date ranges: {len(ranges[['min','max']].drop_duplicates())}; series: {len(ranges)}")

    # 3. Duplicated observations.
    dup_cols = ["date", "series_key", "segment", "fold"]
    ndups = int(oos.duplicated(dup_cols, keep=False).sum())
    add("duplicate_strategy_date_rows", "pass" if ndups == 0 else "fail", f"Duplicate rows: {ndups}")

    # 4. Per-model filtering sanity: NLPI P1-P5 must not have been replaced by benchmark curves.
    suspicious = []
    for model in MAIN_MODELS:
        sub = stitched[(stitched["strategy"] == "NLPI") & (stitched["model"] == model)]
        profiles = sorted(sub["persona"].unique())
        hashes = []
        for p, g in sub.groupby("persona"):
            s = g.sort_values("date")["equity"].round(10).to_numpy()
            hashes.append(hashlib.sha256(s.tobytes()).hexdigest())
        if set(MAIN_POLICIES) - set(profiles) or len(set(hashes)) < 2:
            suspicious.append(f"{model}: profiles={profiles}, unique_curves={len(set(hashes))}")
    add("per_model_filtering_sanity", "fail" if suspicious else "pass",
        "; ".join(suspicious) if suspicious else "Each model is filtered to its own NLPI P1-P5 curves; no benchmark substitution detected.")

    # 5. Figure endpoints versus performance cumulative returns.
    perf = load_performance(root)
    key_col, cum_col = _performance_key_column(perf), _performance_cum_column(perf)
    if perf.empty or not key_col or not cum_col:
        add("endpoint_performance_consistency", "warn", "Performance key/CUM columns unavailable; check skipped.", "warning")
    else:
        pmap = pd.Series(pd.to_numeric(perf[cum_col], errors="coerce").values, index=perf[key_col].astype(str)).to_dict()
        mismatches = []
        for key, g in stitched.groupby("series_key"):
            if key not in pmap or pd.isna(pmap[key]):
                continue
            endpoint = float(g.sort_values("date")["equity"].iloc[-1]) - 1.0
            if abs(endpoint - float(pmap[key])) > 2e-3:
                mismatches.append(f"{key}: endpoint={endpoint:.6f}, table={float(pmap[key]):.6f}")
        add("endpoint_performance_consistency", "fail" if mismatches else "pass",
            "; ".join(mismatches[:12]) if mismatches else "Stitched equity endpoints match performance cumulative returns within tolerance.")

    # 6. Canonical run ID and data hash.
    run_manifest = load_json(root / "run_manifest.json", required=False)
    data_manifest = load_json(root / "data" / "adjusted_close.manifest.json", required=False)
    run_ok = (not run_manifest) or str(run_manifest.get("run_id", root.name)) == root.name
    add("canonical_run_id", "pass" if run_ok else "fail",
        f"folder={root.name}, manifest={run_manifest.get('run_id', 'missing')}")
    data_file = root / "data" / "adjusted_close.csv"
    if data_file.exists() and data_manifest.get("sha256"):
        actual = sha256_file(data_file)
        expected = str(data_manifest["sha256"])
        add("data_snapshot_sha256", "pass" if actual == expected else "fail",
            f"expected={expected}, actual={actual}")
    else:
        add("data_snapshot_sha256", "warn", "Data snapshot or its SHA-256 manifest is unavailable.", "warning")

    report = pd.DataFrame(rows)
    if strict and (report["status"] == "fail").any():
        failed = report[report["status"] == "fail"]
        raise FigureDataError("Canonical figure validation failed:\n" + "\n".join(f"- {r.check}: {r.detail}" for r in failed.itertuples()))
    return report


def extract_cap_target(value: Any) -> float | None:
    txt = str(value)
    m = re.search(r"(?:aligned|conflict)[_-]?(60|70|90)|\b(60|70|90)\b", txt)
    if not m:
        return None
    token = next((x for x in m.groups() if x), None)
    return float(token) / 100.0 if token else None
