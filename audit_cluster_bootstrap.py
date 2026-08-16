#!/usr/bin/env python3
"""Cluster-bootstrap uncertainty for the primary NLPI reliability audit.

No LLM calls are made. The program reads the two retained primary call logs,
forms model-policy-decision-date clusters, and resamples whole clusters with
replacement. It writes machine-readable and manuscript-ready summaries while
leaving all source logs unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MAIN = Path(
    "outputs/NLPI_PAPER_CANONICAL_V1_20260717_152949/"
    "reliability_resumable_20260727/logs/q1_decision_log.csv"
)
DEFAULT_QWEN = Path("outputs/qwen_model_generalization_108/logs/q1_decision_log.csv")
RELEASE_MAIN = Path("results/reliability_primary/q1_decision_log.csv")
RELEASE_QWEN = Path("results/reliability_qwen/q1_decision_log.csv")
CLUSTER_COLUMNS = ["model_id", "policy_id", "decision_date"]
METRICS = ["projected_fidelity", "top3_overlap", "json_valid"]


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = set(CLUSTER_COLUMNS + METRICS + ["experiment_id"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def _available(primary: Path, release: Path) -> Path:
    """Use canonical-tree paths when present and compact-release paths otherwise."""
    return primary if primary.is_file() else release


def _cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    usable = frame.dropna(subset=[metric, *CLUSTER_COLUMNS]).copy()
    grouped = usable.groupby(CLUSTER_COLUMNS, sort=True, observed=True)[metric].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(grouped)
    if n_clusters < 2:
        raise ValueError(f"{metric}: fewer than two clusters")

    draws = np.empty(reps, dtype=float)
    batch = 500
    for start in range(0, reps, batch):
        stop = min(start + batch, reps)
        idx = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        draws[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    estimate = float(usable[metric].mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lower), float(upper), n_clusters


def summarize(frame: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scopes: list[tuple[str, pd.DataFrame]] = [("Primary audit: overall", frame)]
    for experiment, subset in frame.groupby("experiment_id", sort=True):
        scopes.append((str(experiment), subset))

    rows = []
    for scope, subset in scopes:
        for metric in METRICS:
            estimate, lower, upper, clusters = _cluster_bootstrap(subset, metric, reps, rng)
            rows.append({
                "scope": scope,
                "metric": metric,
                "estimate": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "records": int(subset[metric].notna().sum()),
                "clusters": clusters,
                "bootstrap_reps": reps,
                "seed": seed,
                "cluster_definition": "model_id|policy_id|decision_date",
            })
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    fidelity = summary.loc[summary["metric"].eq("projected_fidelity")].copy()
    lines = [
        "# Cluster-bootstrap reliability intervals",
        "",
        "Whole model-policy-decision-date clusters are resampled with replacement.",
        "Retained call records are not treated as independent observations.",
        "",
        "| Scope | Records | Clusters | Fidelity | 95% cluster-bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fidelity.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.records:,} | {row.clusters:,} | "
            f"{row.estimate:.4f} | [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-log", type=Path, default=None)
    parser.add_argument("--qwen-log", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("results/statistical_enhancement"))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.reps < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replications")

    args.main_log = args.main_log or _available(DEFAULT_MAIN, RELEASE_MAIN)
    args.qwen_log = args.qwen_log or _available(DEFAULT_QWEN, RELEASE_QWEN)

    main_frame = _read(args.main_log)
    qwen_frame = _read(args.qwen_log)
    frame = pd.concat([main_frame, qwen_frame], ignore_index=True, sort=False)
    if (len(main_frame), len(qwen_frame), len(frame)) != (5_562, 108, 5_670):
        raise AssertionError(
            f"Unexpected retained counts: {len(main_frame)} + {len(qwen_frame)} = {len(frame)}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = summarize(frame, args.reps, args.seed)
    result.to_csv(args.outdir / "audit_cluster_bootstrap.csv", index=False)
    write_markdown(result, args.outdir / "audit_cluster_bootstrap.md")
    manifest = {
        "source_main": str(args.main_log),
        "source_qwen": str(args.qwen_log),
        "records_main": len(main_frame),
        "records_qwen": len(qwen_frame),
        "records_total": len(frame),
        "cluster_columns": CLUSTER_COLUMNS,
        "metrics": METRICS,
        "bootstrap_reps": args.reps,
        "seed": args.seed,
        "method": "nonparametric percentile bootstrap of whole clusters",
    }
    (args.outdir / "audit_cluster_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    overall = result.loc[
        result["scope"].eq("Primary audit: overall")
        & result["metric"].eq("projected_fidelity")
    ].iloc[0]
    print(
        "CLUSTER_BOOTSTRAP_OK "
        f"records={len(frame)} clusters={int(overall.clusters)} "
        f"fidelity={overall.estimate:.4f} "
        f"ci95=[{overall.ci95_lower:.4f},{overall.ci95_upper:.4f}]"
    )


if __name__ == "__main__":
    main()
