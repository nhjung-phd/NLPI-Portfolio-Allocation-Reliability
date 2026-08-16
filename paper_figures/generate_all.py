from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .loaders import (
    FigureDataError, resolve_run_root, load_oos_tidy, load_reliability_log,
    validate_canonical_inputs, sha256_file,
)
from .equity_figures import plot_main_oos_equity, plot_per_model_wfcv
from .reliability_figures import (
    plot_prompt_robustness, plot_ticker_masking, plot_policy_complexity,
    plot_constraint_conflict, plot_model_family_generalization,
    plot_fidelity_projection_heatmap, plot_latency_reliability,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_all_figures(
    run_root: str | Path,
    *,
    dpi: int = 600,
    overwrite: bool = True,
    strict: bool = True,
    include_reliability: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate all manuscript figures from canonical CSV outputs.

    The function never reads GUI canvas objects. Every figure is reconstructed
    deterministically from canonical CSV files and saved as PNG/PDF/SVG.
    """
    log = log_fn or (lambda msg: print(msg, flush=True))
    root = resolve_run_root(run_root)
    figures_root = root / "figures"
    if overwrite and figures_root.exists():
        shutil.rmtree(figures_root)
    for sub in ["png", "pdf", "svg", "source_data"]:
        (figures_root / sub).mkdir(parents=True, exist_ok=True)

    report = validate_canonical_inputs(root, strict=strict)
    report.to_csv(figures_root / "figure_validation.csv", index=False)
    (figures_root / "figure_validation.json").write_text(
        report.to_json(orient="records", indent=2), encoding="utf-8"
    )
    log("[FIGURES] Canonical input validation complete.")

    oos = load_oos_tidy(root)
    oos.to_csv(figures_root / "source_data" / "oos_tidy_used.csv", index=False)
    generated: dict[str, dict[str, str]] = {}
    skipped: dict[str, str] = {}

    jobs = [
        ("fig_01_main_oos_equity", lambda: plot_main_oos_equity(oos, figures_root, dpi=dpi)),
        ("fig_02_per_model_wfcv", lambda: plot_per_model_wfcv(oos, figures_root, dpi=dpi)),
    ]

    reliability = pd.DataFrame()
    if include_reliability:
        try:
            reliability = load_reliability_log(root, required=strict)
            if not reliability.empty:
                reliability.to_csv(figures_root / "source_data" / "q1_decision_log_used.csv", index=False)
                jobs.extend([
                    ("fig_03_prompt_robustness", lambda: plot_prompt_robustness(reliability, figures_root, dpi=dpi)),
                    ("fig_04_ticker_masking", lambda: plot_ticker_masking(reliability, figures_root, dpi=dpi)),
                    ("fig_05_policy_complexity_ladder", lambda: plot_policy_complexity(reliability, figures_root, dpi=dpi)),
                    ("fig_06_constraint_conflict", lambda: plot_constraint_conflict(reliability, figures_root, dpi=dpi)),
                    ("fig_07_model_family_generalization", lambda: plot_model_family_generalization(reliability, figures_root, dpi=dpi)),
                    ("fig_08_fidelity_projection_heatmap", lambda: plot_fidelity_projection_heatmap(reliability, figures_root, dpi=dpi)),
                    ("fig_09_latency_reliability", lambda: plot_latency_reliability(reliability, figures_root, dpi=dpi)),
                ])
        except Exception as exc:
            if strict:
                raise
            skipped["reliability_figures"] = str(exc)
            log(f"[FIGURES][WARN] Reliability figures skipped: {exc}")

    for stem, fn in jobs:
        try:
            log(f"[FIGURES] Generating {stem} ...")
            generated[stem] = fn()
        except Exception as exc:
            skipped[stem] = str(exc)
            log(f"[FIGURES][WARN] {stem} skipped: {exc}")
            if strict:
                raise FigureDataError(f"Figure generation failed for {stem}: {exc}") from exc

    manifest = {
        "created_utc": utc_now(),
        "run_root": str(root),
        "run_id": root.name,
        "dpi": int(dpi),
        "formats": ["png", "pdf", "svg"],
        "strict_validation": bool(strict),
        "include_reliability": bool(include_reliability),
        "generated": generated,
        "skipped": skipped,
        "source_files": {},
    }
    for p in [root / "main" / "oos_tidy.csv", root / "main" / "performance_main.csv",
              root / "reliability" / "logs" / "q1_decision_log.csv", root / "data" / "adjusted_close.csv"]:
        if p.exists():
            manifest["source_files"][str(p.relative_to(root))] = sha256_file(p)
    manifest_path = figures_root / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"[FIGURES] Completed. Output: {figures_root}")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate publication-quality NLPI paper figures from a canonical run")
    p.add_argument("--run-root", required=True)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--non-strict", action="store_true", help="Generate available figures and report missing inputs as warnings")
    p.add_argument("--main-only", action="store_true", help="Generate only equity/WFCV figures")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generate_all_figures(
        args.run_root,
        dpi=args.dpi,
        overwrite=args.overwrite,
        strict=not args.non_strict,
        include_reliability=not args.main_only,
    )


if __name__ == "__main__":
    main()
