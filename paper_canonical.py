#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical paper-run utilities for the NLPI portfolio study.

This module centralizes the paper protocol, run-directory creation,
source/config hashing, environment capture, and final checksum export.
It deliberately contains no portfolio logic; the existing GUI/engine code
remains the single implementation of the main backtest.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PAPER_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VEA", "VWO",
    "AGG", "BND", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG",
    "GLD", "SLV", "DBC", "VNQ", "UUP", "BIL",
]
PAPER_MAIN_MODELS = ["gemma3:270m", "gemma3:1b", "llama3.1:8b"]
PAPER_ROBUSTNESS_MODELS = ["qwen3.5:4b"]
PAPER_POLICIES = ["P1", "P2", "P3", "P4", "P5"]
PAPER_RELIABILITY_POLICIES = ["P1", "P2", "P3", "P4", "P5", "P6"]
PAPER_BENCHMARKS = ["EQUAL", "RiskParity", "MVP", "MOM6", "SHARPE"]
PAPER_CODED = ["CODED_P1", "CODED_P2", "CODED_P3", "CODED_P4", "CODED_P5"]

PAPER_PROTOCOL: Dict[str, Any] = {
    "protocol_id": "NLPI-PAPER-CANONICAL-V1",
    "start": "2010-01-01",
    "end": "2025-12-29",
    "tickers": PAPER_TICKERS,
    "rebalance_days": 42,
    "transaction_cost": 0.001,
    "max_weight": 0.60,
    "turnover_cap": 0.25,
    "prompt_cap_pct": 60,
    "wfcv": True,
    "holdout": False,
    "execution": "next-period",
    "weight_drift": True,
    "main_models": PAPER_MAIN_MODELS,
    "robustness_models": PAPER_ROBUSTNESS_MODELS,
    "main_policies": PAPER_POLICIES,
    "reliability_policies": PAPER_RELIABILITY_POLICIES,
    "benchmarks": PAPER_BENCHMARKS,
    "coded_references": PAPER_CODED,
    "reliability_experiments": [
        "prompt_robustness",
        "ticker_masking",
        "policy_complexity",
        "constraint_stress",
        "model_generalization",
        "sensitivity_template",
    ],
    "reliability_decision_sample": "stratified",
    "n_per_regime": 6,
    "seed": 42,
    "figure_dpi": 600,
    "figure_formats": ["png", "pdf", "svg"],
    "figure_stage": "deterministic_csv_reconstruction",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_cmd(cmd: List[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        txt = (p.stdout or p.stderr or "").strip()
        return txt
    except Exception as exc:
        return f"unavailable: {exc}"


def git_commit(project_root: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, timeout=5,
        )
        return p.stdout.strip() if p.returncode == 0 else "not-a-git-checkout"
    except Exception:
        return "unavailable"


def ollama_inventory() -> Dict[str, Any]:
    raw = safe_cmd(["ollama", "list"])
    version = safe_cmd(["ollama", "--version"])
    return {"version": version, "list_output": raw}


def source_hashes(project_root: Path) -> Dict[str, str]:
    targets: List[Path] = []
    for name in [
        "gui.py", "core.py", "llm.py", "llm_portfolio.py", "portfolios.py",
        "paper_canonical.py", "requirements.txt",
    ]:
        p = project_root / name
        if p.exists():
            targets.append(p)
    for folder in ["engine", "q1_experiments", "paper_figures", "configs"]:
        base = project_root / folder
        if base.exists():
            targets.extend(sorted(p for p in base.rglob("*") if p.is_file()))
    return {str(p.relative_to(project_root)): sha256_file(p) for p in targets}


def create_run_directory(output_base: str | Path) -> Path:
    base = Path(output_base).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base / f"NLPI_PAPER_CANONICAL_V1_{ts}"
    for d in [
        root,
        root / "config",
        root / "data",
        root / "main",
        root / "reliability",
        root / "logs",
        root / "paper_tables",
        root / "figures",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return root


def write_initial_manifest(
    run_root: Path,
    project_root: Path,
    include_reliability: bool,
    include_qwen: bool,
) -> Path:
    protocol = dict(PAPER_PROTOCOL)
    protocol["include_reliability"] = bool(include_reliability)
    protocol["include_qwen3_5_4b"] = bool(include_qwen)
    protocol_path = run_root / "config" / "paper_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    manifest: Dict[str, Any] = {
        "run_id": run_root.name,
        "protocol_id": PAPER_PROTOCOL["protocol_id"],
        "created_utc": utc_now(),
        "status": "initialized",
        "stage": "initialization",
        "project_root": str(project_root.resolve()),
        "run_root": str(run_root.resolve()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git_commit": git_commit(project_root),
        "ollama": ollama_inventory(),
        "protocol": protocol,
        "source_hashes": source_hashes(project_root),
        "stages": [],
    }
    path = run_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def update_manifest(run_root: Path, **fields: Any) -> None:
    path = run_root / "run_manifest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"run_id": run_root.name, "created_utc": utc_now(), "stages": []}
    stage_event = fields.pop("stage_event", None)
    if stage_event:
        data.setdefault("stages", []).append({"time_utc": utc_now(), **stage_event})
    data.update(fields)
    data["updated_utc"] = utc_now()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_protocol_values(values: Dict[str, Any], include_qwen: bool = True) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    expected = PAPER_PROTOCOL

    def check(name: str, actual: Any, target: Any) -> None:
        if actual != target:
            errors.append(f"{name}: expected {target!r}, found {actual!r}")

    check("start", str(values.get("start")), expected["start"])
    check("end", str(values.get("end")), expected["end"])
    check("rebalance_days", int(values.get("rebalance_days", -1)), expected["rebalance_days"])
    if abs(float(values.get("transaction_cost", -1)) - expected["transaction_cost"]) > 1e-12:
        errors.append("transaction_cost does not match 0.001")
    if abs(float(values.get("max_weight", -1)) - expected["max_weight"]) > 1e-12:
        errors.append("max_weight does not match 0.60")
    if abs(float(values.get("turnover_cap", -1)) - expected["turnover_cap"]) > 1e-12:
        errors.append("turnover_cap does not match 0.25")
    check("wfcv", bool(values.get("wfcv")), True)
    check("holdout", bool(values.get("holdout")), False)

    tickers = list(values.get("tickers") or [])
    if tickers != PAPER_TICKERS:
        errors.append("Ticker universe or order differs from the canonical 22-ETF universe.")

    selected = set(values.get("selected_strategies") or [])
    required = set(PAPER_BENCHMARKS + PAPER_CODED)
    required.update(f"NLPI[{m}|P{p}]" for m in PAPER_MAIN_MODELS for p in range(1, 6))
    missing = sorted(required - selected)
    if missing:
        errors.append("Missing canonical strategies: " + ", ".join(missing))

    installed = set(values.get("installed_models") or [])
    missing_models = [m for m in PAPER_MAIN_MODELS if m not in installed]
    if missing_models:
        errors.append("Missing required Ollama models: " + ", ".join(missing_models))
    if include_qwen and PAPER_ROBUSTNESS_MODELS[0] not in installed:
        errors.append(f"Missing robustness model: {PAPER_ROBUSTNESS_MODELS[0]}")
    if not include_qwen:
        warnings.append("Qwen3.5-4B model-family robustness is disabled.")

    return errors, warnings


def finalize_run(run_root: Path, status: str, reliability_exit_code: int | None = None) -> None:
    # Finalize the manifest first so its checksum describes the final content.
    update_manifest(
        run_root,
        status=status,
        stage="complete" if status == "completed" else "failed",
        completed_utc=utc_now(),
        reliability_exit_code=reliability_exit_code,
        stage_event={"stage": "finalization", "status": status},
    )

    file_index = []
    for p in sorted(run_root.rglob("*")):
        if p.is_file() and p.name not in {"checksums_sha256.txt", "file_index.json"}:
            file_index.append({
                "path": str(p.relative_to(run_root)),
                "bytes": p.stat().st_size,
            })
    (run_root / "file_index.json").write_text(json.dumps(file_index, indent=2), encoding="utf-8")

    checksums: List[str] = []
    for p in sorted(run_root.rglob("*")):
        if p.is_file() and p.name != "checksums_sha256.txt":
            try:
                checksums.append(f"{sha256_file(p)}  {p.relative_to(run_root)}")
            except OSError:
                pass
    (run_root / "checksums_sha256.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")


def write_protocol_file(project_root: Path) -> None:
    cfg = project_root / "configs" / "paper_protocol.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(PAPER_PROTOCOL, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    write_protocol_file(root)
    print(json.dumps(PAPER_PROTOCOL, indent=2))
