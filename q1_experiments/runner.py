# q1_experiments/runner.py
# -*- coding: utf-8 -*-
"""Q1 NLPI reliability and safety experiment runner.

This runner deliberately reuses the existing project modules:
- core.fetch_prices_yf / core.make_features
- llm.call_llm and its parser/metadata
- engine.strategies.project_capped_simplex

It adds experiment-level orchestration, prompt variants, ticker masking,
policy-complexity references, constraint-conflict prompts, and a unified
JSONL/CSV logging schema.
"""
from __future__ import annotations

import argparse, atexit, hashlib, json, os, sqlite3, sys, time
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make the project root importable when running `python -m q1_experiments.runner`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import fetch_prices_yf, fetch_prices_yf_with_audit, make_features, price_audit_table
from engine.strategies import project_capped_simplex
import llm
from llm import call_llm
from portfolios import DEFAULT_TICKERS

from q1_experiments.prompt_library import (
    BASE_POLICY_PROMPTS,
    PROMPT_ROBUSTNESS_PARAPHRASES,
    POLICY_COMPLEXITY_PROMPTS,
    CONSTRAINT_CONFLICT_PROMPTS,
)
from q1_experiments.reference_policies import reference_weights, topk_overlap


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_text(x: str) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:16]


def _json_dumps(x) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_id(parts: dict) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class CheckpointStore:
    """Crash-safe, call-level checkpoint database."""

    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                decision_date TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                elapsed_sec REAL,
                error_type TEXT,
                error_message TEXT,
                result_json TEXT
            )
            """
        )
        self.conn.commit()

    def completed(self, call_id: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM calls WHERE call_id=?", (call_id,)
        ).fetchone()
        return bool(row and row[0] == "completed")

    def start(self, call_id: str, key: dict, attempt: int) -> None:
        self.conn.execute(
            """
            INSERT INTO calls (
                call_id, experiment_id, model_id, condition_id, policy_id,
                decision_date, status, attempt, started_at, completed_at,
                elapsed_sec, error_type, error_message, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT(call_id) DO UPDATE SET
                status='running', attempt=excluded.attempt,
                started_at=excluded.started_at, completed_at=NULL,
                elapsed_sec=NULL, error_type=NULL, error_message=NULL
            """,
            (
                call_id, key["experiment_id"], key["model_id"],
                key["condition_id"], key["policy_id"], key["decision_date"],
                attempt, _utc_now(),
            ),
        )
        self.conn.commit()

    def complete(self, call_id: str, result: dict, elapsed: float) -> None:
        self.conn.execute(
            """
            UPDATE calls SET status='completed', completed_at=?, elapsed_sec=?,
                error_type=NULL, error_message=NULL, result_json=?
            WHERE call_id=?
            """,
            (_utc_now(), float(elapsed), json.dumps(result, ensure_ascii=False), call_id),
        )
        self.conn.commit()

    def fail(self, call_id: str, exc: Exception, elapsed: float) -> None:
        status = "timed_out" if "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower() else "failed"
        failure_meta = getattr(exc, "meta", None)
        failure_json = (
            json.dumps({"failure_diagnostics": failure_meta}, ensure_ascii=False)
            if failure_meta else None
        )
        self.conn.execute(
            """
            UPDATE calls SET status=?, completed_at=?, elapsed_sec=?,
                error_type=?, error_message=?, result_json=? WHERE call_id=?
            """,
            (
                status, _utc_now(), float(elapsed), type(exc).__name__,
                str(exc)[:2000], failure_json, call_id,
            ),
        )
        self.conn.commit()

    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM calls GROUP BY status"
        ).fetchall()
        return {str(k): int(v) for k, v in rows}

    def completed_results(self) -> List[dict]:
        rows = self.conn.execute(
            """
            SELECT result_json FROM calls
            WHERE status='completed' AND result_json IS NOT NULL
            ORDER BY completed_at, call_id
            """
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        self.conn.close()


@contextmanager
def output_lock(outdir: Path):
    lock_path = outdir / ".runner.lock"
    fp = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        raise SystemExit(
            f"[ERROR] Another runner is already using this output directory: {outdir}"
        )
    fp.seek(0)
    fp.truncate()
    fp.write(f"pid={os.getpid()}\nstarted_utc={_utc_now()}\n")
    fp.flush()
    try:
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


def feature_block(row: pd.Series, tickers: List[str], label_map: Optional[Dict[str, str]] = None) -> str:
    """Render feature values for prompt insertion.

    label_map maps real ticker -> prompt label. For ticker masking, feature keys
    shown to the LLM are also masked, so the LLM cannot rely on real ticker names.
    """
    label_map = label_map or {a: a for a in tickers}
    lines = []
    for real in tickers:
        label = label_map.get(real, real)
        for suffix in ("r1m", "r3m", "r12m", "vol3m", "mdd"):
            key = f"{real}_{suffix}"
            v = row.get(key, 0.0)
            try:
                val = float(v)
            except Exception:
                val = 0.0
            lines.append(f"- {label}_{suffix}: {val:.8f}")
    try:
        lines.append(f"- cross_vol: {float(row.get('cross_vol', 0.0)):.8f}")
    except Exception:
        lines.append("- cross_vol: 0.00000000")
    return "\n".join(lines)


def build_eval_prompt(policy_text: str, row: pd.Series, tickers: List[str], cfg: dict, label_map: Optional[Dict[str, str]] = None, constraint_reminder: bool = True) -> str:
    label_map = label_map or {a: a for a in tickers}
    labels = [label_map.get(a, a) for a in tickers]
    txt = policy_text.format(
        prompt_cap_pct=float(cfg.get("prompt_cap_pct", 60.0)),
        maxw_pct=float(cfg.get("max_weight", 0.60)) * 100.0,
        max_weight=float(cfg.get("max_weight", 0.60)),
        turnover_cap=float(cfg.get("turnover_cap", 0.25)),
    )
    constraints = ""
    if constraint_reminder:
        constraints = (
            "\nHard constraints for the execution layer: long-only weights, weights sum to one, "
            f"maximum single-asset weight {float(cfg.get('max_weight', 0.60)):.2f}, "
            f"and turnover cap {float(cfg.get('turnover_cap', 0.25)):.2f}. "
            "You propose weights; deterministic projection will enforce final feasibility."
        )
    return (
        f"{txt}{constraints}\n\n"
        f"Current Features:\n{feature_block(row, tickers, label_map=label_map)}\n\n"
        f"Assets: {', '.join(labels)}"
    )


def decision_indices(features: pd.DataFrame, rebalance_days: int = 42, mode: str = "stratified", n_per_regime: int = 10, seed: int = 42, max_dates: Optional[int] = None) -> List[int]:
    start_i = min(252, max(0, len(features) - 1))
    base = list(range(start_i, len(features), max(int(rebalance_days), 1)))
    if max_dates is not None and max_dates > 0:
        base = base[:int(max_dates)]
    if mode == "full":
        return base
    if mode == "first":
        return base[: max(1, int(n_per_regime))]
    if mode != "stratified" or "cross_vol" not in features.columns or len(base) == 0:
        return base[: min(len(base), max(1, int(n_per_regime) * 3))]

    rng = np.random.default_rng(int(seed))
    xs = features.iloc[base]["cross_vol"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q1, q2 = float(xs.quantile(1/3)), float(xs.quantile(2/3))
    groups = {
        "low": [i for i in base if float(features.iloc[i].get("cross_vol", 0.0)) <= q1],
        "mid": [i for i in base if q1 < float(features.iloc[i].get("cross_vol", 0.0)) <= q2],
        "high": [i for i in base if float(features.iloc[i].get("cross_vol", 0.0)) > q2],
    }
    selected = []
    for g in ("low", "mid", "high"):
        vals = groups[g]
        if len(vals) > int(n_per_regime):
            vals = list(rng.choice(vals, size=int(n_per_regime), replace=False))
        selected.extend(vals)
    return sorted(set(int(i) for i in selected))


def real_to_masked_map(tickers: List[str], condition: str, seed: int = 42) -> Dict[str, str]:
    if condition == "real_ticker":
        return {a: a for a in tickers}
    labels = [f"Asset_{i+1:02d}" for i in range(len(tickers))]
    if condition == "shuffled_masked":
        rng = np.random.default_rng(seed)
        labels = list(rng.permutation(labels))
    if condition == "symbolic_masked":
        labels = [f"X{i*7+3:02d}" for i in range(len(tickers))]
    return {a: lab for a, lab in zip(tickers, labels)}


def _invert(d: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in d.items()}


def _raw_weights_from_call(prompt: str, model: str, cfg: dict, labels: List[str], dry_run: bool, ref_w: Optional[pd.Series] = None) -> Tuple[pd.Series, dict]:
    if dry_run:
        # Dry-run is for schema/testing only. It does not claim LLM evidence.
        if ref_w is None or len(ref_w) == 0:
            w = pd.Series(1.0 / len(labels), index=labels)
        else:
            w = ref_w.copy()
            w.index = labels[:len(w)] if len(labels) == len(w) else w.index
        meta = {
            "json_valid": 1, "parse_failed": 0, "repair_used": 0,
            "elapsed_sec": 0.0, "dry_run": 1, "raw_response": "DRY_RUN_REFERENCE",
            "missing_asset_count": 0, "hallucinated_ticker_count": 0,
            "invalid_weight_count": 0, "negative_weight_count": 0,
        }
        return w.reindex(labels).fillna(0.0), meta

    out = call_llm(
        prompt,
        cfg.get("ollama_url", "http://localhost:11434"),
        model,
        log_fn=lambda msg: print(msg, flush=True),
        timeout=(
            float(cfg.get("ollama_connect_timeout", 30.0)),
            float(cfg.get("ollama_read_timeout", 900.0)),
        ),
        max_retries=int(cfg.get("max_retries", 2)),
        parse_retries=int(cfg.get("parse_retries", 1)),
        timeout_retries=int(cfg.get("timeout_retries", 1)),
        num_predict=int(cfg.get("num_predict", 512)),
        seed=int(cfg.get("seed", 42)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 0.9)),
        keep_alive=str(cfg.get("ollama_keep_alive", "30m")),
    )
    meta = dict(getattr(llm, "LAST_CALL_META", {}) or {})
    raw = None
    if out is None:
        raise RuntimeError(
            f"Ollama call failed after {int(cfg.get('max_retries', 2)) + 1} attempts "
            f"(model={model})"
        )
    elif hasattr(out, "weights"):
        raw = getattr(out, "weights")
    elif isinstance(out, dict):
        raw = out.get("weights", out)
    else:
        raw = {}
    s = pd.Series(raw, dtype=float) if isinstance(raw, dict) else pd.Series(dtype=float)
    return s.reindex(labels).fillna(0.0).astype(float), meta


def _feasibility_metrics(raw_w: pd.Series, proj_w: pd.Series, maxw: float) -> dict:
    raw_sum = float(raw_w.sum()) if len(raw_w) else 0.0
    raw_max = float(raw_w.max()) if len(raw_w) else 0.0
    raw_min = float(raw_w.min()) if len(raw_w) else 0.0
    return {
        "raw_sum": raw_sum,
        "raw_max_weight": raw_max,
        "raw_min_weight": raw_min,
        "raw_budget_violation": float(abs(raw_sum - 1.0)),
        "raw_cap_violation": int(raw_max > float(maxw) + 1e-8),
        "raw_longonly_violation": int(raw_min < -1e-12),
        "post_sum": float(proj_w.sum()) if len(proj_w) else 0.0,
        "post_max_weight": float(proj_w.max()) if len(proj_w) else 0.0,
        "post_min_weight": float(proj_w.min()) if len(proj_w) else 0.0,
        "post_feasible_budget": int(abs(float(proj_w.sum()) - 1.0) <= 1e-6),
        "post_feasible_cap": int(float(proj_w.max()) <= float(maxw) + 1e-8),
        "post_feasible_longonly": int(float(proj_w.min()) >= -1e-10),
    }


def evaluate_one_call(
    *, experiment_id: str, condition_id: str, model: str, policy_id: str, prompt_text: str,
    row: pd.Series, tickers: List[str], cfg: dict, date, label_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False, paraphrase_id: Optional[str] = None, conflict_type: Optional[str] = None,
    complexity_level: Optional[str] = None, mask_condition: str = "real_ticker", constraint_reminder: bool = True,
) -> dict:
    label_map = label_map or {a: a for a in tickers}
    inv_map = _invert(label_map)
    labels = [label_map[a] for a in tickers]
    ref_real = reference_weights(policy_id if policy_id.startswith("L") else policy_id, row, tickers, prompt_cap=float(cfg.get("prompt_cap_pct", 60.0)) / 100.0)
    ref_label = ref_real.copy()
    ref_label.index = [label_map.get(a, a) for a in ref_real.index]
    prompt = build_eval_prompt(prompt_text, row, tickers, cfg, label_map=label_map, constraint_reminder=constraint_reminder)
    raw_label, meta = _raw_weights_from_call(prompt, model, cfg, labels, dry_run=dry_run, ref_w=ref_label)
    raw_real = pd.Series({inv_map.get(k, k): v for k, v in raw_label.items()}, dtype=float).reindex(tickers).fillna(0.0)
    proj_real = project_capped_simplex(raw_real, float(cfg.get("max_weight", 0.60)))
    ref_proj = project_capped_simplex(ref_real, float(cfg.get("max_weight", 0.60)))

    raw_top = str(raw_real.idxmax()) if len(raw_real) else ""
    proj_top = str(proj_real.idxmax()) if len(proj_real) else ""
    ref_top = str(ref_proj.idxmax()) if len(ref_proj) else ""
    eq = pd.Series(1.0 / len(tickers), index=tickers)
    allocation_l1 = float((proj_real - ref_proj).abs().sum())
    allocation_l2 = float(np.sqrt(((proj_real - ref_proj) ** 2).sum()))
    projection_l1 = float((raw_real - proj_real).abs().sum())
    projection_l2 = float(np.sqrt(((raw_real - proj_real) ** 2).sum()))
    ew_l1 = float((proj_real - eq).abs().sum())

    row_out = {
        "run_id": cfg.get("run_id", "q1"),
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "model_id": model,
        "policy_id": policy_id,
        "complexity_level": complexity_level or "",
        "paraphrase_id": paraphrase_id or "",
        "mask_condition": mask_condition,
        "conflict_type": conflict_type or "",
        "decision_date": str(pd.Timestamp(date).date()) if date is not None else "",
        "rebalance_days": int(cfg.get("rebalance_days", 42)),
        "cost_bps": float(cfg.get("tcost", 0.001)) * 10000.0,
        "wmax": float(cfg.get("max_weight", 0.60)),
        "tau": float(cfg.get("turnover_cap", 0.25)),
        "prompt_hash": _hash_text(prompt),
        "prompt_text": prompt,
        "raw_output": str(meta.get("raw_response", ""))[:4000],
        "json_valid": int(meta.get("json_valid", 0) or 0),
        "parse_fail": int(meta.get("parse_failed", 0) or 0),
        "repair_used": int(meta.get("repair_used", 0) or 0),
        "latency_sec": float(meta.get("elapsed_sec", np.nan) or 0.0),
        "output_length": int(meta.get("output_length", 0) or 0),
        "attempt_count": int(meta.get("attempt_count", 1) or 1),
        "parse_retry_count": int(meta.get("parse_retry_count", 0) or 0),
        "network_retry_count": int(meta.get("network_retry_count", 0) or 0),
        "timeout_retry_count": int(meta.get("timeout_retry_count", 0) or 0),
        "eval_count": int(meta.get("eval_count", 0) or 0),
        "eval_duration_sec": float(meta.get("eval_duration_sec", 0.0) or 0.0),
        "prompt_eval_count": int(meta.get("prompt_eval_count", 0) or 0),
        "done_reason": str(meta.get("done_reason", "") or ""),
        "attempt_diagnostics": json.dumps(meta.get("attempt_diagnostics", []), ensure_ascii=False),
        "missing_asset_count": int(meta.get("missing_asset_count", 0) or 0),
        "hallucinated_asset_count": int(meta.get("hallucinated_ticker_count", 0) or 0),
        "negative_weight_count": int(meta.get("negative_weight_count", 0) or 0),
        "invalid_weight_count": int(meta.get("invalid_weight_count", 0) or 0),
        "target_asset": ref_top,
        "top_asset_raw": raw_top,
        "top_asset_projected": proj_top,
        "raw_fidelity": int(raw_top == ref_top),
        "projected_fidelity": int(proj_top == ref_top),
        "top3_overlap": topk_overlap(proj_real, ref_proj, k=3),
        "allocation_l1_to_reference": allocation_l1,
        "allocation_l2_to_reference": allocation_l2,
        "projection_l1": projection_l1,
        "projection_l2": projection_l2,
        "ew_l1_distance": ew_l1,
        "collapse_flag": int(ew_l1 <= float(cfg.get("collapse_l1_threshold", 1e-3))),
        "raw_weights": raw_real.to_json(),
        "projected_weights": proj_real.to_json(),
        "reference_weights": ref_proj.to_json(),
        "dry_run": int(dry_run),
    }
    row_out.update(_feasibility_metrics(raw_real, proj_real, float(cfg.get("max_weight", 0.60))))
    return row_out


def _make_synthetic_prices(tickers: List[str], start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Small deterministic synthetic panel for dry-run/schema tests only."""
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) < 320:
        idx = pd.bdate_range(start="2020-01-01", periods=420)
    rng = np.random.default_rng(int(seed))
    cols = list(tickers)
    data = {}
    for j, t in enumerate(cols):
        mu = 0.00005 + 0.00001 * (j % 5)
        sig = 0.006 + 0.001 * (j % 7)
        r = rng.normal(mu, sig, size=len(idx))
        data[t] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(data, index=idx)


def load_data(args) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    tickers = list(args.tickers or DEFAULT_TICKERS)
    if bool(getattr(args, "synthetic_data", False)):
        prices = _make_synthetic_prices(tickers, args.start, args.end, seed=args.seed)
    else:
        prices = fetch_prices_yf(tickers, args.start, args.end)
    if prices is None or prices.empty:
        raise RuntimeError("Price fetch returned no rows. Check internet/yfinance and date range.")
    tickers = [t for t in tickers if t in prices.columns]
    features = make_features(prices, tickers)
    return prices, features, tickers


def write_manifest(outdir: Path, args, tickers: List[str], decision_idx: List[int], features: pd.DataFrame) -> dict:
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "start": args.start,
        "end": args.end,
        "tickers": tickers,
        "n_features_rows": int(len(features)),
        "n_decision_dates": int(len(decision_idx)),
        "decision_sample_mode": args.decision_sample,
        "models": args.models,
        "experiments": args.experiments,
        "rebalance_days": args.rebalance,
        "tcost": args.tcost,
        "max_weight": args.maxw,
        "turnover_cap": args.turncap,
        "prompt_cap_pct": args.prompt_cap,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    (outdir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _append_rows(rows: List[dict], out_jsonl: Path, out_csv: Path) -> None:
    if not rows:
        return
    with out_jsonl.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Keep CSV updated after each module for crash recovery.
    df = pd.DataFrame(rows)
    if out_csv.exists():
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)


def _rebuild_exports_from_checkpoint(
    store: CheckpointStore, out_jsonl: Path, out_csv: Path
) -> None:
    """Make JSONL/CSV exact mirrors of completed checkpoint rows after a crash."""
    rows = store.completed_results()
    if not rows:
        return
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _execute_one(
    *, store: CheckpointStore, out_jsonl: Path, out_csv: Path,
    total: int, ordinal: int, args, call_kwargs: dict,
) -> str:
    date_value = call_kwargs.get("date")
    key = {
        "experiment_id": str(call_kwargs["experiment_id"]),
        "model_id": str(call_kwargs["model"]),
        "condition_id": str(call_kwargs["condition_id"]),
        "policy_id": str(call_kwargs["policy_id"]),
        "decision_date": str(pd.Timestamp(date_value).date()) if date_value is not None else "",
        "prompt_hash": _hash_text(call_kwargs["prompt_text"]),
        "seed": int(args.seed),
    }
    cid = _call_id(key)
    if store.completed(cid):
        print(
            f"[RESUME SKIP] {ordinal}/{total} id={cid} "
            f"model={key['model_id']} condition={key['condition_id']} "
            f"date={key['decision_date']}",
            flush=True,
        )
        return "skipped"

    attempt = 1
    store.start(cid, key, attempt)
    started = time.perf_counter()
    print(
        f"[CALL START] {ordinal}/{total} id={cid} model={key['model_id']} "
        f"condition={key['condition_id']} date={key['decision_date']}",
        flush=True,
    )
    try:
        result = evaluate_one_call(**call_kwargs)
        result["call_id"] = cid
        elapsed = time.perf_counter() - started
        # Persist the result file first, then mark the checkpoint completed.
        _append_rows([result], out_jsonl, out_csv)
        store.complete(cid, result, elapsed)
        counts = store.counts()
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0) + counts.get("timed_out", 0)
        print(
            f"[CALL DONE] {ordinal}/{total} id={cid} elapsed={elapsed:.1f}s "
            f"saved=yes completed={completed} failed={failed}",
            flush=True,
        )
        return "completed"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        store.fail(cid, exc, elapsed)
        print(
            f"[CALL FAILED] {ordinal}/{total} id={cid} elapsed={elapsed:.1f}s "
            f"type={type(exc).__name__} reason={str(exc)[:300]} saved=yes",
            flush=True,
        )
        if args.stop_on_failure:
            raise
        return "failed"


def run_prompt_robustness(features, tickers, decision_idx, args, cfg, out_jsonl, out_csv, store):
    plan = []
    for model in args.models:
        for pid, plist in PROMPT_ROBUSTNESS_PARAPHRASES.items():
            if args.policies and pid not in args.policies:
                continue
            for j, prompt_text in enumerate(plist):
                for i in decision_idx:
                    plan.append(dict(
                        experiment_id="prompt_robustness", condition_id=f"{pid}_paraphrase_{j:02d}", model=model,
                        policy_id=pid, prompt_text=prompt_text, row=features.iloc[i], tickers=tickers, cfg=cfg,
                        date=features.index[i], dry_run=args.dry_run, paraphrase_id=f"{j:02d}", mask_condition="real_ticker",
                    ))
    if args.max_calls:
        plan = plan[:args.max_calls]
    for ordinal, kwargs in enumerate(plan, 1):
        _execute_one(store=store, out_jsonl=out_jsonl, out_csv=out_csv,
                     total=len(plan), ordinal=ordinal, args=args, call_kwargs=kwargs)


def run_ticker_masking(features, tickers, decision_idx, args, cfg, out_jsonl, out_csv, store):
    plan = []
    conditions = ["real_ticker", "masked_ticker", "shuffled_masked"]
    for model in args.models:
        for pid, prompt_text in BASE_POLICY_PROMPTS.items():
            if args.policies and pid not in args.policies:
                continue
            for cond in conditions:
                lm = real_to_masked_map(tickers, "real_ticker" if cond == "real_ticker" else ("shuffled_masked" if cond == "shuffled_masked" else "masked_ticker"), seed=args.seed)
                for i in decision_idx:
                    plan.append(dict(
                        experiment_id="ticker_masking", condition_id=f"{pid}_{cond}", model=model,
                        policy_id=pid, prompt_text=prompt_text, row=features.iloc[i], tickers=tickers, cfg=cfg,
                        date=features.index[i], label_map=lm, dry_run=args.dry_run, mask_condition=cond,
                    ))
    if args.max_calls:
        plan = plan[:args.max_calls]
    for ordinal, kwargs in enumerate(plan, 1):
        _execute_one(store=store, out_jsonl=out_jsonl, out_csv=out_csv,
                     total=len(plan), ordinal=ordinal, args=args, call_kwargs=kwargs)


def run_policy_complexity(features, tickers, decision_idx, args, cfg, out_jsonl, out_csv, store):
    plan = []
    for model in args.models:
        for level, prompt_text in POLICY_COMPLEXITY_PROMPTS.items():
            for i in decision_idx:
                plan.append(dict(
                    experiment_id="policy_complexity", condition_id=level, model=model,
                    policy_id=level, prompt_text=prompt_text, row=features.iloc[i], tickers=tickers, cfg=cfg,
                    date=features.index[i], dry_run=args.dry_run, complexity_level=level,
                ))
    if args.max_calls:
        plan = plan[:args.max_calls]
    for ordinal, kwargs in enumerate(plan, 1):
        _execute_one(store=store, out_jsonl=out_jsonl, out_csv=out_csv,
                     total=len(plan), ordinal=ordinal, args=args, call_kwargs=kwargs)


def run_constraint_stress(features, tickers, decision_idx, args, cfg, out_jsonl, out_csv, store):
    plan = []
    # Use P1 as a directional base reference, because conflict prompts are generally target-seeking.
    for model in args.models:
        for conflict_type, prompt_text in CONSTRAINT_CONFLICT_PROMPTS.items():
            for reminded in (True, False):
                cid = f"{conflict_type}_{'reminded' if reminded else 'blind'}"
                for i in decision_idx:
                    plan.append(dict(
                        experiment_id="constraint_conflict_stress", condition_id=cid, model=model,
                        policy_id="P1", prompt_text=prompt_text, row=features.iloc[i], tickers=tickers, cfg=cfg,
                        date=features.index[i], dry_run=args.dry_run, conflict_type=conflict_type,
                        constraint_reminder=reminded,
                    ))
    if args.max_calls:
        plan = plan[:args.max_calls]
    for ordinal, kwargs in enumerate(plan, 1):
        _execute_one(store=store, out_jsonl=out_jsonl, out_csv=out_csv,
                     total=len(plan), ordinal=ordinal, args=args, call_kwargs=kwargs)


def run_model_generalization(features, tickers, decision_idx, args, cfg, out_jsonl, out_csv, store):
    plan = []
    for model in args.models:
        for pid, prompt_text in BASE_POLICY_PROMPTS.items():
            if args.policies and pid not in args.policies:
                continue
            for i in decision_idx:
                plan.append(dict(
                    experiment_id="model_family_generalization", condition_id=pid, model=model,
                    policy_id=pid, prompt_text=prompt_text, row=features.iloc[i], tickers=tickers, cfg=cfg,
                    date=features.index[i], dry_run=args.dry_run,
                ))
    if args.max_calls:
        plan = plan[:args.max_calls]
    for ordinal, kwargs in enumerate(plan, 1):
        _execute_one(store=store, out_jsonl=out_jsonl, out_csv=out_csv,
                     total=len(plan), ordinal=ordinal, args=args, call_kwargs=kwargs)


def run_sensitivity_template(args, outdir: Path):
    """Create sensitivity grid files.

    Cost/tau/wmax sensitivity can be computed from saved raw/projected weights;
    rebalance frequency requires new decision dates and usually new LLM calls.
    This template is exported so the paper package has a reproducible plan even
    when not all sensitivity branches are executed immediately.
    """
    base = {"cost_bps": args.tcost * 10000, "rebalance_days": args.rebalance, "tau": args.turncap, "wmax": args.maxw}
    rows = []
    for c in [0, 5, 10, 25, 50]: rows.append({**base, "sensitivity_type": "cost", "cost_bps": c})
    for r in [21, 42, 63, 126]: rows.append({**base, "sensitivity_type": "rebalance", "rebalance_days": r})
    for tau in [0.10, 0.25, 0.50, 1.00]: rows.append({**base, "sensitivity_type": "turnover_cap", "tau": tau})
    for mw in [0.40, 0.60, 0.80]: rows.append({**base, "sensitivity_type": "max_weight", "wmax": mw})
    pd.DataFrame(rows).to_csv(outdir / "q1_sensitivity_grid_template.csv", index=False)


def export_tables(outdir: Path):
    try:
        from q1_experiments.tables import generate_summary_tables
        generate_summary_tables(outdir)
    except Exception as e:
        print(f"[WARN] q1 table generation failed: {e}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NLPI Q1 reliability/safety experiment runner")
    p.add_argument("--experiments", nargs="+", default=["prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization"],
                   choices=["all", "prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization", "sensitivity_template"])
    p.add_argument("--models", nargs="+", default=["gemma3:270m", "gemma3:1b", "llama3.1:8b"], help="Ollama model tags")
    p.add_argument("--policies", nargs="+", default=None, help="Optional subset: P1 P2 P3 P4 P5 P6")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2025-12-29")
    p.add_argument("--rebalance", type=int, default=42)
    p.add_argument("--tcost", type=float, default=0.0010)
    p.add_argument("--maxw", type=float, default=0.60)
    p.add_argument("--turncap", type=float, default=0.25)
    p.add_argument("--prompt-cap", type=float, default=60.0, help="language-level target percentage")
    p.add_argument("--decision-sample", choices=["full", "stratified", "first"], default="stratified")
    p.add_argument("--n-per-regime", type=int, default=10)
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--max-calls", type=int, default=None, help="Debug guard per module")
    p.add_argument("--outdir", default="outputs/q1_reliability_package")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-connect-timeout", type=float, default=30.0)
    p.add_argument("--ollama-read-timeout", type=float, default=900.0,
                   help="Maximum seconds to wait for one Ollama response")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Connection/HTTP retries after the first attempt")
    p.add_argument("--parse-retries", type=int, default=1,
                   help="Full regenerations after invalid JSON")
    p.add_argument("--timeout-retries", type=int, default=1,
                   help="Retries after a read timeout")
    p.add_argument("--num-predict", type=int, default=512,
                   help="Maximum Ollama output tokens per generation")
    p.add_argument("--ollama-keep-alive", default="30m",
                   help="How long Ollama keeps the current model loaded")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="Stop the run after a final failed call; default records failure and continues")
    p.add_argument("--fresh-start", action="store_true",
                   help="Delete this outdir's reliability call logs/checkpoint before running")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Do not call Ollama; generate schema-valid reference outputs only")
    p.add_argument("--synthetic-data", action="store_true", help="Use deterministic synthetic prices for local schema tests")
    p.add_argument("--skip-table-generation", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    _safe_mkdir(outdir)
    lock_ctx = output_lock(outdir)
    lock_ctx.__enter__()
    atexit.register(lambda: lock_ctx.__exit__(None, None, None))
    logs_dir = outdir / "logs"
    tables_dir = outdir / "tables"
    _safe_mkdir(logs_dir); _safe_mkdir(tables_dir)
    out_jsonl = logs_dir / "q1_decision_log.jsonl"
    out_csv = logs_dir / "q1_decision_log.csv"
    checkpoint_db = logs_dir / "q1_call_checkpoint.sqlite3"
    if args.fresh_start:
        for path in (
            out_jsonl, out_csv, checkpoint_db,
            Path(str(checkpoint_db) + "-wal"),
            Path(str(checkpoint_db) + "-shm"),
        ):
            if path.exists():
                path.unlink()
    store = CheckpointStore(checkpoint_db)
    atexit.register(store.close)
    _rebuild_exports_from_checkpoint(store, out_jsonl, out_csv)

    cfg = {
        "run_id": f"q1_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "ollama_url": args.ollama_url,
        "rebalance_days": args.rebalance,
        "tcost": args.tcost,
        "max_weight": args.maxw,
        "turnover_cap": args.turncap,
        "prompt_cap_pct": args.prompt_cap,
        "collapse_l1_threshold": 1e-3,
        "ollama_connect_timeout": args.ollama_connect_timeout,
        "ollama_read_timeout": args.ollama_read_timeout,
        "max_retries": args.max_retries,
        "parse_retries": args.parse_retries,
        "timeout_retries": args.timeout_retries,
        "num_predict": args.num_predict,
        "ollama_keep_alive": args.ollama_keep_alive,
        "seed": args.seed,
    }

    print("[INFO] Loading prices and features...")
    prices, features, tickers = load_data(args)
    try:
        features.to_parquet(outdir / "features.parquet")
    except Exception:
        features.to_pickle(outdir / "features.pkl")
        features.to_csv(outdir / "features.csv")
    price_audit_table(prices).to_csv(outdir / "price_audit.csv", index=False)

    idx = decision_indices(features, rebalance_days=args.rebalance, mode=args.decision_sample, n_per_regime=args.n_per_regime, seed=args.seed, max_dates=args.max_dates)
    pd.DataFrame({"decision_index": idx, "decision_date": [features.index[i] for i in idx]}).to_csv(outdir / "decision_dates.csv", index=False)
    write_manifest(outdir, args, tickers, idx, features)
    print(f"[INFO] Decision dates selected: {len(idx)}")

    experiments = set(args.experiments)
    if "all" in experiments:
        experiments = {"prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization", "sensitivity_template"}

    if "prompt_robustness" in experiments:
        print("[RUN] prompt_robustness")
        run_prompt_robustness(features, tickers, idx, args, cfg, out_jsonl, out_csv, store)
    if "ticker_masking" in experiments:
        print("[RUN] ticker_masking")
        run_ticker_masking(features, tickers, idx, args, cfg, out_jsonl, out_csv, store)
    if "policy_complexity" in experiments:
        print("[RUN] policy_complexity")
        run_policy_complexity(features, tickers, idx, args, cfg, out_jsonl, out_csv, store)
    if "constraint_stress" in experiments:
        print("[RUN] constraint_conflict_stress")
        run_constraint_stress(features, tickers, idx, args, cfg, out_jsonl, out_csv, store)
    if "model_generalization" in experiments:
        print("[RUN] model_family_generalization")
        run_model_generalization(features, tickers, idx, args, cfg, out_jsonl, out_csv, store)
    if "sensitivity_template" in experiments:
        print("[EXPORT] sensitivity grid template")
        run_sensitivity_template(args, outdir)

    if not args.skip_table_generation:
        export_tables(outdir)
    print(f"[DONE] Q1 experiment package outputs: {outdir}")


if __name__ == "__main__":
    main()
