from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import llm
from core import make_features
from llm import build_fewshot_db, build_prompt, render_fewshot_block
from engine.strategies import project_capped_simplex
from q1_experiments.prompt_library import BASE_POLICY_PROMPTS
from q1_experiments.reference_policies import reference_weights
from q1_experiments.runner import build_eval_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUN = ROOT / "outputs" / "NLPI_PAPER_CANONICAL_V1_20260717_152949"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "p2_reconciliation"
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VEA", "VWO", "AGG", "BND", "SHY",
    "IEF", "TLT", "TIP", "LQD", "HYG", "GLD", "SLV", "DBC", "VNQ", "UUP", "BIL",
]
CONDITIONS = ("A_wfcv_exact", "B_wfcv_no_fewshot", "C_bridge_canonical")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def select_wfcv_dates(index: pd.DatetimeIndex, train_days: int, test_days: int, rebalance: int, n_dates: int) -> list[dict]:
    candidates: list[dict] = []
    fold = 1
    test_start = int(train_days)
    while test_start < len(index) - 1:
        test_end = min(len(index), test_start + int(test_days))
        if test_end - test_start < 2:
            break
        for pos in range(test_start, test_end, int(rebalance)):
            candidates.append({
                "fold": fold,
                "position": pos,
                "decision_date": str(pd.Timestamp(index[pos]).date()),
                "train_start": 0,
                "train_end": test_start,
            })
        fold += 1
        test_start = test_end
    if len(candidates) < n_dates:
        raise RuntimeError(f"Only {len(candidates)} WFCV-compatible dates are available; requested {n_dates}.")
    chosen = np.linspace(0, len(candidates) - 1, num=n_dates, dtype=int)
    return [candidates[int(i)] for i in sorted(set(chosen.tolist()))]


def ollama_inventory(url: str) -> dict:
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=5)
        r.raise_for_status()
        rows = []
        for item in r.json().get("models", []):
            rows.append({
                "name": item.get("name") or item.get("model"),
                "digest": item.get("digest", ""),
                "modified_at": item.get("modified_at", ""),
                "size": item.get("size", 0),
            })
        return {"status": "ok", "models": rows}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "models": []}


def parse_weights(output, tickers: list[str]) -> pd.Series:
    if output is None:
        raise RuntimeError("LLM returned no parsed output")
    if hasattr(output, "weights"):
        raw = output.weights
    elif isinstance(output, dict):
        raw = output.get("weights", output)
    else:
        raise RuntimeError(f"Unsupported LLM output type: {type(output)!r}")
    vals = {}
    for k, v in dict(raw).items():
        try:
            vals[str(k).strip()] = max(float(v), 0.0)
        except Exception:
            continue
    return pd.Series(vals, dtype=float).reindex(tickers).fillna(0.0)


def make_prompt(condition: str, row: pd.Series, tickers: list[str], fewshot: str, cfg: dict) -> str:
    wfcv_cfg = {
        "prompt_profile": 2,
        "prompt_cap_pct": cfg["prompt_cap_pct"],
        "max_weight": cfg["max_weight"],
        "turnover_cap": cfg["turnover_cap"],
    }
    if condition == "A_wfcv_exact":
        return build_prompt(row.to_dict(), tickers, fewshot, wfcv_cfg)
    if condition == "B_wfcv_no_fewshot":
        return build_prompt(row.to_dict(), tickers, "", wfcv_cfg)
    if condition == "C_bridge_canonical":
        return build_eval_prompt(
            BASE_POLICY_PROMPTS["P2"], row, tickers, wfcv_cfg,
            label_map={a: a for a in tickers}, constraint_reminder=True,
        )
    raise ValueError(condition)


def call_one(prompt: str, model: str, args, dry_run: bool, ref: pd.Series) -> tuple[pd.Series, dict]:
    if dry_run:
        return ref.copy(), {
            "json_valid": 1, "parse_failed": 0, "repair_used": 0, "elapsed_sec": 0.0,
            "raw_response": "DRY_RUN_REFERENCE", "attempt_count": 0, "dry_run": 1,
        }
    out = llm.call_llm(
        prompt, args.ollama_url, model, log_fn=lambda x: print(x, flush=True),
        timeout=(args.connect_timeout, args.timeout), max_retries=args.max_retries,
        parse_retries=args.parse_retries, timeout_retries=args.timeout_retries,
        num_predict=args.num_predict, seed=args.seed, temperature=args.temperature,
        top_p=args.top_p, keep_alive=args.keep_alive,
    )
    return parse_weights(out, TICKERS), dict(llm.LAST_CALL_META)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["condition_id", "model_id"], as_index=False).agg(
        n=("call_id", "size"),
        projected_fidelity=("projected_fidelity", "mean"),
        raw_fidelity=("raw_fidelity", "mean"),
        mean_allocation_l1=("allocation_l1_to_reference", "mean"),
        json_valid=("json_valid", "mean"),
        repair_rate=("repair_used", "mean"),
        mean_latency_sec=("latency_sec", "mean"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Controlled P2 WFCV/bridge reconciliation without modifying existing results.")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--models", nargs="+", default=["gemma3:270m", "gemma3:1b", "llama3.1:8b"])
    p.add_argument("--n-dates", type=int, default=7)
    p.add_argument("--train-days", type=int, default=756)
    p.add_argument("--test-days", type=int, default=252)
    p.add_argument("--rebalance-days", type=int, default=42)
    p.add_argument("--max-weight", type=float, default=0.60)
    p.add_argument("--turnover-cap", type=float, default=0.25)
    p.add_argument("--prompt-cap-pct", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--connect-timeout", type=float, default=30.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--parse-retries", type=int, default=1)
    p.add_argument("--timeout-retries", type=int, default=1)
    p.add_argument("--keep-alive", default="30m")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    source_run = args.source_run.resolve()
    price_path = source_run / "data" / "adjusted_close.csv"
    protocol_path = source_run / "config" / "paper_protocol.json"
    if not price_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("source-run must contain data/adjusted_close.csv and config/paper_protocol.json")
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).reindex(columns=TICKERS)
    features = make_features(prices, TICKERS)
    dates = select_wfcv_dates(features.index, args.train_days, args.test_days, args.rebalance_days, args.n_dates)

    settings = {
        "experiment_id": args.experiment_id,
        "source_run": str(source_run),
        "source_price_sha256": sha256_file(price_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "models": args.models,
        "conditions": list(CONDITIONS),
        "policy_id": "P2",
        "n_dates": len(dates),
        "train_days": args.train_days,
        "test_days": args.test_days,
        "rebalance_days": args.rebalance_days,
        "max_weight": args.max_weight,
        "turnover_cap": args.turnover_cap,
        "prompt_cap_pct": args.prompt_cap_pct,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.num_predict,
        "dry_run": bool(args.dry_run),
    }
    settings_hash = sha256_text(json.dumps(settings, sort_keys=True))
    out = args.results_root.resolve() / args.experiment_id
    manifest_path = out / "manifest.json"
    if out.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"Refusing to use existing directory without manifest: {out}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("settings_hash") != settings_hash:
            raise RuntimeError("Experiment ID already exists with different settings. Use a new --experiment-id.")
    else:
        out.mkdir(parents=True)
        (out / "checkpoints").mkdir()
        json_dump(out / "decision_dates.json", dates)
        json_dump(out / "environment.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "ollama": ollama_inventory(args.ollama_url),
        })
        json_dump(manifest_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running", "settings_hash": settings_hash, "settings": settings,
            "expected_calls": len(args.models) * len(CONDITIONS) * len(dates),
        })

    calls_path = out / "p2_reconciliation_calls.csv"
    existing = pd.read_csv(calls_path) if calls_path.is_file() else pd.DataFrame()
    completed = set(existing["call_id"].astype(str)) if not existing.empty else set()
    rows: list[dict] = []
    total = len(args.models) * len(CONDITIONS) * len(dates)
    done = len(completed)
    print(f"[RESUME] completed={done} pending={total-done} output={out}", flush=True)

    cfg = {"prompt_cap_pct": args.prompt_cap_pct, "max_weight": args.max_weight, "turnover_cap": args.turnover_cap}
    for date_info in dates:
        pos = int(date_info["position"])
        row = features.iloc[pos]
        train = features.iloc[int(date_info["train_start"]):int(date_info["train_end"])]
        fewshots = build_fewshot_db(train, TICKERS, k=8, profile_id=2, prompt_cap_pct=args.prompt_cap_pct)
        fewshot_block = render_fewshot_block(fewshots, max_k=4)
        ref = project_capped_simplex(reference_weights("P2", row, TICKERS, args.prompt_cap_pct / 100.0), args.max_weight)
        target = str(ref.idxmax())
        for model in args.models:
            for condition in CONDITIONS:
                prompt = make_prompt(condition, row, TICKERS, fewshot_block, cfg)
                applied_fewshot = fewshot_block if condition == "A_wfcv_exact" else ""
                key = f"{args.experiment_id}|{condition}|{model}|{date_info['decision_date']}"
                call_id = sha256_text(key)[:24]
                if call_id in completed:
                    continue
                raw, meta = call_one(prompt, model, args, args.dry_run, ref)
                projected = project_capped_simplex(raw, args.max_weight)
                raw_top = str(raw.idxmax())
                projected_top = str(projected.idxmax())
                rec = {
                    "call_id": call_id, "experiment_id": args.experiment_id,
                    "condition_id": condition, "model_id": model, "policy_id": "P2",
                    "decision_date": date_info["decision_date"], "fold": date_info["fold"],
                    "position": pos, "target_asset": target, "top_asset_raw": raw_top,
                    "top_asset_projected": projected_top, "raw_fidelity": int(raw_top == target),
                    "projected_fidelity": int(projected_top == target),
                    "allocation_l1_to_reference": float((projected - ref).abs().sum()),
                    "projection_l1": float((raw - projected).abs().sum()),
                    "prompt_hash": sha256_text(prompt), "prompt_length": len(prompt),
                    "fewshot_hash": sha256_text(applied_fewshot), "fewshot_length": len(applied_fewshot),
                    "json_valid": int(meta.get("json_valid", 0) or 0),
                    "parse_fail": int(meta.get("parse_failed", 0) or 0),
                    "repair_used": int(meta.get("repair_used", 0) or 0),
                    "latency_sec": float(meta.get("elapsed_sec", 0.0) or 0.0),
                    "attempt_count": int(meta.get("attempt_count", 0) or 0),
                    "raw_output": str(meta.get("raw_response", "")),
                    "prompt_text": prompt, "raw_weights": raw.to_json(),
                    "projected_weights": projected.to_json(), "reference_weights": ref.to_json(),
                    "dry_run": int(args.dry_run),
                }
                rows.append(rec)
                current = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
                current.to_csv(calls_path, index=False)
                json_dump(out / "checkpoints" / f"{call_id}.json", rec)
                done += 1
                print(f"[CHECKPOINT] {done}/{total} {condition} {model} {date_info['decision_date']} fidelity={rec['projected_fidelity']}", flush=True)

    final = pd.read_csv(calls_path)
    summarize(final).to_csv(out / "p2_reconciliation_summary.csv", index=False)
    prompts = final[["condition_id", "decision_date", "fold", "prompt_hash", "prompt_length", "fewshot_hash", "fewshot_length"]].drop_duplicates()
    prompts.to_csv(out / "prompt_audit.csv", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(), "completed_calls": len(final)})
    json_dump(manifest_path, manifest)
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
