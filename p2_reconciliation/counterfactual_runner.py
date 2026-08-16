from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import llm
from core import make_features
from engine.strategies import project_capped_simplex
from llm import build_fewshot_db, build_prompt, render_fewshot_block
from q1_experiments.reference_policies import reference_weights

from .runner import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SOURCE_RUN,
    TICKERS,
    json_dump,
    ollama_inventory,
    parse_weights,
    select_wfcv_dates,
    sha256_file,
    sha256_text,
    summarize,
)


CONDITIONS = ("D_counterfactual_fewshot", "E_counterfactual_no_fewshot")
COUNTERFACTUAL_TARGETS = ("AGG", "BND", "SHY", "IEF", "TIP", "LQD", "UUP")


def counterfactual_row(row: pd.Series, target_asset: str) -> tuple[pd.Series, dict]:
    """Swap only the P2 decision variable so that a non-BIL asset is the minimum-volatility asset."""
    if target_asset == "BIL" or target_asset not in TICKERS:
        raise ValueError(f"Invalid counterfactual target: {target_asset}")
    bil_key = "BIL_vol3m"
    target_key = f"{target_asset}_vol3m"
    if bil_key not in row.index or target_key not in row.index:
        raise KeyError(f"Missing counterfactual feature: {bil_key} or {target_key}")
    changed = row.copy()
    original_bil = float(changed[bil_key])
    original_target = float(changed[target_key])
    changed[bil_key], changed[target_key] = original_target, original_bil
    observed_min = min(TICKERS, key=lambda t: float(changed[f"{t}_vol3m"]))
    if observed_min != target_asset:
        raise RuntimeError(
            f"Counterfactual construction failed: assigned={target_asset}, observed={observed_min}"
        )
    return changed, {
        "manipulation": "swap_vol3m_only",
        "assigned_target_asset": target_asset,
        "original_bil_vol3m": original_bil,
        "original_target_vol3m": original_target,
        "counterfactual_bil_vol3m": float(changed[bil_key]),
        "counterfactual_target_vol3m": float(changed[target_key]),
    }


def make_counterfactual_prompt(condition: str, row: pd.Series, fewshot: str, cfg: dict) -> str:
    wfcv_cfg = {
        "prompt_profile": 2,
        "prompt_cap_pct": cfg["prompt_cap_pct"],
        "max_weight": cfg["max_weight"],
        "turnover_cap": cfg["turnover_cap"],
    }
    if condition == "D_counterfactual_fewshot":
        return build_prompt(row.to_dict(), TICKERS, fewshot, wfcv_cfg)
    if condition == "E_counterfactual_no_fewshot":
        return build_prompt(row.to_dict(), TICKERS, "", wfcv_cfg)
    raise ValueError(condition)


def call_one(prompt: str, model: str, args, dry_run: bool, ref: pd.Series) -> tuple[pd.Series, dict]:
    if dry_run:
        return ref.copy(), {
            "json_valid": 1,
            "parse_failed": 0,
            "repair_used": 0,
            "elapsed_sec": 0.0,
            "raw_response": "DRY_RUN_REFERENCE",
            "attempt_count": 0,
            "dry_run": 1,
        }
    out = llm.call_llm(
        prompt,
        args.ollama_url,
        model,
        log_fn=lambda x: print(x, flush=True),
        timeout=(args.connect_timeout, args.timeout),
        max_retries=args.max_retries,
        parse_retries=args.parse_retries,
        timeout_retries=args.timeout_retries,
        num_predict=args.num_predict,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        keep_alive=args.keep_alive,
    )
    return parse_weights(out, TICKERS), dict(llm.LAST_CALL_META)


def main() -> None:
    p = argparse.ArgumentParser(
        description="P2 counterfactual minimum-volatility test; writes only to a new experiment directory."
    )
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

    if args.n_dates != len(COUNTERFACTUAL_TARGETS):
        raise ValueError(
            f"This preregistered design requires exactly {len(COUNTERFACTUAL_TARGETS)} dates; "
            f"received {args.n_dates}."
        )
    source_run = args.source_run.resolve()
    price_path = source_run / "data" / "adjusted_close.csv"
    protocol_path = source_run / "config" / "paper_protocol.json"
    if not price_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            "source-run must contain data/adjusted_close.csv and config/paper_protocol.json"
        )
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).reindex(columns=TICKERS)
    features = make_features(prices, TICKERS)
    dates = select_wfcv_dates(
        features.index, args.train_days, args.test_days, args.rebalance_days, args.n_dates
    )
    schedule = []
    for date_info, target in zip(dates, COUNTERFACTUAL_TARGETS):
        item = dict(date_info)
        item["counterfactual_target_asset"] = target
        schedule.append(item)

    settings = {
        "experiment_id": args.experiment_id,
        "source_run": str(source_run),
        "source_price_sha256": sha256_file(price_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "models": args.models,
        "conditions": list(CONDITIONS),
        "policy_id": "P2",
        "counterfactual_method": "swap BIL_vol3m with the preregistered target asset vol3m",
        "counterfactual_targets": list(COUNTERFACTUAL_TARGETS),
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
            raise RuntimeError(
                "Experiment ID already exists with different settings. Use a new --experiment-id."
            )
    else:
        out.mkdir(parents=True)
        (out / "checkpoints").mkdir()
        json_dump(out / "decision_dates.json", schedule)
        json_dump(
            out / "environment.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "ollama": ollama_inventory(args.ollama_url),
            },
        )
        json_dump(
            manifest_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "settings_hash": settings_hash,
                "settings": settings,
                "expected_calls": len(args.models) * len(CONDITIONS) * len(dates),
            },
        )

    calls_path = out / "p2_counterfactual_calls.csv"
    existing = pd.read_csv(calls_path) if calls_path.is_file() else pd.DataFrame()
    completed = set(existing["call_id"].astype(str)) if not existing.empty else set()
    rows: list[dict] = []
    total = len(args.models) * len(CONDITIONS) * len(dates)
    done = len(completed)
    print(f"[RESUME] completed={done} pending={total-done} output={out}", flush=True)

    cfg = {
        "prompt_cap_pct": args.prompt_cap_pct,
        "max_weight": args.max_weight,
        "turnover_cap": args.turnover_cap,
    }
    for date_info, assigned_target in zip(dates, COUNTERFACTUAL_TARGETS):
        pos = int(date_info["position"])
        original_row = features.iloc[pos]
        row, manipulation = counterfactual_row(original_row, assigned_target)
        train = features.iloc[int(date_info["train_start"]):int(date_info["train_end"])]
        fewshots = build_fewshot_db(
            train, TICKERS, k=8, profile_id=2, prompt_cap_pct=args.prompt_cap_pct
        )
        fewshot_block = render_fewshot_block(fewshots, max_k=4)
        ref = project_capped_simplex(
            reference_weights("P2", row, TICKERS, args.prompt_cap_pct / 100.0),
            args.max_weight,
        )
        target = str(ref.idxmax())
        if target != assigned_target:
            raise RuntimeError(f"Reference target mismatch: assigned={assigned_target}, target={target}")
        for model in args.models:
            for condition in CONDITIONS:
                prompt = make_counterfactual_prompt(condition, row, fewshot_block, cfg)
                applied_fewshot = fewshot_block if condition == "D_counterfactual_fewshot" else ""
                key = f"{args.experiment_id}|{condition}|{model}|{date_info['decision_date']}|{target}"
                call_id = sha256_text(key)[:24]
                if call_id in completed:
                    continue
                raw, meta = call_one(prompt, model, args, args.dry_run, ref)
                projected = project_capped_simplex(raw, args.max_weight)
                raw_top = str(raw.idxmax())
                projected_top = str(projected.idxmax())
                rec = {
                    "call_id": call_id,
                    "experiment_id": args.experiment_id,
                    "condition_id": condition,
                    "model_id": model,
                    "policy_id": "P2",
                    "decision_date": date_info["decision_date"],
                    "fold": date_info["fold"],
                    "position": pos,
                    "original_target_asset": str(
                        reference_weights(
                            "P2", original_row, TICKERS, args.prompt_cap_pct / 100.0
                        ).idxmax()
                    ),
                    "assigned_counterfactual_target": assigned_target,
                    "target_asset": target,
                    "top_asset_raw": raw_top,
                    "top_asset_projected": projected_top,
                    "raw_fidelity": int(raw_top == target),
                    "projected_fidelity": int(projected_top == target),
                    "bil_copy_raw": int(raw_top == "BIL"),
                    "bil_copy_projected": int(projected_top == "BIL"),
                    "allocation_l1_to_reference": float((projected - ref).abs().sum()),
                    "projection_l1": float((raw - projected).abs().sum()),
                    "prompt_hash": sha256_text(prompt),
                    "prompt_length": len(prompt),
                    "fewshot_hash": sha256_text(applied_fewshot),
                    "fewshot_length": len(applied_fewshot),
                    "json_valid": int(meta.get("json_valid", 0) or 0),
                    "parse_fail": int(meta.get("parse_failed", 0) or 0),
                    "repair_used": int(meta.get("repair_used", 0) or 0),
                    "latency_sec": float(meta.get("elapsed_sec", 0.0) or 0.0),
                    "attempt_count": int(meta.get("attempt_count", 0) or 0),
                    "raw_output": str(meta.get("raw_response", "")),
                    "prompt_text": prompt,
                    "raw_weights": raw.to_json(),
                    "projected_weights": projected.to_json(),
                    "reference_weights": ref.to_json(),
                    "dry_run": int(args.dry_run),
                    **manipulation,
                }
                rows.append(rec)
                current = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
                current.to_csv(calls_path, index=False)
                json_dump(out / "checkpoints" / f"{call_id}.json", rec)
                done += 1
                print(
                    f"[CHECKPOINT] {done}/{total} {condition} {model} "
                    f"{date_info['decision_date']} target={target} fidelity={rec['projected_fidelity']} "
                    f"bil_copy={rec['bil_copy_projected']}",
                    flush=True,
                )

    final = pd.read_csv(calls_path)
    summary = summarize(final)
    bil = final.groupby(["condition_id", "model_id"], as_index=False).agg(
        bil_copy_rate=("bil_copy_projected", "mean")
    )
    summary.merge(bil, on=["condition_id", "model_id"], how="left").to_csv(
        out / "p2_counterfactual_summary.csv", index=False
    )
    final[
        [
            "condition_id",
            "decision_date",
            "fold",
            "assigned_counterfactual_target",
            "prompt_hash",
            "prompt_length",
            "fewshot_hash",
            "fewshot_length",
        ]
    ].drop_duplicates().to_csv(out / "prompt_audit.csv", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_calls": len(final),
        }
    )
    json_dump(manifest_path, manifest)
    print(f"[STATUS] completed={len(final)} expected={total}", flush=True)
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
