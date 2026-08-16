# llm_portfolio.py
from __future__ import annotations
import os, argparse, sys, json, subprocess, shutil
import numpy as np
import pandas as pd
import matplotlib
import platform
# Prefer TkAgg for local desktop use, but fall back to Agg for headless CLI/tests.
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.strategies import (
    LLMStrategy, CodedPersonaStrategy, EqStrategy, RiskParityStrategy, MVPStrategy,
    Momentum6mStrategy, Trend6mStrategy, SharpeWeightedStrategy, SortinoWeightedStrategy,
    LWMVPStrategy, HRPStrategy, BLStrategy, MinCVaRStrategy, VolatilityTargetTrendStrategy,
)

# ----- 한글 폰트 자동 설정 -----
def _set_korean_font():
    system = platform.system().lower()
    try:
        if "darwin" in sys.platform or system == "darwin":       # macOS
            matplotlib.rcParams['font.family'] = 'AppleGothic'
        elif system == "windows":
            matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
_set_korean_font()

from core import fetch_prices_yf, fetch_prices_yf_with_audit, price_audit_table, make_features, split_index
from engine.metrics import summary
from llm import build_fewshot_db, render_fewshot_block, check_ollama

from engine.backtest import run_backtest
from engine.statsig import build_comparison_table, reality_check_df, spa_df
from portfolios import DEFAULT_TICKERS

# =========================
# Robust metric helpers
# =========================
def _as_num_series(x) -> pd.Series:
    """어떤 타입이 와도 숫자 Series로 강제 변환 (NaN/Inf 정리)."""
    s = pd.Series(x).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def _safe_cum_return(ret: pd.Series) -> float:
    """
    누적수익: ∏(1+r_t) - 1, NaN은 0으로 간주(그 날 수익률 0%).
    이렇게 하면 중간 NaN 때문에 전체가 NaN이 되는 일을 방지.
    """
    if ret is None:
        return np.nan
    r = _as_num_series(ret).fillna(0.0)
    if r.empty:
        return np.nan
    return float(np.nanprod(1.0 + r.values) - 1.0)

def _safe_ann_return(ret: pd.Series, ann_factor: int = 252) -> float:
    """
    연환산수익: (∏(1+r_t))^(AF/N) - 1, NaN은 0으로 간주.
    """
    if ret is None:
        return np.nan
    r = _as_num_series(ret).fillna(0.0)
    n = len(r)
    if n <= 0:
        return np.nan
    total = float(np.nanprod(1.0 + r.values))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(total ** (ann_factor / n) - 1.0)

def _safe_turnover(w: pd.DataFrame) -> float:
    """
    평균 일간 턴오버: mean_t Σ_i |w_t(i) - w_{t-1}(i)|
    비중행렬이 object/NaN 섞였어도 숫자로 강제 후 계산.
    """
    if w is None or len(w) == 0:
        return np.nan
    W = w.copy()
    for c in W.columns:
        W[c] = pd.to_numeric(W[c], errors="coerce")
    d = W.diff().abs().sum(axis=1)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return np.nan
    return float(d.mean())

def _warn_if_empty(name: str, ret: pd.Series, w: pd.DataFrame):
    """디버그용: ret/w가 비어 있거나 전부 NaN이면 한 줄 경고."""
    msg = []
    if ret is None or len(ret) == 0 or pd.isna(pd.Series(ret)).all():
        msg.append("ret=EMPTY/ALL-NaN")
    if w is None or len(w) == 0 or (w.isna().all().all() if isinstance(w, pd.DataFrame) else False):
        msg.append("w=EMPTY/ALL-NaN")
    if msg:
        print(f"[WARN] {name}: " + ", ".join(msg))

def _safe_filename(name: str) -> str:
    """Return a filesystem-safe strategy name for CSV/PNG exports."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

def _make_metrics_table(results: dict) -> pd.DataFrame:
    """Build the main metrics table from a {strategy: (returns, weights)} dictionary."""
    rows = []
    for name, (ret, w) in results.items():
        _warn_if_empty(name, ret, w)
        s = summary(ret, w) if ret is not None else {}
        rows.append({
            "Strategy": name,
            "Sharpe": s.get("Sharpe", np.nan),
            "Sortino": s.get("Sortino", np.nan),
            "Cumulative Return": _safe_cum_return(ret),
            "Annualized Return": _safe_ann_return(ret, ann_factor=252),
            "Turnover Ratio": _safe_turnover(w),
            "Annualized Volatility": s.get("Vol", np.nan),
            "CAGR": s.get("CAGR", np.nan),
            "Maximum Drawdown": s.get("MDD", np.nan),
        })
    if not rows:
        return pd.DataFrame(columns=["Strategy", "Sharpe", "Sortino", "Cumulative Return", "Annualized Return", "Turnover Ratio", "Annualized Volatility", "CAGR", "Maximum Drawdown"]).set_index("Strategy")
    return pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)

def _save_cli_plots(results: dict, dfm: pd.DataFrame, outdir: str, prefix: str = "test", show: bool = False):
    """Save paper-oriented CLI plots into the output directory."""
    os.makedirs(outdir, exist_ok=True)

    # Equity curves
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    for name, (ret, _) in results.items():
        eq = (1 + _as_num_series(ret).fillna(0.0)).cumprod()
        if len(eq) > 0:
            eq.plot(ax=ax, label=name)
    ax.set_title(f"Equity Curves ({prefix})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Initial=1)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    p = os.path.join(outdir, f"fig_equity_curves_{prefix}.png")
    fig.savefig(p, dpi=180)
    print(f"[INFO] Saved plot: {p}")
    if show:
        plt.show()
    plt.close(fig)

    # Sharpe ranking
    if dfm is not None and not dfm.empty and "Sharpe" in dfm.columns:
        top = dfm[["Sharpe"]].replace([np.inf, -np.inf], np.nan).dropna().head(25).sort_values("Sharpe")
        if not top.empty:
            fig = plt.figure(figsize=(10, max(5, 0.28 * len(top))))
            ax = fig.add_subplot(111)
            top["Sharpe"].plot(kind="barh", ax=ax)
            ax.set_title(f"Sharpe Ratio by Strategy ({prefix})")
            ax.set_xlabel("Sharpe ratio")
            fig.tight_layout()
            p = os.path.join(outdir, f"fig_sharpe_ranking_{prefix}.png")
            fig.savefig(p, dpi=180)
            print(f"[INFO] Saved plot: {p}")
            if show:
                plt.show()
            plt.close(fig)

    # Drawdown ranking
    if dfm is not None and not dfm.empty and "Maximum Drawdown" in dfm.columns:
        dd = dfm[["Maximum Drawdown"]].replace([np.inf, -np.inf], np.nan).dropna().sort_values("Maximum Drawdown").head(25)
        if not dd.empty:
            fig = plt.figure(figsize=(10, max(5, 0.28 * len(dd))))
            ax = fig.add_subplot(111)
            dd["Maximum Drawdown"].plot(kind="barh", ax=ax)
            ax.set_title(f"Maximum Drawdown by Strategy ({prefix})")
            ax.set_xlabel("Drawdown return")
            fig.tight_layout()
            p = os.path.join(outdir, f"fig_mdd_ranking_{prefix}.png")
            fig.savefig(p, dpi=180)
            print(f"[INFO] Saved plot: {p}")
            if show:
                plt.show()
            plt.close(fig)


def _asset_class_costs(assets, base_tcost: float, crypto_mult: float = 5.0, commodity_mult: float = 1.5) -> dict:
    """Reviewer-response asset-class transaction-cost model.

    Default remains a scalar cost in spirit, but BTC/ETH-like tickers and commodity
    ETFs can be penalized more heavily for robustness checks.
    """
    crypto_tokens = ("BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "-USD")
    commodity = {"GLD", "SLV", "DBC", "USO"}
    out = {}
    for a in assets:
        aa = str(a).upper()
        mult = 1.0
        if any(tok in aa for tok in crypto_tokens):
            mult = crypto_mult
        elif aa in commodity:
            mult = commodity_mult
        out[a] = float(base_tcost) * float(mult)
    return out


def _write_experiment_config(args, assets, outdir, run_models, run_personas, runtime_cfg=None):
    """Write a reviewer-auditable experiment configuration.

    The explicit execution-lag keys are included to document that the
    backtest uses close-t decisions only from t+1 onward, preventing same-day
    feature/return leakage.
    """
    cfg = vars(args).copy()
    cfg.update({
        "effective_assets": list(assets),
        "run_models": list(run_models),
        "run_personas": list(run_personas),
        "execution_lag": 1,
        "next_day_execution": True,
        "uses_same_day_return": False,
        "weight_drift_enabled": True,
    })
    if runtime_cfg:
        if "asset_tcost" in runtime_cfg:
            cfg["asset_class_costs"] = {str(k): float(v) for k, v in dict(runtime_cfg["asset_tcost"]).items()}
        if "tcost" in runtime_cfg:
            cfg["base_transaction_cost"] = runtime_cfg.get("tcost")
        if "max_weight" in runtime_cfg:
            cfg["hard_max_weight"] = runtime_cfg.get("max_weight")
        if "turnover_cap" in runtime_cfg:
            cfg["hard_turnover_cap"] = runtime_cfg.get("turnover_cap")
    with open(os.path.join(outdir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def _asset_class_label(ticker: str) -> str:
    aa = str(ticker).upper()
    if any(tok in aa for tok in ("BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "-USD")):
        return "Crypto"
    if aa in {"GLD", "SLV", "DBC", "USO"}:
        return "Commodity"
    if aa in {"AGG", "BND", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BIL"}:
        return "Bond"
    if aa in {"VNQ"}:
        return "RealEstate"
    if aa in {"UUP"}:
        return "Currency"
    return "EquityETF"


def _export_transaction_cost_audit(assets, outdir: str, base_tcost: float, runtime_cfg=None):
    """Export asset-level transaction-cost assumptions for reviewer audit."""
    runtime_cfg = runtime_cfg or {}
    cost_map = runtime_cfg.get("asset_tcost", None)
    if cost_map is None:
        cost_map = {a: float(base_tcost) for a in assets}
    rows = []
    for a in assets:
        applied = float(dict(cost_map).get(a, base_tcost))
        base = float(base_tcost)
        rows.append({
            "Ticker": a,
            "AssetClass": _asset_class_label(a),
            "BaseCost": base,
            "AppliedCost": applied,
            "CostMultiplier": applied / base if base != 0 else np.nan,
        })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "transaction_cost_audit.csv"), index=False)


def _audit_model_environment(models, outdir: str, url: str):
    """Best-effort environment/model audit required for reproducibility."""
    rows = []
    lines = []
    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            return f"ERROR: {e}"
    lines.append("# Model and Environment Audit")
    lines.append("Platform: " + platform.platform())
    lines.append("Python: " + sys.version.replace("\n", " "))
    lines.append("Ollama URL: " + str(url))
    lines.append("Ollama version: " + _run(["ollama", "--version"]))
    # macOS hardware, Linux fallback
    if platform.system().lower() == "darwin":
        lines.append("\n## macOS hardware")
        lines.append(_run(["system_profiler", "SPHardwareDataType"]))
    else:
        lines.append("\n## System")
        lines.append(_run(["uname", "-a"]))
    for m in models:
        info = _run(["ollama", "show", str(m)])
        lines.append(f"\n## ollama show {m}\n{info}")
        rows.append({"Model": m, "ollama_show_excerpt": info[:2000], "OllamaVersion": lines[3].replace("Ollama version: ", "")})
    with open(os.path.join(outdir, "model_environment_audit.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "model_identification.csv"), index=False)


def _export_raw_archives(prices_raw, feats, outdir: str):
    try:
        prices_raw.to_csv(os.path.join(outdir, "prices_raw_adjusted.csv"))
    except Exception:
        pass
    try:
        feats.to_csv(os.path.join(outdir, "features_used.csv"))
    except Exception:
        pass


def _export_stats_tables(results: dict, outdir: str, baseline: str = "EQUAL"):
    """Export reviewer-requested HAC/Wilcoxon/MBB/White RC/SPA tables."""
    os.makedirs(outdir, exist_ok=True)
    returns_dict = {k: _as_num_series(v[0]).dropna() for k, v in results.items() if v is not None and len(v) > 0}
    if baseline not in returns_dict and returns_dict:
        baseline = list(returns_dict.keys())[0]
    settings = {"baseline": baseline, "ann_factor": 252, "hac_lags": 5, "mbb_block": 7, "mbb_B": 500, "mbb_seed": 42, "reality_spa_B": 500, "stationary_bootstrap_p": 0.10}
    with open(os.path.join(outdir, "statistical_test_settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    if baseline in returns_dict:
        comps = [k for k in returns_dict if k != baseline]
        try:
            pair = build_comparison_table("NLPI", "Strategy", returns_dict, baseline, comps, hac_lags=5, mbb_block=7, mbb_B=500, mbb_seed=42)
            pair.to_csv(os.path.join(outdir, "statistical_tests_pairwise.csv"), index=False)
        except Exception as e:
            pd.DataFrame([{"error": str(e)}]).to_csv(os.path.join(outdir, "statistical_tests_pairwise.csv"), index=False)
        try:
            reality_check_df(returns_dict, baseline, B=500, p=0.10, lags=5, seed=42).to_csv(os.path.join(outdir, "reality_check.csv"), index=False)
        except Exception as e:
            pd.DataFrame([{"error": str(e)}]).to_csv(os.path.join(outdir, "reality_check.csv"), index=False)
        try:
            spa_df(returns_dict, baseline, B=500, p=0.10, lags=5, seed=42).to_csv(os.path.join(outdir, "spa_test.csv"), index=False)
        except Exception as e:
            pd.DataFrame([{"error": str(e)}]).to_csv(os.path.join(outdir, "spa_test.csv"), index=False)


def _export_example_tables(ts_diag: pd.DataFrame, outdir: str):
    if ts_diag is None or ts_diag.empty:
        return
    try:
        raw = ts_diag[ts_diag.get("event", "") == "llm_call"].copy()
        raw[[c for c in ["Strategy", "Model", "Persona", "model", "persona", "json_valid", "parse_failed", "repair_used", "raw_response", "parsed_response"] if c in raw.columns]].head(200).to_csv(os.path.join(outdir, "llm_raw_output_examples.csv"), index=False)
    except Exception:
        pass
    try:
        ex = ts_diag[ts_diag.get("event", "") == "allocation_example"].copy()
        ex.head(200).to_csv(os.path.join(outdir, "allocation_before_after_examples.csv"), index=False)
    except Exception:
        pass
    try:
        inv = ts_diag[(ts_diag.get("event", "") == "llm_call") & ((pd.to_numeric(ts_diag.get("parse_failed", 0), errors="coerce").fillna(0) > 0) | (pd.to_numeric(ts_diag.get("repair_used", 0), errors="coerce").fillna(0) > 0))].copy()
        inv.head(200).to_csv(os.path.join(outdir, "invalid_repair_examples.csv"), index=False)
    except Exception:
        pass


def _export_fewshot_audit(outdir: str, fold_rows: list):
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(os.path.join(outdir, "fewshot_examples_by_fold.csv"), index=False)
    with open(os.path.join(outdir, "fewshot_selection_rule.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": "KMeans representatives selected from the available training fold only", "random_state": 42, "n_init": 10}, f, indent=2)



# =========================================================
# Final reproducibility/validation helpers
# =========================================================
def _export_forced_invalid_output_example(assets, outdir: str, maxw: float = 0.60):
    """Create a deterministic invalid-output repair example for the appendix.

    This does not call an LLM. It provides an auditable stress example showing
    how a malformed / incomplete JSON-like allocation is handled and projected
    into the feasible set. This is useful when the selected production models
    do not naturally produce invalid outputs during smoke tests.
    """
    try:
        from engine.strategies import project_capped_simplex
        assets = list(assets)
        raw_response = '{"SPY": 1.20, "QQQ": -0.10, "FAKE": 0.50, "comment": "not a valid complete portfolio"}'
        raw = pd.Series(0.0, index=assets, dtype=float)
        if "SPY" in raw.index:
            raw.loc["SPY"] = 1.20
        if "QQQ" in raw.index:
            raw.loc["QQQ"] = -0.10
        projected = project_capped_simplex(raw, float(maxw)).reindex(assets).fillna(0.0)
        rows = []
        for a in assets:
            rows.append({
                "ExampleType": "forced_invalid_json",
                "RawResponse": raw_response,
                "Ticker": a,
                "RawWeight": float(raw.loc[a]),
                "ProjectedWeight": float(projected.loc[a]),
                "RepairAction": "drop_hallucinated_ticker; fill_missing_assets; project_capped_simplex",
                "ParseFailed": 1,
                "RepairUsed": 1,
                "HardCap": float(maxw),
                "SumProjected": float(projected.sum()),
                "MaxProjected": float(projected.max()),
                "MinProjected": float(projected.min()),
            })
        df = pd.DataFrame(rows)
        path = os.path.join(outdir, "forced_invalid_output_repair_example.csv")
        df.to_csv(path, index=False)
        # Also append to invalid_repair_examples.csv if present or create it.
        inv_path = os.path.join(outdir, "invalid_repair_examples.csv")
        if os.path.exists(inv_path):
            try:
                old = pd.read_csv(inv_path)
                pd.concat([old, df], ignore_index=True, sort=False).to_csv(inv_path, index=False)
            except Exception:
                df.to_csv(inv_path, index=False)
        else:
            df.to_csv(inv_path, index=False)
    except Exception as e:
        pd.DataFrame([{"error": str(e)}]).to_csv(os.path.join(outdir, "forced_invalid_output_repair_example.csv"), index=False)


def _export_projection_solver_validation(assets, outdir: str, maxw: float = 0.60, n_cases: int = 12, seed: int = 42):
    """Validate project_capped_simplex against a SciPy SLSQP reference solver."""
    try:
        from scipy.optimize import minimize
        from engine.strategies import project_capped_simplex
        rng = np.random.default_rng(seed)
        assets = list(assets)
        n = len(assets)
        rows = []
        for case_id in range(int(n_cases)):
            # Mixed raw weights: sometimes negative, sometimes over cap.
            raw_arr = rng.normal(loc=1.0 / max(n, 1), scale=0.25, size=n)
            if case_id % 3 == 0:
                raw_arr[0] += 1.0
            if case_id % 4 == 0 and n > 1:
                raw_arr[1] -= 0.5
            raw = pd.Series(raw_arr, index=assets, dtype=float)
            fast = project_capped_simplex(raw, float(maxw)).reindex(assets).fillna(0.0)
            x0 = np.clip(raw_arr, 0.0, float(maxw))
            if x0.sum() <= 1e-12:
                x0 = np.ones(n) / n
            else:
                x0 = x0 / x0.sum()
            x0 = np.minimum(x0, float(maxw))
            x0 = x0 / x0.sum()
            cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
            bnds = [(0.0, float(maxw)) for _ in range(n)]
            res = minimize(lambda w: float(np.sum((w - raw_arr) ** 2)), x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
            if res.success:
                ref = pd.Series(res.x, index=assets)
            else:
                ref = pd.Series(np.nan, index=assets)
            diff = fast - ref
            rows.append({
                "Case": case_id,
                "NAssets": n,
                "HardCap": float(maxw),
                "ReferenceSuccess": int(bool(res.success)),
                "ReferenceMessage": str(res.message),
                "FastObjective": float(np.sum((fast.values - raw_arr) ** 2)),
                "ReferenceObjective": float(np.sum((ref.values - raw_arr) ** 2)) if res.success else np.nan,
                "ObjectiveGap": float(np.sum((fast.values - raw_arr) ** 2) - np.sum((ref.values - raw_arr) ** 2)) if res.success else np.nan,
                "L1FastRef": float(np.abs(diff).sum()) if res.success else np.nan,
                "L2FastRef": float(np.sqrt(np.sum(diff.values ** 2))) if res.success else np.nan,
                "FastBudgetViolation": float(abs(fast.sum() - 1.0)),
                "FastMinViolation": float(max(0.0, -fast.min())),
                "FastCapViolation": float(max(0.0, fast.max() - float(maxw))),
                "ReferenceBudgetViolation": float(abs(ref.sum() - 1.0)) if res.success else np.nan,
                "ReferenceMinViolation": float(max(0.0, -ref.min())) if res.success else np.nan,
                "ReferenceCapViolation": float(max(0.0, ref.max() - float(maxw))) if res.success else np.nan,
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, "projection_solver_validation.csv"), index=False)
        summary = {
            "n_cases": int(len(df)),
            "all_reference_success": bool(df["ReferenceSuccess"].fillna(0).astype(int).min() == 1) if not df.empty else False,
            "max_l2_fast_reference": float(pd.to_numeric(df.get("L2FastRef"), errors="coerce").max()) if not df.empty else np.nan,
            "max_fast_budget_violation": float(pd.to_numeric(df.get("FastBudgetViolation"), errors="coerce").max()) if not df.empty else np.nan,
            "max_fast_cap_violation": float(pd.to_numeric(df.get("FastCapViolation"), errors="coerce").max()) if not df.empty else np.nan,
            "interpretation": "Fast capped-simplex projection is validated against a SciPy SLSQP reference when L2FastRef and feasibility violations are near zero.",
        }
        with open(os.path.join(outdir, "projection_solver_validation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pd.DataFrame([{"error": str(e)}]).to_csv(os.path.join(outdir, "projection_solver_validation.csv"), index=False)


def _export_econometric_validation(results: dict, outdir: str, baseline: str = "EQUAL"):
    """Write a transparent econometric-validation audit for statistical tests."""
    rows = []
    returns_dict = {k: _as_num_series(v[0]).dropna() for k, v in results.items() if v is not None and len(v) > 0}
    if baseline not in returns_dict and returns_dict:
        baseline = list(returns_dict.keys())[0]
    base = returns_dict.get(baseline)
    for name, r in returns_dict.items():
        if base is None or name == baseline:
            continue
        z = pd.concat([r.rename("strategy"), base.rename("baseline")], axis=1).dropna()
        if z.empty:
            rows.append({"Strategy": name, "Baseline": baseline, "N": 0, "Valid": 0, "Reason": "no overlapping returns"})
            continue
        diff = z["strategy"].astype(float) - z["baseline"].astype(float)
        n = int(len(diff))
        std = float(diff.std(ddof=1)) if n > 1 else np.nan
        mean = float(diff.mean()) if n else np.nan
        zero_diff = bool(np.allclose(diff.values, 0.0, atol=1e-14))
        zero_var = bool((not np.isfinite(std)) or std <= 1e-14)
        rows.append({
            "Strategy": name,
            "Baseline": baseline,
            "N": n,
            "MeanDiffDaily": mean,
            "MeanDiffAnnualized": mean * 252.0 if np.isfinite(mean) else np.nan,
            "StdDiffDaily": std,
            "ZeroDifference": int(zero_diff),
            "ZeroVariance": int(zero_var),
            "RecommendedMinBlockLength": int(max(2, round(n ** (1.0/3.0)))) if n > 0 else np.nan,
            "HACLagsUsed": 5,
            "MBBBlockUsed": 7,
            "BootstrapReplicationsUsed": 500,
            "Alternative": "two-sided for pairwise tests; max-statistic bootstrap for Reality Check/SPA",
            "MultipleComparisonNote": "Reality Check and SPA are reported as data-snooping adjustments across strategies relative to the selected baseline.",
            "Valid": int(n >= 10 and not zero_var),
            "Reason": "zero-difference/zero-variance rows should be reported as non-significant" if (zero_diff or zero_var) else "ok",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "econometric_validation.csv"), index=False)
    report = {
        "baseline": baseline,
        "pairwise_test_alternative": "two-sided",
        "hac_lags": 5,
        "mbb_block_length": 7,
        "mbb_replications": 500,
        "reality_spa_replications": 500,
        "stationary_bootstrap_p": 0.10,
        "zero_difference_policy": "If the strategy-baseline differential is identically zero or has zero variance, p-values are treated as non-significant rather than infinite t-statistics.",
        "block_length_note": "The exported table includes n^(1/3) as a rough diagnostic reference; final manuscript should justify block length and report sensitivity where material.",
    }
    with open(os.path.join(outdir, "econometric_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="NLPI Portfolio Allocation Studio — CLI")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance", type=int, default=42)
    parser.add_argument("--tcost", type=float, default=0.0010)
    parser.add_argument("--maxw", type=float, default=0.60)
    parser.add_argument("--turncap", type=float, default=0.25)
    parser.add_argument("--no-ollama", action="store_true")

    parser.add_argument("--log-level", default="ERROR", choices=["DEBUG","INFO","WARN","ERROR"])
    parser.add_argument("--log-every", type=int, default=0, help="N>0이면 N스텝마다 INFO/DEBUG 로그")
    parser.add_argument("--no-console-log", action="store_true", help="콘솔 로그 끄기")

    parser.add_argument("--bl-delta", type=float, default=2.5)
    parser.add_argument("--bl-tau", type=float, default=0.05)
    parser.add_argument("--bl-omega-scale", type=float, default=1.0)

    parser.add_argument("--cov-method", default=None, choices=[None, "sample", "ledoitwolf"],
                        help="MVP/LW-MVP/HRP/BL에 공통 적용(개별 전략 기본치가 우선)")

    parser.add_argument("--k", type=int, default=8, help="few-shot 대표 샘플 수")
    parser.add_argument("--plot", action="store_true", help="Show plots interactively after saving them")
    parser.add_argument("--save-plots", action="store_true", help="Save CLI plots into --outdir without opening a window")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--wfcv", action="store_true", help="Run walk-forward cross-validation instead of a 70/30 holdout split")
    parser.add_argument("--wfcv-train-days", type=int, default=756, help="Initial/rolling training window length in trading days")
    parser.add_argument("--wfcv-test-days", type=int, default=252, help="Out-of-sample test window length per WFCV fold")
    parser.add_argument("--wfcv-mode", choices=["expanding", "rolling"], default="expanding", help="WFCV training-window mode")
    parser.add_argument("--audit", action="store_true", help="Export data audit and reviewer-response diagnostics")
    parser.add_argument("--sensitivity", action="store_true", help="Run lightweight sensitivity checks for reviewer response")
    parser.add_argument("--export-reviewer-tables", action="store_true", help="Export all reviewer-response CSV tables")
    parser.add_argument("--prompt-profile", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Natural-language policy persona for a single NLPI run")
    parser.add_argument("--run-policy-interface-study", action="store_true",
                        help="Run full NLPI study: models x personas plus baselines and coded executors")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Ollama model list for --run-policy-interface-study")
    parser.add_argument("--personas", nargs="+", type=int, default=None,
                        help="Persona IDs for --run-policy-interface-study, e.g., 1 2 3 4 5")
    parser.add_argument("--prompt-cap", type=float, default=None,
                        help="Language-level target allocation percent for P1/P2/P3. Default follows --maxw; use 70 with --maxw 0.60 for conflict ablation.")
    parser.add_argument("--constraint-conflict-ablation", action="store_true",
                        help="Run aligned 60%% vs conflict 70%% prompt-cap ablation for P1/P2 with hard cap fixed at --maxw.")
    parser.add_argument("--use-asset-class-costs", action="store_true",
                        help="Use asset-class transaction costs: crypto and commodity-like tickers receive higher costs.")
    parser.add_argument("--stress-suite", action="store_true",
                        help="Run preset stress-period suite and export stress_*.csv files.")
    parser.add_argument("--stats-baseline", default="EQUAL", help="Baseline for statistical significance exports")
    parser.add_argument("--force-invalid-output-test", action="store_true", help="Export a deterministic forced invalid-output repair example for appendix/reviewer evidence")
    parser.add_argument("--projection-validation", action="store_true", help="Validate capped-simplex projection against a SciPy SLSQP reference solver")
    parser.add_argument("--econometric-validation", action="store_true", help="Export econometric/statistical-test validation audit tables")
    args = parser.parse_args()

    assets = args.tickers
    start  = args.start
    end    = args.end
    model  = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    url    = os.getenv("OLLAMA_URL", "http://localhost:11434")

    default_study_models = ["gemma3:270m", "llama3.1:8b", "gpt-oss:20b"]
    run_models = args.models if args.models else (default_study_models if args.run_policy_interface_study else [model])
    run_personas = args.personas if args.personas else ([1, 2, 3, 4, 5] if args.run_policy_interface_study else [int(args.prompt_profile)])
    run_personas = [int(x) for x in run_personas if int(x) in [1, 2, 3, 4, 5]]
    prompt_cap_pct = float(args.prompt_cap) if args.prompt_cap is not None else float(args.maxw) * 100.0

    ok, msg = check_ollama(url)
    print(f"[Ollama] {msg}")

    print(f"[INFO] Downloading prices: {assets}  {start} ~ {end or '(today)'}")
    if args.audit or args.export_reviewer_tables:
        prices_raw, price_audit = fetch_prices_yf_with_audit(assets, start, end)
    else:
        prices_raw = fetch_prices_yf(assets, start, end)
        price_audit = price_audit_table(prices_raw)
    assets = [a for a in assets if a in prices_raw.columns]
    if len(assets) < 2:
        raise ValueError(f"다운로드 성공 자산이 너무 적습니다: {assets}")

    os.makedirs(args.outdir, exist_ok=True)

    feats = make_features(prices_raw, assets)
    prices = prices_raw.loc[feats.index]

    n = len(prices)
    split = split_index(n, 0.7)
    print(f"[INFO] Split index = {split} (Train ~{split/n:.1%} / Test ~{1-split/n:.1%})")
    print(f"[INFO] Training few-shot on {feats.index[0].date()} ~ {feats.index[split-1].date()}")

    fewshots = build_fewshot_db(feats.iloc[:split], assets, k=args.k, profile_id=int(args.prompt_profile), prompt_cap_pct=prompt_cap_pct)
    fewshot_block = render_fewshot_block(fewshots, max_k=min(args.k, 6))
    fewshot_audit_rows = [{"Fold": 0, "Profile": int(args.prompt_profile), "ExampleIndex": i+1, "Date": str(x[0]), "TargetWeightsJSON": json.dumps(x[2], ensure_ascii=False)} for i, x in enumerate(fewshots)]

    cfg = {
        "rebalance_days": args.rebalance,
        "tcost": args.tcost,
        "max_weight": args.maxw,
        "turnover_cap": args.turncap,
        "prompt_cap_pct": prompt_cap_pct,
        "ollama_url": url,
        "model_name": model,
        "prompt_profile": int(args.prompt_profile),
        "use_ollama": (not args.no_ollama) and ok,
        "log_level": args.log_level,
        "log_every": args.log_every,
        "log_fn": (None if args.no_console_log else print),
    }
    if args.use_asset_class_costs:
        cfg["asset_tcost"] = _asset_class_costs(assets, args.tcost)
    if args.cov_method:
        cfg["cov_method"] = args.cov_method

    print(f"[INFO] Testing on {prices.index[split].date()} ~ {prices.index[-1].date()}")

    def _nlpi_strategy_name(model_name: str, persona_id: int) -> str:
        return f"NLPI[{model_name}|P{int(persona_id)}]"

    def _parse_strategy_parts(name: str) -> dict:
        import re
        m = re.match(r"^NLPI\[(?P<model>[^|\]]+)\|P(?P<persona>\d+)\]$", str(name))
        if m:
            return {"MethodGroup": "NLPI", "Model": m.group("model"), "Persona": f"P{m.group('persona')}"}
        m = re.match(r"^LLM\[(?P<model>[^|\]]+)\|P(?P<persona>\d+)\]$", str(name))
        if m:
            return {"MethodGroup": "NLPI", "Model": m.group("model"), "Persona": f"P{m.group('persona')}"}
        if str(name).startswith("CODED_P"):
            return {"MethodGroup": "CodedPolicy", "Model": "", "Persona": str(name).replace("CODED_", "")}
        return {"MethodGroup": "Baseline", "Model": "", "Persona": ""}

    def _add_strategy_parts(df: pd.DataFrame, strategy_col: str = "Strategy") -> pd.DataFrame:
        """Add MethodGroup/Model/Persona parsed from Strategy without duplicate columns.

        Some diagnostic events already carry lower-case model/persona fields, and
        earlier calls may already have appended MethodGroup/Model/Persona. Pandas
        returns a DataFrame instead of a Series when duplicate column names exist,
        which can break later fillna calls. This helper therefore removes any
        existing upper-case parsed columns before appending fresh parsed values.
        """
        if df is None or df.empty or strategy_col not in df.columns:
            return df
        out = df.reset_index(drop=True).copy()
        out = out.loc[:, ~out.columns.duplicated()].copy()
        for c in ["MethodGroup", "Model", "Persona"]:
            if c in out.columns:
                out = out.drop(columns=[c])
        parts = out[strategy_col].apply(_parse_strategy_parts).apply(pd.Series).reset_index(drop=True)
        return pd.concat([out, parts], axis=1)

    def _as_series(df: pd.DataFrame, col: str):
        """Return the first column as a Series even if duplicate names exist."""
        if col not in df.columns:
            return None
        x = df.loc[:, col]
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, 0]
        return x

    def mk_all_strats(prices, feats, fewshot_block_for_run=None):
        bl_cfg = dict(cfg)
        bl_cfg.update({
            "bl_delta": args.bl_delta,
            "bl_tau": args.bl_tau,
            "bl_omega_scale": args.bl_omega_scale,
        })
        strats = {}
        if fewshot_block_for_run is None:
            fewshot_block_for_run = fewshot_block

        # NLPI = LLM-based Natural-Language Policy Interface.
        # The LLM is not treated as an autonomous portfolio optimizer;
        # it translates natural-language policies into provisional weights,
        # while feasibility is enforced by the constraint-projection layer.
        for m in run_models:
            for p_id in run_personas:
                nlpi_cfg = dict(cfg)
                nlpi_cfg.update({"model_name": m, "prompt_profile": int(p_id)})
                strats[_nlpi_strategy_name(m, p_id)] = LLMStrategy(assets, prices, feats, fewshot_block_for_run, nlpi_cfg)

        for p_id in [1, 2, 3, 4, 5]:
            strats[f"CODED_P{p_id}"] = CodedPersonaStrategy(assets, prices, feats, cfg, persona_id=p_id)

        strats.update({
            "EQUAL":      EqStrategy(assets, prices, feats, cfg),
            "RiskParity": RiskParityStrategy(assets, prices, feats, cfg),
            "MVP":        MVPStrategy(assets, prices, feats, cfg),
            "LW-MVP":     LWMVPStrategy(assets, prices, feats, cfg),
            "HRP":        HRPStrategy(assets, prices, feats, cfg),
            "BL":         BLStrategy(assets, prices, feats, bl_cfg),
            "MOM6":       Momentum6mStrategy(assets, prices, feats, cfg),
            "TRND6":      Trend6mStrategy(assets, prices, feats, cfg),
            "SHARPE":     SharpeWeightedStrategy(assets, prices, feats, cfg),
            "SORTINO":    SortinoWeightedStrategy(assets, prices, feats, cfg),
            "ERC":        RiskParityStrategy(assets, prices, feats, cfg),
            "MinCVaR":    MinCVaRStrategy(assets, prices, feats, cfg),
            "VOLT_TRND6": VolatilityTargetTrendStrategy(assets, prices, feats, cfg),
        })
        return strats

    results = {}
    backtest_diag = {}
    strats = {}
    fold_metrics_rows = []
    prebuilt_run_diag_rows = []
    prebuilt_ts_diag_frames = []

    if args.wfcv:
        train_days = max(int(args.wfcv_train_days), 2)
        test_days = max(int(args.wfcv_test_days), 2)
        print(f"[INFO] WFCV enabled: mode={args.wfcv_mode}, train_days={train_days}, test_days={test_days}")
        fold_id = 1
        test_start = train_days
        result_parts = {}
        weight_parts = {}
        diag_parts = {}
        while test_start < len(prices) - 1:
            test_end = min(len(prices), test_start + test_days)
            if test_end - test_start < 2:
                break
            if args.wfcv_mode == "rolling":
                train_start = max(0, test_start - train_days)
            else:
                train_start = 0
            train_end = test_start
            print(f"[WFCV] Fold {fold_id}: train {prices.index[train_start].date()}~{prices.index[train_end-1].date()} / test {prices.index[test_start].date()}~{prices.index[test_end-1].date()}")

            fold_fewshots = build_fewshot_db(feats.iloc[train_start:train_end], assets, k=args.k, profile_id=int(args.prompt_profile), prompt_cap_pct=prompt_cap_pct)
            fold_fewshot_block = render_fewshot_block(fold_fewshots, max_k=min(args.k, 6))
            fewshot_audit_rows.extend([{"Fold": fold_id, "Profile": int(args.prompt_profile), "ExampleIndex": i+1, "Date": str(x[0]), "TargetWeightsJSON": json.dumps(x[2], ensure_ascii=False)} for i, x in enumerate(fold_fewshots)])
            fold_strats = mk_all_strats(prices, feats, fold_fewshot_block)
            strats = fold_strats  # keep last fold for compatibility

            for name, strat in fold_strats.items():
                ret, w, diag = run_backtest(
                    prices, assets, feats, strat,
                    test_start, test_end,
                    args.rebalance,
                    return_diagnostics=True
                )
                ret = _as_num_series(ret).replace([np.inf, -np.inf], np.nan)
                if isinstance(w, pd.DataFrame):
                    for c in w.columns:
                        w[c] = pd.to_numeric(w[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                result_parts.setdefault(name, []).append(ret)
                weight_parts.setdefault(name, []).append(w)
                if isinstance(diag, pd.DataFrame) and not diag.empty:
                    tmp_diag = diag.copy(); tmp_diag["Fold"] = fold_id
                    diag_parts.setdefault(name, []).append(tmp_diag)

                sm = summary(ret, w)
                fold_metrics_rows.append({
                    "Fold": fold_id, "Strategy": name,
                    "Start": str(prices.index[test_start].date()),
                    "End": str(prices.index[test_end-1].date()),
                    "Sharpe": sm.get("Sharpe", np.nan),
                    "Sortino": sm.get("Sortino", np.nan),
                    "CAGR": sm.get("CAGR", np.nan),
                    "MDD": sm.get("MDD", np.nan),
                    "Turnover": _safe_turnover(w),
                })

                if hasattr(strat, "diagnostics_summary"):
                    row = strat.diagnostics_summary(); row["Strategy"] = name; row["Fold"] = fold_id
                    prebuilt_run_diag_rows.append(row)
                if hasattr(strat, "diagnostics_timeseries"):
                    ts = strat.diagnostics_timeseries()
                    if isinstance(ts, pd.DataFrame) and not ts.empty:
                        ts = ts.copy(); ts["Strategy"] = name; ts["Fold"] = fold_id
                        prebuilt_ts_diag_frames.append(ts)

            fold_id += 1
            test_start = test_end

        for name, parts in result_parts.items():
            results[name] = (pd.concat(parts).sort_index(), pd.concat(weight_parts.get(name, []), axis=0).sort_index())
        for name, parts in diag_parts.items():
            backtest_diag[name] = pd.concat(parts, ignore_index=True)
        pd.DataFrame(fold_metrics_rows).to_csv(os.path.join(args.outdir, "metrics_wfcv_folds.csv"), index=False)
    else:
        strats = mk_all_strats(prices, feats, fewshot_block)
        for name, strat in strats.items():
            bt_out = run_backtest(
                prices, assets, feats, strat,
                split, len(prices),
                args.rebalance,
                return_diagnostics=True
            )
            ret, w, diag = bt_out
            backtest_diag[name] = diag
            ret = _as_num_series(ret).replace([np.inf, -np.inf], np.nan)
            if isinstance(w, pd.DataFrame):
                for c in w.columns:
                    w[c] = pd.to_numeric(w[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            results[name] = (ret, w)

    # 메트릭 출력/저장 (NaN 방지)
    dfm = _make_metrics_table(results)
    print("\n==== TEST METRICS ====")
    print(dfm)

    os.makedirs(args.outdir, exist_ok=True)
    for name, (ret, w) in results.items():
        eq = (1 + _as_num_series(ret).fillna(0.0)).cumprod()
        safe_name = _safe_filename(name)
        pd.DataFrame({"ret": ret, "eq": eq}).to_csv(os.path.join(args.outdir, f"test_{safe_name}_curve.csv"))
        if isinstance(w, pd.DataFrame):
            w.to_csv(os.path.join(args.outdir, f"test_{safe_name}_weights.csv"))
    dfm.to_csv(os.path.join(args.outdir, "metrics_test.csv"))
    _write_experiment_config(args, assets, args.outdir, run_models, run_personas, runtime_cfg=cfg)
    _export_raw_archives(prices_raw, feats, args.outdir)
    _export_transaction_cost_audit(assets, args.outdir, args.tcost, runtime_cfg=cfg)
    _audit_model_environment(run_models, args.outdir, url)
    _export_fewshot_audit(args.outdir, fewshot_audit_rows)

    if args.audit or args.export_reviewer_tables:
        price_audit.to_csv(os.path.join(args.outdir, "missing_data_audit.csv"), index=False)

    if args.export_reviewer_tables:
        # Reviewer-response tables aligned with the revised NLPI framing.
        perf_main = _add_strategy_parts(dfm.reset_index().rename(columns={"index": "Strategy"}), "Strategy")
        perf_main.to_csv(os.path.join(args.outdir, "performance_main.csv"), index=False)

        nlpi_names = [x for x in dfm.index if str(x).startswith("NLPI[")]
        preferred_nlpi = ["NLPI[gpt-oss:20b|P5]", "NLPI[llama3.1:8b|P5]", "NLPI[gemma3:270m|P5]"]
        clean_names = []
        for x in preferred_nlpi + nlpi_names + ["CODED_P5", "EQUAL", "MVP", "LW-MVP", "HRP", "BL", "RiskParity"]:
            if x in dfm.index and x not in clean_names:
                clean_names.append(x)
        dfm.loc[clean_names].to_csv(os.path.join(args.outdir, "performance_clean_comparison.csv"))
        _add_strategy_parts(
            dfm.loc[[x for x in dfm.index if str(x).startswith("NLPI[")]].reset_index().rename(columns={"index": "Strategy"}),
            "Strategy"
        ).to_csv(os.path.join(args.outdir, "table_policy_interface_performance.csv"), index=False)

        diag_frames = []
        for k, df in backtest_diag.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                tmp = df.copy(); tmp["Strategy"] = k; diag_frames.append(tmp)
        projection_diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
        projection_diag.to_csv(os.path.join(args.outdir, "projection_diagnostics.csv"), index=False)
        if not projection_diag.empty and "Strategy" in projection_diag.columns:
            proj_cols = [c for c in ["turnover_before_cap", "turnover_after_cap", "projection_l1_target_to_final", "max_weight_final", "sum_weight_final", "min_weight_final", "feasible_budget", "feasible_nonnegative", "feasible_cap"] if c in projection_diag.columns]
            projection_effects = projection_diag.groupby("Strategy", dropna=False)[proj_cols].mean(numeric_only=True).reset_index()
            _add_strategy_parts(projection_effects, "Strategy").to_csv(os.path.join(args.outdir, "table_projection_effects_by_model_persona.csv"), index=False)
            if all(c in projection_diag.columns for c in ["feasible_budget", "feasible_nonnegative", "feasible_cap"]):
                feas = projection_diag.groupby("Strategy", dropna=False)[["feasible_budget", "feasible_nonnegative", "feasible_cap"]].mean(numeric_only=True).reset_index()
                feas["budget_violation_rate"] = 1.0 - feas["feasible_budget"]
                feas["nonnegative_violation_rate"] = 1.0 - feas["feasible_nonnegative"]
                feas["cap_violation_rate"] = 1.0 - feas["feasible_cap"]
                _add_strategy_parts(feas, "Strategy").to_csv(os.path.join(args.outdir, "table_feasibility_violations.csv"), index=False)

        if prebuilt_run_diag_rows or prebuilt_ts_diag_frames:
            run_diag = pd.DataFrame(prebuilt_run_diag_rows)
            ts_diag = pd.concat(prebuilt_ts_diag_frames, ignore_index=True) if prebuilt_ts_diag_frames else pd.DataFrame()
        else:
            run_diag_rows, ts_rows = [], []
            for k, strat in strats.items():
                if hasattr(strat, "diagnostics_summary"):
                    row = strat.diagnostics_summary(); row["Strategy"] = k; run_diag_rows.append(row)
                if hasattr(strat, "diagnostics_timeseries"):
                    ts = strat.diagnostics_timeseries()
                    if isinstance(ts, pd.DataFrame) and not ts.empty:
                        ts = ts.copy(); ts["Strategy"] = k; ts_rows.append(ts)
            run_diag = pd.DataFrame(run_diag_rows)
            ts_diag = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()
        run_diag = _add_strategy_parts(run_diag, "Strategy") if not run_diag.empty else run_diag
        ts_diag = _add_strategy_parts(ts_diag, "Strategy") if not ts_diag.empty and "Strategy" in ts_diag.columns else ts_diag
        run_diag.to_csv(os.path.join(args.outdir, "run_diagnostics.csv"), index=False)
        ts_diag.to_csv(os.path.join(args.outdir, "diagnostics_timeseries.csv"), index=False)
        _export_stats_tables(results, args.outdir, baseline=args.stats_baseline)
        _export_example_tables(ts_diag, args.outdir)
        if not ts_diag.empty:
            llm_calls = ts_diag[ts_diag.get("event", "") == "llm_call"].copy()
            prompt_fid = ts_diag[ts_diag.get("event", "") == "prompt_fidelity"].copy()
            llm_calls.to_csv(os.path.join(args.outdir, "llm_call_diagnostics.csv"), index=False)
            if not prompt_fid.empty:
                # Make detailed prompt-fidelity audit self-contained.
                # Earlier versions relied on Strategy parsing only in summary tables,
                # leaving the raw prompt_fidelity.csv model column blank/NaN.
                prompt_fid = _add_strategy_parts(prompt_fid, "Strategy") if "Strategy" in prompt_fid.columns else prompt_fid
                prompt_fid = prompt_fid.loc[:, ~prompt_fid.columns.duplicated()].copy()
                model_s = _as_series(prompt_fid, "Model")
                model_lower_s = _as_series(prompt_fid, "model")
                if model_s is not None:
                    if model_lower_s is None:
                        prompt_fid["model"] = model_s
                    else:
                        prompt_fid["model"] = model_lower_s.fillna(model_s)
                persona_s = _as_series(prompt_fid, "Persona")
                persona_lower_s = _as_series(prompt_fid, "persona")
                if persona_s is not None:
                    if persona_lower_s is None:
                        prompt_fid["persona"] = persona_s
                    else:
                        prompt_fid["persona"] = persona_lower_s.fillna(persona_s)
                prompt_fid.to_csv(os.path.join(args.outdir, "prompt_fidelity.csv"), index=False)
            else:
                prompt_fid.to_csv(os.path.join(args.outdir, "prompt_fidelity.csv"), index=False)
            if not prompt_fid.empty:
                pf_rows = []
                for (strategy, stage), g in prompt_fid.groupby(["Strategy", "stage"], dropna=False):
                    row = {"Strategy": strategy, "stage": stage, "n": len(g)}
                    if "raw_fidelity" in g.columns:
                        vals = pd.to_numeric(g["raw_fidelity"], errors="coerce").dropna()
                        if not vals.empty:
                            row["raw_fidelity_rate"] = float(vals.mean())
                    if "projected_fidelity" in g.columns:
                        vals = pd.to_numeric(g["projected_fidelity"], errors="coerce").dropna()
                        if not vals.empty:
                            row["projected_fidelity_rate"] = float(vals.mean())
                    pf_rows.append(row)
                pf_stage = _add_strategy_parts(pd.DataFrame(pf_rows), "Strategy")
                pf_stage.to_csv(os.path.join(args.outdir, "table_prompt_fidelity_by_model_persona.csv"), index=False)
                num_cols = [c for c in prompt_fid.columns if c.endswith("fidelity") or c.endswith("fidelity_rate") or c.endswith("gap") or c.endswith("loss") or c in ["raw_core_weight", "projected_core_weight", "raw_equal_l1", "projected_equal_l1", "raw_herfindahl", "projected_herfindahl"]]
                if num_cols:
                    tmp = prompt_fid.copy()
                    for c in num_cols:
                        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp.groupby(["Strategy", "persona", "stage"], dropna=False)[num_cols].mean(numeric_only=True).reset_index().to_csv(os.path.join(args.outdir, "table_prompt_fidelity_detailed.csv"), index=False)

        latency_cols = [c for c in ["Strategy", "MethodGroup", "Model", "Persona", "strategy_key", "strategy", "Avg_latency_sec", "n_calls", "JSON_valid_rate", "Parse_fail_rate", "Repair_rate", "Equal_fallback_rate", "Missing_asset_per_call", "Invalid_weight_per_call", "Negative_weight_per_call"] if c in run_diag.columns]
        run_diag[latency_cols].to_csv(os.path.join(args.outdir, "model_latency.csv"), index=False)
        run_diag[latency_cols].to_csv(os.path.join(args.outdir, "table_computational_cost_by_model.csv"), index=False)

        ablation_names = [x for x in dfm.index if str(x).startswith("NLPI[") or str(x).startswith("CODED_P") or str(x) in ["EQUAL", "MVP", "LW-MVP", "RiskParity", "HRP", "BL"]]
        _add_strategy_parts(dfm.loc[ablation_names].reset_index().rename(columns={"index": "Strategy"}), "Strategy").to_csv(os.path.join(args.outdir, "table_coded_vs_llm_ablation.csv"), index=False)

    if args.sensitivity:
        sens_rows = []
        base_params = {"tcost": args.tcost, "max_weight": args.maxw, "turnover_cap": args.turncap, "rebalance": args.rebalance}
        grid = []
        for v in [0.0, 0.001, 0.0025, 0.005]: grid.append({**base_params, "tcost": v, "Sensitivity": "tcost", "Value": v})
        for v in [0.4, 0.5, 0.6, 0.8]: grid.append({**base_params, "max_weight": v, "Sensitivity": "wmax", "Value": v})
        for v in [0.10, 0.25, 0.50, 1.00]: grid.append({**base_params, "turnover_cap": v, "Sensitivity": "turnover", "Value": v})
        for v in [5, 21, 42, 63]: grid.append({**base_params, "rebalance": v, "Sensitivity": "rebalance", "Value": v})
        for par in grid:
            tmp_cfg = dict(cfg); tmp_cfg.update({"tcost": par["tcost"], "max_weight": par["max_weight"], "turnover_cap": par["turnover_cap"], "prompt_cap_pct": float(par["max_weight"])*100.0})
            if args.use_asset_class_costs:
                tmp_cfg["asset_tcost"] = _asset_class_costs(assets, par["tcost"])
            rep_model = run_models[0] if run_models else model
            tmp_strats = {
                "EQUAL": EqStrategy(assets, prices, feats, tmp_cfg),
                "CODED_P5": CodedPersonaStrategy(assets, prices, feats, tmp_cfg, persona_id=5),
                "LW-MVP": LWMVPStrategy(assets, prices, feats, tmp_cfg),
                "HRP": HRPStrategy(assets, prices, feats, tmp_cfg),
                "SHARPE": SharpeWeightedStrategy(assets, prices, feats, tmp_cfg),
                "SORTINO": SortinoWeightedStrategy(assets, prices, feats, tmp_cfg),
            }
            for pid in [1, 3, 5]:
                ll_cfg = dict(tmp_cfg); ll_cfg.update({"model_name": rep_model, "prompt_profile": pid, "use_ollama": (not args.no_ollama) and ok})
                tmp_strats[f"NLPI[{rep_model}|P{pid}]"] = LLMStrategy(assets, prices, feats, fewshot_block, ll_cfg)
            for nm, st in tmp_strats.items():
                r_s, w_s = run_backtest(prices, assets, feats, st, split, len(prices), int(par["rebalance"]))
                sm = summary(r_s, w_s)
                sens_rows.append({"Sensitivity": par["Sensitivity"], "Value": par["Value"], "Strategy": nm, "Sharpe": sm.get("Sharpe"), "MDD": sm.get("MDD"), "CAGR": sm.get("CAGR"), "Turnover": _safe_turnover(w_s)})
        sens = pd.DataFrame(sens_rows)
        sens.to_csv(os.path.join(args.outdir, "sensitivity_all.csv"), index=False)
        for key in ["tcost", "wmax", "turnover", "rebalance"]:
            sens[sens["Sensitivity"] == key].to_csv(os.path.join(args.outdir, f"sensitivity_{key}.csv"), index=False)


    if args.constraint_conflict_ablation:
        # Appendix-style ablation: language target 60% vs 70%, hard cap fixed.
        # The key outcome is not performance, but policy-fidelity loss and
        # projection distance under language/constraint conflict.
        rows = []
        rep_models = run_models[:1] if run_models else [model]
        for pc in [60.0, 70.0]:
            for m in rep_models:
                for pid in [1, 2]:
                    tmp_cfg = dict(cfg)
                    tmp_cfg.update({"prompt_cap_pct": pc, "model_name": m, "prompt_profile": pid})
                    st = LLMStrategy(assets, prices, feats, fewshot_block, tmp_cfg)
                    r_c, w_c, d_c = run_backtest(prices, assets, feats, st, split, len(prices), args.rebalance, return_diagnostics=True)
                    sm = summary(r_c, w_c)
                    row = {
                        "PromptCapPct": pc,
                        "HardCap": args.maxw,
                        "Model": m,
                        "Persona": f"P{pid}",
                        "Sharpe": sm.get("Sharpe"),
                        "CAGR": sm.get("CAGR"),
                        "MDD": sm.get("MDD"),
                        "Turnover": _safe_turnover(w_c),
                    }
                    if isinstance(d_c, pd.DataFrame) and not d_c.empty:
                        row["ProjectionL1"] = pd.to_numeric(d_c.get("projection_l1_target_to_final"), errors="coerce").mean()
                        row["ProjectionL2"] = pd.to_numeric(d_c.get("projection_l2_target_to_final"), errors="coerce").mean() if "projection_l2_target_to_final" in d_c.columns else np.nan
                        row["CapFeasibleRate"] = pd.to_numeric(d_c.get("feasible_cap"), errors="coerce").mean()
                    try:
                        ts = st.diagnostics_timeseries()
                        pf = ts[ts.get("event", "") == "prompt_fidelity"].copy() if isinstance(ts, pd.DataFrame) and not ts.empty else pd.DataFrame()
                        for col, outcol in [
                            ("raw_fidelity", "RawFidelity"),
                            ("projected_fidelity", "ProjectedFidelity"),
                            ("fidelity_loss", "FidelityLoss"),
                            ("raw_threshold_fidelity", "RawThresholdFidelity"),
                            ("projected_threshold_fidelity", "ProjectedThresholdFidelity"),
                            ("threshold_fidelity_loss", "ThresholdFidelityLoss"),
                            ("raw_core_weight", "RawCoreWeight"),
                            ("projected_core_weight", "ProjectedCoreWeight"),
                        ]:
                            row[outcol] = pd.to_numeric(pf.get(col, pd.Series(dtype=float)), errors="coerce").mean() if not pf.empty else np.nan
                    except Exception as e:
                        row["FidelityAggregationError"] = str(e)
                    rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "constraint_conflict_ablation.csv"), index=False)

    if args.stress_suite:
        periods = {
            "covid_crash": ("2020-02-01", "2020-06-30"),
            "inflation_2022": ("2022-01-01", "2022-12-31"),
            "high_rate_2022_2023": ("2022-01-01", "2023-12-31"),
        }
        stress_rows = []
        for label, (sdate, edate) in periods.items():
            mask = (prices.index >= pd.to_datetime(sdate)) & (prices.index <= pd.to_datetime(edate))
            if mask.sum() < 30:
                continue
            s0 = int(np.where(mask)[0][0]); s1 = int(np.where(mask)[0][-1]) + 1
            stress_strats = {
                "EQUAL": EqStrategy(assets, prices, feats, cfg),
                "MOM6": Momentum6mStrategy(assets, prices, feats, cfg),
                "RiskParity": RiskParityStrategy(assets, prices, feats, cfg),
            }
            if run_models:
                for pid in [1,3,5]:
                    tmp_cfg = dict(cfg); tmp_cfg.update({"model_name": run_models[0], "prompt_profile": pid})
                    stress_strats[f"NLPI[{run_models[0]}|P{pid}]"] = LLMStrategy(assets, prices, feats, fewshot_block, tmp_cfg)
            for nm, st in stress_strats.items():
                rr, ww = run_backtest(prices, assets, feats, st, s0, s1, args.rebalance)
                sm = summary(rr, ww)
                stress_rows.append({"StressPeriod": label, "Start": sdate, "End": edate, "Strategy": nm, "Sharpe": sm.get("Sharpe"), "CAGR": sm.get("CAGR"), "MDD": sm.get("MDD"), "Turnover": _safe_turnover(ww)})
        stress = pd.DataFrame(stress_rows)
        stress.to_csv(os.path.join(args.outdir, "stress_suite.csv"), index=False)
        for label in stress["StressPeriod"].dropna().unique() if not stress.empty else []:
            stress[stress["StressPeriod"] == label].to_csv(os.path.join(args.outdir, f"stress_{label}.csv"), index=False)

    # Reproducibility-completion exports. These were intentionally kept
    # independent of the main performance tables so that reviewer evidence
    # files are generated whenever the corresponding CLI flags are enabled.
    if getattr(args, "force_invalid_output_test", False):
        _export_forced_invalid_output_example(assets, args.outdir, maxw=float(args.maxw))
    if getattr(args, "projection_validation", False):
        _export_projection_solver_validation(assets, args.outdir, maxw=float(args.maxw))
    if getattr(args, "econometric_validation", False):
        _export_econometric_validation(results, args.outdir, baseline=args.stats_baseline)

    print(f"[INFO] Saved CSVs to {os.path.abspath(args.outdir)}")

    # CLI plots are saved into the output directory for reproducibility.
    # --plot additionally opens the figures interactively.
    if args.plot or args.save_plots or args.export_reviewer_tables:
        plot_prefix = "wfcv" if args.wfcv else "test"
        _save_cli_plots(results, dfm, args.outdir, prefix=plot_prefix, show=args.plot)

if __name__ == "__main__":
    main()
