# -*- coding: utf-8 -*-
"""
Generic backtest runner used by the GUI/CLI.

Reviewer-response version:
- Enforces an execution lag: weights decided at close t are applied from t+1.
- Maintains market-drifted weights between rebalance dates.
- Computes turnover against drifted end-of-day weights, not stale target weights.
- Exports feasibility and leakage diagnostics when requested.
"""

from __future__ import annotations
from typing import Tuple, List, Callable, Optional, Iterable
import pandas as pd
import numpy as np

from .strategies import Strategy


def _trade_cost(prev_w: pd.Series, new_w: pd.Series, tcost) -> float:
    """Proportional transaction cost (negative).

    tcost can be a scalar or an asset-level mapping/Series. When a mapping is
    supplied, turnover is charged asset by asset, enabling ETF/crypto/liquidity
    cost sensitivity without changing strategy code.
    """
    if prev_w is None or getattr(prev_w, "empty", True):
        return 0.0
    delta = (new_w - prev_w).abs().astype(float)
    if isinstance(tcost, (dict, pd.Series)):
        c = pd.Series(tcost).reindex(delta.index).fillna(0.0).astype(float).abs()
        cost = float((delta * c).sum())
    else:
        cost = abs(float(tcost)) * float(delta.sum())
    if cost <= 1e-12:
        return 0.0
    return -cost


def _safe_date_str(dt) -> str:
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            return str(getattr(dt, "date", dt))
        except Exception:
            return str(dt)


def _drift_weights(prev_w: pd.Series, r_t: pd.Series, tickers: List[str]) -> tuple[pd.Series, float]:
    """Update beginning-of-day weights after same-day asset returns.

    If w_{t-1} is the portfolio weight after the previous close, then before any
    rebalance at close t the weight is w^-_t = w_{t-1}(1+r_t)/(1+w_{t-1}'r_t).
    """
    if prev_w is None or getattr(prev_w, "empty", True):
        z = pd.Series(0.0, index=tickers, dtype=float)
        return z, 0.0
    w0 = prev_w.reindex(tickers).fillna(0.0).astype(float)
    rr = r_t.reindex(tickers).fillna(0.0).astype(float)
    gross = float((w0 * rr).sum())
    denom = 1.0 + gross
    if denom <= 1e-12 or not np.isfinite(denom):
        # Conservative fallback: keep previous weights if numerical drift fails.
        return w0, gross
    wd = w0 * (1.0 + rr) / denom
    wd = wd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = float(wd.sum())
    if s > 1e-12:
        wd = wd / s
    return wd.reindex(tickers).fillna(0.0), gross


def run_backtest(
    prices: pd.DataFrame,
    tickers: List[str],
    features: pd.DataFrame,
    strat: Strategy,
    start: int,
    end: int,
    rebalance_days: int,
    progress_cb: Optional[Callable[[int, int, pd.Timestamp], None]] = None,
    cancel_event: Optional[object] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    rebalance_calendar: Optional[Iterable[pd.Timestamp]] = None,
    apply_warmup_offset: bool = True,
    return_diagnostics: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run a long-only backtest for a strategy.

    Important timing convention:
    - Features at row t are interpreted as being known at the close of t.
    - A target generated at row t is executed at the close of t.
    - The new weights therefore affect returns from t+1 onward.
    This removes same-day feature/return leakage.
    """
    tickers = list(tickers)
    if not tickers:
        idx = prices.index[start:end]
        empty_ret = pd.Series(dtype=float, index=idx)
        empty_w = pd.DataFrame(columns=tickers, index=idx)
        return (empty_ret, empty_w, pd.DataFrame()) if return_diagnostics else (empty_ret, empty_w)

    # Strategy-tickers alignment sanity.
    try:
        if getattr(strat, "tickers", None) is not None:
            strat.tickers = list(tickers)
    except Exception:
        pass

    # Warmup: only shift full-sample backtests starting at zero. In WFCV/test
    # windows start>0, target_weights receives absolute indices and can use
    # the full historical panel, so truncating the test fold would be wrong.
    warmup = 0
    if apply_warmup_offset:
        for key in ("lookback",):
            if hasattr(strat, key):
                try:
                    warmup = max(warmup, int(getattr(strat, key)))
                except Exception:
                    pass
        try:
            warmup = max(warmup, int(getattr(strat, "cfg", {}).get("lookback", 0)))
        except Exception:
            pass
        warmup = max(0, warmup)
        if warmup > 0 and start == 0:
            start = min(end, start + warmup)
        else:
            warmup = 0

    px = prices[tickers].iloc[start:end].copy()
    idx = px.index
    n = len(idx)
    if n < 2:
        port_ret = pd.Series(0.0, index=idx)
        w_mat = pd.DataFrame(0.0, index=idx, columns=tickers)
        return (port_ret, w_mat, pd.DataFrame()) if return_diagnostics else (port_ret, w_mat)

    rets = (px.ffill().pct_change(fill_method=None)
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0))
    w_mat = pd.DataFrame(0.0, index=idx, columns=tickers)
    port_ret = pd.Series(0.0, index=idx)
    diag_records = []

    calendar_mask = None
    if rebalance_calendar is not None:
        cal_set = set(pd.to_datetime(list(rebalance_calendar)))
        calendar_mask = idx.isin(cal_set)
    step = max(int(rebalance_days) if rebalance_days is not None else 1, 1)

    cfg = getattr(strat, "cfg", {}) or {}
    tcost = cfg.get("asset_tcost", cfg.get("tcost", 0.0))
    maxw  = float(cfg.get("max_weight", 1.0))
    turn  = float(cfg.get("turnover_cap", 1.0))
    if log_fn is None:
        log_fn = cfg.get("log_fn", None)

    # prev_w is the executable weight vector after the previous close. It is
    # applied to today's return and then drifted before a possible close-t rebalance.
    prev_w = pd.Series(0.0, index=tickers, dtype=float)
    has_position = False

    for i, dt in enumerate(idx):
        abs_i = start + i
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            if log_fn:
                log_fn("Backtest canceled by user.")
            break

        # 1) Apply weights decided at the previous close to today's return.
        if has_position:
            drifted_w, gross_ret = _drift_weights(prev_w, rets.loc[dt], tickers)
        else:
            drifted_w = pd.Series(0.0, index=tickers, dtype=float)
            gross_ret = 0.0

        # 2) Decide whether to rebalance at the close of today.
        if calendar_mask is not None:
            do_rebalance = bool(calendar_mask[i]) and (i >= warmup)
        else:
            do_rebalance = (i >= warmup) and ((i - warmup) % step == 0)

        fee = 0.0
        final_w = drifted_w.copy()
        if do_rebalance:
            w_raw = strat.target_weights(abs_i).reindex(tickers).fillna(0.0)
            w_tgt = strat.clamp_and_normalize(w_raw)
            if not has_position:
                final_w = w_tgt.copy()
                fee = 0.0  # initial entry cost is reported as zero by convention
                turnover_before = float("nan")
                turnover_after = float("nan")
            else:
                final_w = strat.apply_turnover_cap(drifted_w, w_tgt)
                fee = _trade_cost(drifted_w, final_w, tcost)
                turnover_before = float((w_tgt - drifted_w).abs().sum())
                turnover_after = float((final_w - drifted_w).abs().sum())
            has_position = True

            try:
                diag_records.append({
                    "date": _safe_date_str(dt),
                    "decision_date": _safe_date_str(dt),
                    "execution_effective_from": _safe_date_str(idx[i + 1]) if i + 1 < len(idx) else "END",
                    "execution_lag_days": 1,
                    "uses_same_day_return": 0,
                    "weight_drift_applied": 1,
                    "turnover_before_cap": turnover_before,
                    "turnover_after_cap": turnover_after,
                    "projection_l1_target_to_final": float((w_tgt - final_w).abs().sum()),
                    "projection_l2_target_to_final": float(np.sqrt(((w_tgt - final_w) ** 2).sum())),
                    "max_weight_final": float(final_w.max()),
                    "sum_weight_final": float(final_w.sum()),
                    "min_weight_final": float(final_w.min()),
                    "feasible_budget": int(abs(float(final_w.sum()) - 1.0) <= 1e-6),
                    "feasible_nonnegative": int(float(final_w.min()) >= -1e-10),
                    "feasible_cap": int(float(final_w.max()) <= maxw + 1e-8),
                    "turnover_cap": turn,
                    "max_weight_cap": maxw,
                    "fee": float(fee),
                    "gross_return_before_fee": float(gross_ret),
                })
            except Exception:
                pass
            if log_fn:
                # tcost may be a scalar or an asset-level dict/Series. Avoid numeric formatting
                # for mappings used by --use-asset-class-costs.
                try:
                    if isinstance(tcost, (dict, pd.Series)):
                        vals = pd.Series(tcost).astype(float)
                        tcost_label = f"asset-class[min={vals.min():.4f}, max={vals.max():.4f}]"
                    else:
                        tcost_label = f"{float(tcost):.4f}"
                except Exception:
                    tcost_label = str(type(tcost).__name__)
                log_fn(f"Rebalanced on {_safe_date_str(dt)} (effective next day; max_w={maxw:.2f}, turn_cap={turn:.2f}, tcost={tcost_label})")

        # 3) Return for today is based on previous close weights; rebalance fee is
        # charged at today's close and included in today's net return.
        port_ret.iloc[i] = float(gross_ret + fee)
        w_mat.loc[dt] = final_w
        prev_w = final_w

        if progress_cb is not None:
            try:
                progress_cb(abs_i, end - start, dt)
            except Exception:
                pass

    if return_diagnostics:
        return port_ret, w_mat, pd.DataFrame(diag_records)
    return port_ret, w_mat
