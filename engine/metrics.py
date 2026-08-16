# -*- coding: utf-8 -*-
"""
Portfolio performance metrics.

Exposes:
    summary(returns: pd.Series, weights: pd.DataFrame | None) -> dict
Keys produced:
    CAGR, Sharpe, Sortino, Vol, MDD, Calmar, AvgTurnover, Terminal
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def _cagr(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    total = float(eq.iloc[-1] / eq.iloc[0]) if eq.iloc[0] != 0 else 0.0
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    return (total ** (1.0 / years) - 1.0) if total > 0 else 0.0

def _sharpe(rets: pd.Series, rf: float = 0.0) -> float:
    if rets.empty:
        return 0.0
    mu = rets.mean() - rf / TRADING_DAYS
    sd = rets.std()
    if sd <= 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * mu / sd)

def _sortino(rets: pd.Series, rf: float = 0.0) -> float:
    if rets.empty:
        return 0.0
    dr = rets.copy()
    dr[dr > 0] = 0.0
    dd = dr.std()
    mu = rets.mean() - rf / TRADING_DAYS
    if dd <= 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * mu / dd)

def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def _avg_turnover(weights: pd.DataFrame) -> float:
    if weights is None or weights.empty or len(weights) < 2:
        return float("nan")
    tw = (weights.diff().abs().sum(axis=1)).dropna()
    return float(tw.mean())

def summary(rets: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    """
    Compute key metrics from daily returns (and optional weights).
    """
    rets = rets.dropna()
    eq = (1.0 + rets).cumprod()
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    cagr = _cagr(eq)
    mdd = _max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("inf")
    return {
        "CAGR": cagr,
        "Sharpe": _sharpe(rets),
        "Sortino": _sortino(rets),
        "Vol": vol,
        "MDD": mdd,
        "Calmar": calmar,
        "AvgTurnover": _avg_turnover(weights),
        "Terminal": float(eq.iloc[-1]) if not eq.empty else 1.0,
    }
