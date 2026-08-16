# q1_experiments/reference_policies.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd


def _feat(row: pd.Series, asset: str, suffix: str, default: float = 0.0) -> float:
    try:
        v = float(row.get(f"{asset}_{suffix}", default))
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _z(s: pd.Series) -> pd.Series:
    x = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sd = float(x.std(ddof=0))
    if sd <= 1e-12 or not np.isfinite(sd):
        return pd.Series(0.0, index=x.index)
    return (x - float(x.mean())) / sd


def _remainder_equal(tickers: List[str], core: str, cap: float) -> pd.Series:
    w = pd.Series(0.0, index=tickers, dtype=float)
    cap = max(0.0, min(float(cap), 1.0))
    if core not in w.index:
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    w.loc[core] = cap
    others = [a for a in tickers if a != core]
    rem = max(0.0, 1.0 - cap)
    if others:
        w.loc[others] = rem / len(others)
    return w


def _ranked_allocation(tickers: List[str], scores: pd.Series, top_weights=(0.40, 0.25, 0.15)) -> pd.Series:
    scores = scores.reindex(tickers).replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    order = list(scores.sort_values(ascending=False).index)
    w = pd.Series(0.0, index=tickers, dtype=float)
    used = 0.0
    for i, tw in enumerate(top_weights):
        if i < len(order):
            w.loc[order[i]] = float(tw)
            used += float(tw)
    rest = [a for a in tickers if w.loc[a] <= 0]
    rem = max(0.0, 1.0 - used)
    if rest:
        w.loc[rest] = rem / len(rest)
    s = float(w.sum())
    return w / s if s > 1e-12 else pd.Series(1.0 / len(tickers), index=tickers)


def _softmax_allocation(tickers: List[str], scores: pd.Series, temp: float = 0.75, cap: float = 0.30) -> pd.Series:
    scores = scores.reindex(tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = scores.values.astype(float)
    x = x - np.nanmax(x)
    e = np.exp(x / max(float(temp), 1e-6))
    if not np.isfinite(e).all() or e.sum() <= 1e-12:
        return pd.Series(1.0 / len(tickers), index=tickers)
    w = pd.Series(e / e.sum(), index=tickers)
    w = w.clip(upper=float(cap))
    return w / float(w.sum())


def reference_weights(policy_id: str, row: pd.Series, tickers: List[str], prompt_cap: float = 0.60) -> pd.Series:
    """Return deterministic reference weights for canonical P1-P5 or ladder L1-L5.

    These references are not claimed to be superior portfolios. They define the
    audit target against which natural-language-to-action fidelity is measured.
    """
    pid = str(policy_id).upper()
    tickers = list(tickers)
    cap = float(prompt_cap)

    if pid == "P1":
        core = max(tickers, key=lambda a: _feat(row, a, "r12m", -np.inf))
        return _remainder_equal(tickers, core, cap)
    if pid == "P2" or pid == "L1":
        core = min(tickers, key=lambda a: _feat(row, a, "vol3m", np.inf))
        return _remainder_equal(tickers, core, cap)
    if pid == "P3":
        core = min(tickers, key=lambda a: _feat(row, a, "r3m", np.inf))
        return _remainder_equal(tickers, core, cap)
    if pid == "P4":
        return pd.Series(1.0 / len(tickers), index=tickers)
    if pid == "P5":
        score = pd.Series({a: max(_feat(row, a, "r1m", 0.0) / max(_feat(row, a, "vol3m", 1e-12), 1e-12), 0.0) for a in tickers})
        return score / float(score.sum()) if float(score.sum()) > 1e-12 else pd.Series(1.0 / len(tickers), index=tickers)

    r12 = pd.Series({a: _feat(row, a, "r12m", 0.0) for a in tickers})
    r3 = pd.Series({a: _feat(row, a, "r3m", 0.0) for a in tickers})
    vol = pd.Series({a: _feat(row, a, "vol3m", 0.0) for a in tickers})
    dd = pd.Series({a: _feat(row, a, "mdd", 0.0) for a in tickers})

    if pid in {"P6", "L6"}:
        # Hybrid defensive-momentum reference: this is not a claim of optimality.
        # It is an auditable heuristic target for a complex natural-language policy.
        above_median_mom = r12 >= float(r12.median())
        severe_dd = dd <= -0.20
        unusually_strong_mom = r12 >= float(r12.quantile(0.90))
        below_median_vol = vol <= float(vol.median())
        eligible = above_median_mom & ((~severe_dd) | (unusually_strong_mom & below_median_vol))
        score = 0.75 * _z(r12) - 0.75 * _z(vol) + 0.50 * _z(dd)
        score = score.where(eligible, score - 1.0)
        # Allocate a defensive sleeve when cross-sectional volatility is high.
        cross_vol = float(row.get("cross_vol", vol.mean()))
        high_xvol = bool(cross_vol >= float(vol.quantile(0.75)))
        defensive_candidates = [a for a in tickers if a.upper() in {"AGG", "BND", "SHY", "IEF", "TIP", "BIL", "LQD"}]
        w = _softmax_allocation(tickers, score, temp=0.75, cap=0.40)
        if high_xvol and defensive_candidates:
            sleeve = 0.20
            w = w * (1.0 - sleeve)
            defensive = pd.Series(0.0, index=tickers)
            for a in defensive_candidates:
                defensive.loc[a] = sleeve / len(defensive_candidates)
            w = w + defensive
            w = w / float(w.sum())
        return w

    if pid == "L2":
        score = _z(r12) - _z(vol) + _z(dd)
        return _ranked_allocation(tickers, score)

    if pid == "L3":
        market_vol = float(vol.mean())
        # Relative decision-date classification: high/low regimes are better set by runner.
        # If no thresholds are supplied, use cross-sectional median-like fallback.
        if market_vol >= float(vol.quantile(0.75)):
            score = -_z(vol) + _z(dd)
        elif market_vol <= float(vol.quantile(0.25)):
            score = _z(r12) + 0.5 * _z(r3)
        else:
            score = _z(r12) - _z(vol) + _z(dd)
        return _ranked_allocation(tickers, score)

    if pid == "L4":
        severe = dd <= -0.20
        strong = r12 >= float(r12.quantile(0.75))
        low_vol = vol <= float(vol.median())
        eligible = (~severe) | (strong & low_vol)
        score = _z(r12) - _z(vol) + _z(dd)
        score = score.where(eligible, -1e6)
        return _ranked_allocation(tickers, score)

    if pid == "L5":
        score = -_z(vol) + 0.5 * _z(r12) + _z(dd)
        score = score + ((r12 >= float(r12.quantile(0.90))) & (vol <= float(vol.median()))).astype(float) * 0.5
        score = score - (dd <= -0.25).astype(float) * 1.0
        return _softmax_allocation(tickers, score, temp=0.75, cap=0.30)

    return pd.Series(1.0 / len(tickers), index=tickers)


def topk_overlap(a: pd.Series, b: pd.Series, k: int = 3) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.nan
    A = set(a.sort_values(ascending=False).head(k).index)
    B = set(b.sort_values(ascending=False).head(k).index)
    return float(len(A & B) / max(k, 1))
