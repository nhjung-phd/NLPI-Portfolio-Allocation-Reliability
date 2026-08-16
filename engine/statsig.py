# statsig.py
# -*- coding: utf-8 -*-
"""
Statistical significance utilities for time-series strategy comparisons.

This module provides:
- HAC/Newey–West mean-difference t-test
- Wilcoxon signed-rank test (two-sided)
- Jackknife Sharpe z-statistic (delete-1 jackknife variance)
- Moving-Block Bootstrap (two-sided) p-value for mean differences
- (Kept) Stationary/Circular bootstrap tests, Diebold–Mariano,
         Reality Check (White) and SPA (Hansen)

Convenience:
- build_comparison_table(...) to reproduce a table like:
  [Group | Algo | Comparator | N | Mean Diff (Ann.) | t_HAC | p_HAC |
   Wilcoxon p | JK z | MBB p_two | Stars]
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
from math import isnan
import warnings
from scipy.stats import norm, wilcoxon

# =========================================================
# Common helpers
# =========================================================

def stars_for_p(p: float) -> str:
    """Star markers by p-value: * (<0.10), ** (<0.05), *** (<0.01)."""
    try:
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""
    except Exception:
        return ""

def _align_pair(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    z = pd.concat([x, y], axis=1).dropna()
    if z.shape[1] != 2 or len(z) == 0:
        return np.array([]), np.array([])
    a = z.iloc[:,0].to_numpy(dtype=float)
    b = z.iloc[:,1].to_numpy(dtype=float)
    return a, b

def _autocov(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n: return 0.0
    mu = x.mean()
    return float(((x[:n-lag] - mu) * (x[lag:] - mu)).sum() / n)

def _hac_var_bartlett(x: np.ndarray, lags: int) -> float:
    """Newey–West HAC variance with Bartlett kernel."""
    n = len(x)
    if n < 2: return np.nan
    L = max(min(int(lags), n-1), 0)
    gamma0 = _autocov(x, 0)
    var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        g = _autocov(x, k)
        var += 2.0 * w * g
    return max(var, 0.0)

def _stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap indices with continuation prob (1-p)."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx

def _circular_block_sample(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    B = int(np.ceil(n / block_len))
    out = []
    for _ in range(B):
        s = rng.integers(0, n)
        blk = np.take(x, (np.arange(s, s + block_len) % n))
        out.append(blk)
    y = np.concatenate(out)[:n]
    return y

def _loss_from_returns(r: np.ndarray, power: int = 1) -> np.ndarray:
    # power=1: -r, power=2: -r^2, else: -|r|
    if power == 1:  return -r
    if power == 2:  return -(r**2)
    return -np.abs(r)

# =========================================================
# Single-model tests
# =========================================================

def hac_t_test_mean_diff(diff, lags: int = 5) -> Tuple[float, float]:
    """Two-sided H0: E[diff] = 0 using HAC/Newey–West variance."""
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    var_hac = _hac_var_bartlett(x, lags=lags)
    se = np.sqrt(var_hac / n) if var_hac > 0 else float("inf")
    t = mu / se if se > 0 else float("nan")
    if isnan(t): return t, float("nan")
    p = 2.0 * (1.0 - norm.cdf(abs(t)))
    return float(t), float(p)

def wilcoxon_signed_rank_p(diff) -> float:
    """
    Two-sided Wilcoxon signed-rank test for median(diff) = 0.
    Robust to degenerate cases that cause SciPy warnings (se=0).
    Returns p-value (float); NaN/degenerate -> 1.0 (no evidence).
    """
    x = np.asarray(diff, dtype=float)
    # 유효값만
    x = x[np.isfinite(x)]
    # 0 차이는 Wilcoxon 정의상 제외
    x = x[x != 0.0]

    n = len(x)
    if n < 10:
        # 표본이 너무 작으면 의미있는 검정 어렵게 보아 NaN 반환 (원하시면 1.0로 바꿔도 됨)
        return float("nan")

    # |diff|의 분산이 0이면(모두 같은 크기) 순위합 분산이 0 -> se=0 문제
    if np.nanstd(np.abs(x), ddof=1) == 0.0:
        return 1.0  # 증거 없음으로 보수적 처리

    # SciPy 내부 경고 억제
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            # zeros는 이미 제거했으니 zero_method='wilcox'
            stat = wilcoxon(
                x,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",  # n 작으면 exact, 크면 정규근사
            )
            p = float(stat.pvalue)
            # 드물게 NaN이 올 수 있으니 방어
            if not np.isfinite(p):
                return 1.0
            return p
        except Exception:
            # 어떤 이유로든 실패하면 보수적으로 p=1.0
            return 1.0

def jackknife_sharpe_z(returns, ann_factor: int = 252) -> Tuple[float, float]:
    """
    Delete-1 jackknife standard error for Sharpe ratio; returns (S_ann, z_jk).

    S_ann = mean(r)*sqrt(ann) / std(r)
    z_jk  = S_ann / se_jk_ann  (where se_jk_ann from jackknife variance)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or np.allclose(r.std(ddof=1), 0.0):
        return float("nan"), float("nan")

    def sharpe(a: np.ndarray) -> float:
        mu = a.mean()
        sd = a.std(ddof=1)
        if sd <= 0: return np.nan
        return (mu * np.sqrt(ann_factor)) / sd

    theta = sharpe(r)
    if not np.isfinite(theta):
        return float("nan"), float("nan")

    # delete-1 jackknife
    thetas = np.empty(n, dtype=float)
    for i in range(n):
        thetas[i] = sharpe(np.delete(r, i))

    theta_dot = thetas.mean()
    var_jk = (n - 1) / n * np.sum((thetas - theta_dot) ** 2)
    se_jk = np.sqrt(var_jk)
    if se_jk <= 0 or not np.isfinite(se_jk):
        return float(theta), float("nan")
    z = float(theta / se_jk)
    return float(theta), z

def moving_block_bootstrap_p(diff,
                             block_len: int = 7,
                             B: int = 2000,
                             seed: Optional[int] = None,
                             two_sided: bool = True) -> float:
    """
    Moving-Block Bootstrap p-value for H0: mean(diff)=0 (centered bootstrap).
    Returns p_two if two_sided else one-sided (greater).
    """
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5 or B < 200 or block_len < 2:
        return float("nan")
    mu_obs = float(x.mean())
    x0 = x - mu_obs
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        y = _circular_block_sample(x0, block_len=block_len, rng=rng)
        boots[b] = float(y.mean())
    if two_sided:
        p = float((np.abs(boots) >= abs(mu_obs)).mean())
    else:
        p = float((boots >= mu_obs).mean())
    return p

# =========================================================
# DM / Reality Check / SPA (kept from earlier)
# =========================================================

def diebold_mariano(x, y, h: int = 1, power: int = 1, lags: Optional[int] = None) -> Tuple[float, float]:
    """DM test using loss differential d_t = l_x - l_y derived from returns x,y."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    lx = _loss_from_returns(x, power=power)
    ly = _loss_from_returns(y, power=power)
    d = lx - ly
    mu = float(d.mean())
    L = max(h - 1, 0) if lags is None else int(lags)
    var_hac = _hac_var_bartlett(d, lags=L)
    se = np.sqrt(var_hac / n) if var_hac > 0 else float("inf")
    t = mu / se if se > 0 else float("nan")
    if isnan(t): return t, float("nan")
    p = 2.0 * (1.0 - norm.cdf(abs(t)))
    return float(t), float(p)

def _pairwise_diffs(returns_dict: Dict[str, pd.Series], baseline: str) -> Dict[str, np.ndarray]:
    if baseline not in returns_dict:
        return {}
    base = returns_dict[baseline]
    out = {}
    for name, r in returns_dict.items():
        if name == baseline: 
            continue
        a, b = _align_pair(r, base)
        if len(a) and len(b):
            out[name] = a - b
    return out

def _studentize(x: np.ndarray, lags: int = 5) -> Tuple[float, float]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mu = float(np.nanmean(x))
    var = _hac_var_bartlett(np.asarray(x, dtype=float), lags=lags)
    sd = float(np.sqrt(max(var, 0.0) / n)) if var > 0 else float("inf")
    return mu, sd



def _safe_t_like(mu: float, sd: float, eps: float = 1e-12) -> float:
    """Return a finite, reviewer-safe t-like statistic when variance is zero.

    If both the mean difference and its estimated standard error are numerically
    zero, the strategy is indistinguishable from the baseline, so t=0 and p=1
    are appropriate. If the mean is non-zero but the variance is degenerate,
    return a signed large finite value rather than inf to avoid misleading
    zero-difference significance artifacts.
    """
    try:
        mu = float(mu); sd = float(sd)
    except Exception:
        return float("nan")
    if not np.isfinite(mu):
        return float("nan")
    if (not np.isfinite(sd)) or sd <= eps:
        if abs(mu) <= eps:
            return 0.0
        return float(np.sign(mu) * 1e6)
    return float(mu / sd)

def reality_check_df(returns_dict: Dict[str, pd.Series],
                     baseline: str,
                     B: int = 2000,
                     p: float = 0.1,
                     studentize: bool = True,
                     lags: int = 5,
                     seed: Optional[int] = None) -> pd.DataFrame:
    """White's Reality Check with stationary bootstrap (family-wise control)."""
    diffs = _pairwise_diffs(returns_dict, baseline)
    if not diffs:
        return pd.DataFrame(columns=["Strategy","MeanDiff","t_like","p","Stars"])
    names = list(diffs.keys())
    obs_mu = []; obs_t = []
    for nm in names:
        mu, sd = _studentize(diffs[nm], lags=lags) if studentize else (float(np.mean(diffs[nm])), float(np.std(diffs[nm], ddof=1)/np.sqrt(len(diffs[nm]))))
        t_like = _safe_t_like(mu, sd)
        obs_mu.append(mu); obs_t.append(t_like)
    obs_mu = np.array(obs_mu, dtype=float); obs_t = np.array(obs_t, dtype=float)

    X0 = {nm: diffs[nm] - np.mean(diffs[nm]) for nm in names}
    n = len(next(iter(X0.values()))); rng = np.random.default_rng(seed)

    boot_max = np.empty(B, dtype=float)
    for b in range(B):
        t_vals = []
        idx = _stationary_bootstrap_indices(n, p, rng)
        for nm in names:
            y = X0[nm][idx]
            mu_b, sd_b = _studentize(y, lags=lags) if studentize else (float(np.mean(y)), float(np.std(y, ddof=1)/np.sqrt(len(y))))
            t_vals.append(mu_b / sd_b if (sd_b and sd_b>0 and np.isfinite(sd_b)) else 0.0)
        boot_max[b] = np.max(t_vals)

    pvals = [(boot_max >= t0).mean() for t0 in obs_t]
    df = pd.DataFrame({"Strategy": names, "MeanDiff": obs_mu, "t_like": obs_t, "p": pvals})
    df["Stars"] = [stars_for_p(p) for p in df["p"]]
    return df.sort_values("p").reset_index(drop=True)

def spa_df(returns_dict: Dict[str, pd.Series],
           baseline: str,
           B: int = 2000,
           p: float = 0.1,
           studentize: bool = True,
           lags: int = 5,
           seed: Optional[int] = None) -> pd.DataFrame:
    """Hansen's SPA (simplified) with stationary bootstrap."""
    diffs = _pairwise_diffs(returns_dict, baseline)
    if not diffs:
        return pd.DataFrame(columns=["Strategy","MeanDiff","t_like","p","Stars"])
    names = list(diffs.keys())

    obs_mu = []; obs_t = []
    for nm in names:
        mu, sd = _studentize(diffs[nm], lags=lags) if studentize else (float(np.mean(diffs[nm])), float(np.std(diffs[nm], ddof=1)/np.sqrt(len(diffs[nm]))))
        t_like = _safe_t_like(mu, sd)
        obs_mu.append(mu); obs_t.append(max(_safe_t_like(mu, sd), 0.0))
    obs_mu = np.array(obs_mu, dtype=float); obs_t = np.array(obs_t, dtype=float)

    X0 = {nm: diffs[nm] - np.mean(diffs[nm]) for nm in names}
    n = len(next(iter(X0.values()))); rng = np.random.default_rng(seed)

    boot_sup = np.empty(B, dtype=float)
    for b in range(B):
        t_vals = []
        idx = _stationary_bootstrap_indices(n, p, rng)
        for nm in names:
            y = X0[nm][idx]
            mu_b, sd_b = _studentize(y, lags=lags) if studentize else (float(np.mean(y)), float(np.std(y, ddof=1)/np.sqrt(len(y))))
            t_b = mu_b / sd_b if (sd_b and sd_b>0 and np.isfinite(sd_b)) else 0.0
            t_vals.append(max(t_b, 0.0))
        boot_sup[b] = np.max(t_vals)

    pvals = [(boot_sup >= t0).mean() for t0 in obs_t]
    df = pd.DataFrame({"Strategy": names, "MeanDiff": obs_mu, "t_like": obs_t, "p": pvals})
    df["Stars"] = [stars_for_p(p) for p in df["p"]]
    return df.sort_values("p").reset_index(drop=True)

# =========================================================
# Table builder (S2-style) for one baseline vs many comparators
# =========================================================

def _annualize_mean(x: np.ndarray, ann_factor: int) -> float:
    return float(np.nanmean(x) * ann_factor)

def build_comparison_table(group: str,
                           algo: str,
                           returns_dict: Dict[str, pd.Series],
                           baseline: str,
                           comparators: Optional[List[str]] = None,
                           ann_factor: int = 252,
                           hac_lags: int = 5,
                           mbb_block: int = 7,
                           mbb_B: int = 2000,
                           mbb_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Construct a comparison table similar to the example:

    Columns:
    - Group, Algo, Comparator, N
    - Mean Diff (Ann.)           [ann_factor * mean(strat - base)]
    - t_HAC, p_HAC               [Newey–West mean-diff test]
    - Wilcoxon p                 [signed-rank]
    - JK z                       [jackknife Sharpe z for (strat - base)]
    - MBB p_two                  [moving-block bootstrap p-value]
    - Stars                      [*, **, *** by p_HAC]

    Notes:
    - All tests are on the *difference series* diff_t = r_strat_t - r_base_t.
    - JK z is computed on the diff series' Sharpe (annualized) via delete‑1 jackknife.
    """
    if baseline not in returns_dict:
        raise ValueError("baseline not found in returns_dict")
    if comparators is None:
        comparators = [k for k in returns_dict.keys() if k != baseline]

    rows = []
    base = returns_dict[baseline]
    for comp in comparators:
        if comp == baseline or comp not in returns_dict: 
            continue
        a, b = _align_pair(returns_dict[comp], base)
        if len(a) == 0:
            continue
        diff = a - b
        n = len(diff)

        # Mean diff (annualized)
        mean_ann = _annualize_mean(diff, ann_factor=ann_factor)

        # HAC t and p
        t_hac, p_hac = hac_t_test_mean_diff(diff, lags=hac_lags)

        # Wilcoxon p
        p_w = wilcoxon_signed_rank_p(diff)

        # JK Sharpe z on diff series
        S_ann, z_jk = jackknife_sharpe_z(diff, ann_factor=ann_factor)

        # MBB p_two
        p_mbb = moving_block_bootstrap_p(diff, block_len=mbb_block, B=mbb_B, seed=mbb_seed, two_sided=True)

        rows.append({
            "Group": group,
            "Algo": algo,
            "Comparator": comp,
            "N": int(n),
            "Mean Diff (Ann.)": mean_ann,
            "t_HAC": float(t_hac),
            "p_HAC": float(p_hac),
            "Wilcoxon p": float(p_w),
            "JK z": float(z_jk),
            "MBB p_two": float(p_mbb),
            "Stars": stars_for_p(p_hac)
        })

    df = pd.DataFrame(rows)
    # Nice ordering like the example
    cols = ["Group","Algo","Comparator","N","Mean Diff (Ann.)","t_HAC","p_HAC","Wilcoxon p","JK z","MBB p_two","Stars"]
    if not df.empty:
        df = df[cols]
    return df

# --- Compatibility wrapper for GUI expecting build_significance_table ---
def build_significance_table(returns_dict: Dict[str, pd.Series],
                             baseline: Optional[str] = None,
                             comparators: Optional[List[str]] = None,
                             ann_factor: int = 252,
                             hac_lags: int = 5,
                             mbb_block: int = 7,
                             mbb_B: int = 2000,
                             mbb_seed: Optional[int] = None,
                             # ← 추가: GUI 호환용 별칭/잡다한 키 흡수
                             benchmark_key: Optional[str] = None,
                             group: str = "Default",
                             algo: str = "Strategy",
                             group_name: Optional[str] = None,
                             algo_name: Optional[str] = None,
                             **kwargs) -> pd.DataFrame:
    """
    Thin wrapper so GUI can call a canonical name with various legacy keys.
    Accepts:
      - baseline or benchmark_key (alias)
      - group/algo or group_name/algo_name (alias)
      - ignores any extra **kwargs coming from the GUI
    """
    # alias 매핑
    if baseline is None:
        baseline = benchmark_key
    if group_name:
        group = group_name
    if algo_name:
        algo = algo_name

    if baseline is None:
        raise ValueError("baseline (or benchmark_key) must be provided")

    return build_comparison_table(group=group,
                                  algo=algo,
                                  returns_dict=returns_dict,
                                  baseline=baseline,
                                  comparators=comparators,
                                  ann_factor=ann_factor,
                                  hac_lags=hac_lags,
                                  mbb_block=mbb_block,
                                  mbb_B=mbb_B,
                                  mbb_seed=mbb_seed)

