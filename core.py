# core.py
# -*- coding: utf-8 -*-
"""
Core utilities:
- fetch_prices_yf(tickers, start, end) -> DataFrame (Adj Close)
- make_features(prices, tickers) -> DataFrame of engineered features
- split_index(n, train_ratio) -> int (train end index)

변경점:
- 특징 생성 시 NaN/Inf 안전화: rolling에 min_periods 설정, 계산 후 전체적으로
  replace([inf,-inf], nan) -> (중앙값) 채움, 앞/뒤 보간까지 적용.
- 열 단위 insert 반복 대신 dict 수집 → concat으로 성능 개선 (fragmentation 경고 제거).
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

# Yahoo Finance download with an optional immutable local snapshot.
def fetch_prices_yf(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Return adjusted-close prices in the requested ticker order.

    Canonical paper runs set ``NLPI_DATA_SNAPSHOT`` and
    ``NLPI_USE_SNAPSHOT=1``. The first stage downloads the panel and writes
    it to that path; later stages read the same file, preventing Yahoo data
    revisions or API differences from changing results within one canonical
    run. Ordinary GUI runs behave exactly as before when these environment
    variables are absent.
    """
    import os, json, hashlib
    from pathlib import Path

    if not tickers:
        return pd.DataFrame()

    snap_raw = os.getenv("NLPI_DATA_SNAPSHOT", "").strip()
    use_snapshot = os.getenv("NLPI_USE_SNAPSHOT", "0").strip().lower() in {"1", "true", "yes"}
    save_snapshot = os.getenv("NLPI_SAVE_SNAPSHOT", "0").strip().lower() in {"1", "true", "yes"}
    snap = Path(snap_raw).expanduser() if snap_raw else None

    if use_snapshot and snap is not None and snap.exists():
        px = pd.read_csv(snap, index_col=0, parse_dates=True)
        px.index = pd.to_datetime(px.index)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        px = px.loc[(px.index >= start_ts) & (px.index < end_ts)]
        cols = [c for c in tickers if c in px.columns]
        return px[cols].dropna(how="all")

    try:
        import yfinance as yf
    except Exception:
        raise RuntimeError("yfinance is required. Install it with: pip install yfinance")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"].copy()
        else:
            px = data["Close"].copy()
    else:
        px = data.copy()

    if isinstance(px, pd.Series):
        px = px.to_frame()
    cols = [c for c in tickers if c in px.columns]
    px = px[cols].dropna(how="all")

    if snap is not None and (save_snapshot or use_snapshot) and not px.empty:
        snap.parent.mkdir(parents=True, exist_ok=True)
        px.to_csv(snap)
        digest = hashlib.sha256(snap.read_bytes()).hexdigest()
        meta = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "requested_start": start,
            "requested_end": end,
            "tickers": cols,
            "rows": int(len(px)),
            "columns": int(len(px.columns)),
            "source": "Yahoo Finance via yfinance, auto_adjust=True",
            "sha256": digest,
        }
        snap.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return px


def price_audit_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Return asset-level audit metadata required for reproducible backtests.

    The table reports first/last valid observation, missingness, and observation
    counts. It should be exported with every revised experiment to document
    asset inception dates and missing-data handling.
    """
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["Ticker", "FirstValidDate", "LastValidDate", "MissingRate", "NObs"] )
    rows = []
    for c in prices.columns:
        ser = prices[c]
        rows.append({
            "Ticker": c,
            "FirstValidDate": ser.first_valid_index(),
            "LastValidDate": ser.last_valid_index(),
            "MissingRate": float(ser.isna().mean()),
            "NObs": int(ser.notna().sum()),
        })
    return pd.DataFrame(rows)

def fetch_prices_yf_with_audit(tickers: List[str], start: str, end: str):
    """Download adjusted prices and return (prices, audit_table)."""
    px = fetch_prices_yf(tickers, start, end)
    audit = price_audit_table(px)
    audit["DownloadTimestampUTC"] = pd.Timestamp.utcnow().isoformat()
    audit["PriceType"] = "yfinance auto_adjust=True adjusted close"
    audit["Calendar"] = "Yahoo Finance trading calendar; crypto aligned to price-panel index"
    return px, audit

def make_features(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Feature set:
      - r1m (21d cumulative return)
      - r3m (63d cumulative return)
      - r12m (252d cumulative return)
      - vol3m (63d daily vol -> ann)
      - mdd (drawdown)
      - cross_vol (equal-weight avg of vol3m across assets)

    모든 계산 후:
      1) inf -> nan 치환
      2) 과거 방향 ffill만 적용
      3) 전 컬럼이 결측인 행은 제거
      4) 남는 NaN은 최종 0
    주의: bfill/median imputation을 사용하지 않아 pre-inception artificial histories를 줄인다.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    prices = prices.copy()
    prices = prices.replace([np.inf, -np.inf], np.nan)

    # 수익률
    #rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    px = prices.sort_index().ffill()
    rets = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    feat_cols = {}

    # 누적수익률: rolling product - 1
    for a in tickers:
        if a not in prices.columns: 
            continue
        r = rets[a]

        # min_periods를 지정해 초기 NaN 길이 축소
        r1m = (1 + r).rolling(21, min_periods=5).apply(np.prod, raw=True) - 1.0
        r3m = (1 + r).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
        r12m= (1 + r).rolling(252, min_periods=63).apply(np.prod, raw=True) - 1.0
        vol3m = r.rolling(63, min_periods=21).std() * np.sqrt(252)

        # mdd: 누적최고점 대비 낙폭
        px = prices[a]
        mdd = px / px.cummax() - 1.0

        feat_cols[f"{a}_r1m"]   = r1m
        feat_cols[f"{a}_r3m"]   = r3m
        feat_cols[f"{a}_r12m"]  = r12m
        feat_cols[f"{a}_vol3m"] = vol3m
        feat_cols[f"{a}_mdd"]   = mdd

    feats = pd.concat(feat_cols, axis=1)
    # 교차 변동성 (있을 때만)
    vol_cols = [f"{a}_vol3m" for a in tickers if f"{a}_vol3m" in feats.columns]
    # Defragment feature matrix before adding cross-sectional aggregate features.
    # This prevents pandas PerformanceWarning caused by repeated column insertion.
    feats = feats.copy()

    feats["cross_vol"] = feats[vol_cols].mean(axis=1) if vol_cols else 0.0

    # 안전화: inf->nan
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Reviewer-response change: no median imputation and no backward fill.
    # Only information available up to time t is propagated forward.
    feats = feats.ffill()
    feats = feats.dropna(how="all")
    feats = feats.fillna(0.0)

    # 가격 인덱스와 정렬
    feats = feats.loc[prices.index]
    return feats

def split_index(n: int, train_ratio: float = 0.7) -> int:
    train_end = int(max(1, min(n - 1, round(n * train_ratio))))
    return train_end
