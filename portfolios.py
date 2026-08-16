# -*- coding: utf-8 -*-
"""
Predefined portfolios for selection in the GUI.

The first three universes were added for the reviewer-response revision.
They allow the main Computational Economics experiments to use a
pre-specified ETF-only multi-asset universe rather than an ex-post
single-stock-heavy universe.
"""

# ---------------------------------------------------------------------
# Reviewer-response experimental universes
# ---------------------------------------------------------------------

# Main experimental universe: ETF-only multi-asset universe
# Purpose: reduce single-stock survivorship / hindsight-selection concerns.
etf_multi_asset_main = [
    "SPY", "QQQ", "IWM", "DIA",
    "EFA", "EEM", "VEA", "VWO",
    "AGG", "BND", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG",
    "GLD", "SLV", "DBC",
    "VNQ", "UUP", "BIL",
]

# Crypto-inclusive robustness universe
# Purpose: test whether conclusions change when BTC-USD is added, while
# keeping BTC as a robustness setting rather than the main universe.
etf_multi_asset_crypto = etf_multi_asset_main + ["BTC-USD"]

# Factor ETF robustness universe
# Purpose: optional robustness / bridge to the follow-up factor-allocation paper.
factor_etf_assets = [
    "MTUM", "VLUE", "QUAL", "USMV", "SPLV",
    "VTV", "VUG", "IWD", "IWF", "IVE", "IVW",
    "RSP", "SPY",
]

# ---------------------------------------------------------------------
# Legacy universes from the EAAI submission
# ---------------------------------------------------------------------

# Equity (Big Tech + S&P/Dow)
equity_assets = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "JPM", "V", "DIS", "INTC", "BA", "XOM", "JNJ", "KO", "PG", "WMT", "UNH",
    "SPY", "QQQ", "DIA",
]

# Simple ETF control universe
etf_assets = ["SPY", "QQQ", "DIA"]

# Crypto (multi)
crypto_assets = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

# Crypto (single)
crypto_assets2 = ["BTC-USD"]

# Macro
macro_assets = [
    "TLT", "IEF",
    "GLD", "SLV",
    "TIP",
]

# Multi asset (equity + BTC + macro)
multi_assets = equity_assets + crypto_assets2 + macro_assets

# Public dictionary. The first entry becomes the default GUI/CLI research universe.
PORTFOLIOS = {
    "ETF Multi-Asset Main (22)": etf_multi_asset_main,
    "ETF Multi-Asset + BTC": etf_multi_asset_crypto,
    "Factor ETF Robustness": factor_etf_assets,
    "Equity (Big Tech + Indexes)": equity_assets,
    "ETF (SPY/QQQ/DIA)": etf_assets,
    "Crypto (BTC/ETH/BNB/SOL)": crypto_assets,
    "Crypto (BTC only)": crypto_assets2,
    "Macro (Bonds/Gold/Silver/TIPS)": macro_assets,
    "Multi-Asset (Equity+BTC+Macro)": multi_assets,
}

DEFAULT_PORTFOLIO_NAME = "ETF Multi-Asset Main (22)"
DEFAULT_TICKERS = etf_multi_asset_main
