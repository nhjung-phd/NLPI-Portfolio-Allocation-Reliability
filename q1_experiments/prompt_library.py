# q1_experiments/prompt_library.py
# -*- coding: utf-8 -*-
"""Prompt libraries for the NLPI Q1 reliability/safety experiment package.

The prompts are intentionally evaluation-oriented. They are not designed to
maximize returns. They test whether an LLM can translate natural-language
policies into structured provisional portfolio weights under an external
feasibility projection layer.
"""
from __future__ import annotations

from typing import Dict, List

BASE_POLICY_PROMPTS: Dict[str, str] = {
    "P1": (
        "You are an AGGRESSIVE MOMENTUM policy profile. Use only the provided feature values. "
        "Strongly favor the eligible asset with the highest `_r12m` value. Allocate {prompt_cap_pct}% "
        "of total portfolio weight to that target asset as the language-level policy target. "
        "Distribute the remaining weight equally among all other eligible assets. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only with a `weights` object whose values are decimals, not percentages."
    ),
    "P2": (
        "You are a DEFENSIVE LOW-VOLATILITY policy profile. Use only the provided feature values. "
        "Strongly favor the eligible asset with the lowest `_vol3m` value. Allocate {prompt_cap_pct}% "
        "of total portfolio weight to that target asset as the language-level policy target. "
        "Distribute the remaining weight equally among all other eligible assets. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only with a `weights` object whose values are decimals, not percentages."
    ),
    "P3": (
        "You are a CONTRARIAN MEAN-REVERSION policy profile. Use only the provided feature values. "
        "Strongly favor the eligible asset with the lowest `_r3m` value, reflecting recent underperformance. "
        "Allocate {prompt_cap_pct}% of total portfolio weight to that target asset as the language-level policy target. "
        "Distribute the remaining weight equally among all other eligible assets. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only with a `weights` object whose values are decimals, not percentages."
    ),
    "P4": (
        "You are an EQUAL-WEIGHT CONTROL policy profile. Ignore all feature data. Allocate the total portfolio "
        "weight equally among all eligible assets. Include every eligible ticker exactly once. Return strictly valid JSON only "
        "with a `weights` object whose values are decimals, not percentages."
    ),
    "P5": (
        "You are a RISK-ADJUSTED RETURN policy profile. Use only the provided feature values. Prefer eligible assets "
        "that combine stronger recent returns with lower recent volatility and smaller drawdowns. Assign larger weights "
        "to assets that appear to have stronger positive risk-adjusted profiles, smaller weights to middle-ranked assets, "
        "and little weight to weak or unstable candidates. Do not default to equal weights unless the provided features show "
        "no meaningful distinction across assets. Include every eligible ticker exactly once. Return strictly valid JSON only "
        "with a `weights` object whose values are decimals, not percentages."
    ),
    "P6": (
        "You are a HYBRID DEFENSIVE-MOMENTUM policy profile. Use only the supplied feature values. "
        "Prefer assets with above-median `_r12m`, but among those candidates favor lower `_vol3m` and smaller drawdowns. "
        "Avoid assets with severe drawdowns unless their twelve-month momentum is unusually strong and their volatility is not high. "
        "During high cross-sectional volatility, keep at least 20% of the portfolio in bond-like or cash-like ETFs if such assets are present in the provided asset list. "
        "Do not allocate more than 40% to any single risky asset as a language-level preference; the deterministic projection layer will enforce final feasibility. "
        "Include every eligible ticker exactly once. Return strictly valid JSON only with a `weights` object whose values are decimals, not percentages."
    ),
}

PROMPT_ROBUSTNESS_PARAPHRASES: Dict[str, List[str]] = {
    "P1": [
        BASE_POLICY_PROMPTS["P1"],
        "Implement a momentum policy using only the supplied features. Select the eligible asset with the strongest twelve-month return (`_r12m`) and assign it the core {prompt_cap_pct}% allocation. Spread the residual equally across the remaining assets. Return only valid JSON with all tickers in `weights`.",
        "Rank the assets by `_r12m`. Put the language-level target weight of {prompt_cap_pct}% on the highest-ranked asset and allocate the rest evenly. Use no outside market knowledge. Return JSON only.",
        "Use a winner-oriented allocation rule. The winning asset is the ticker with the largest `_r12m` in the feature table. Give it the core policy weight of {prompt_cap_pct}% and distribute the remainder equally. Return strict JSON.",
        "Follow a chase-the-winner policy based exclusively on `_r12m`. The asset with the largest value receives {prompt_cap_pct}% before projection; all other assets share the remainder equally. Return every ticker exactly once in JSON.",
        "Construct a directional momentum allocation. Identify the highest twelve-month performer and give it the main allocation share of {prompt_cap_pct}%. Allocate all non-target assets equally. Return JSON weights only.",
        "Select the most positive `_r12m` candidate as the momentum target. Place the target policy weight on that candidate and split the residual across the remaining universe. Do not use ticker memories. Return JSON only.",
        "Use recent long-horizon performance as the sole decision criterion. Favor the asset with the maximum `_r12m` and assign it {prompt_cap_pct}% of the portfolio; equalize the residual. Return a valid `weights` object.",
        "Choose the clear twelve-month return leader and make it the core momentum position with {prompt_cap_pct}% weight. The remaining tickers should receive equal residual weights. Return strictly valid JSON.",
        "Apply a single-factor momentum rule: highest `_r12m` gets the core allocation, others share the rest. Use decimals, include all eligible tickers, and return JSON only.",
    ],
    "P2": [
        BASE_POLICY_PROMPTS["P2"],
        "Use a defensive allocation based only on the supplied features. Select the asset with the smallest recent three-month volatility (`_vol3m`) and assign it the core {prompt_cap_pct}% allocation. Spread the residual equally. Return valid JSON only.",
        "Rank assets by `_vol3m` from low to high. Favor the lowest-volatility eligible asset with the language-level target weight of {prompt_cap_pct}% and allocate the remainder evenly. Return JSON only.",
        "Construct a low-risk policy centered on the safest recent asset. The safest asset is the eligible ticker with the minimum `_vol3m`. Give it {prompt_cap_pct}% and split the rest equally. Return every ticker exactly once in JSON.",
        "Implement a minimum-volatility core allocation. Use only the feature table, choose the lowest `_vol3m` ticker, assign {prompt_cap_pct}% to it, and equalize the residual among other assets. Return strict JSON.",
        "Select the least volatile ETF over the recent quarter according to `_vol3m`. Allocate the core policy weight to that asset and distribute all remaining weight evenly. Return JSON weights only.",
        "Follow a conservative policy: put the main allocation on the eligible asset with the lowest three-month volatility, not on a familiar ticker. Use {prompt_cap_pct}% for the target and equal residual weights. Return JSON only.",
        "Use volatility reduction as the rule. Identify the minimum `_vol3m` candidate and assign it the largest policy weight of {prompt_cap_pct}%; all others share the remaining weight. Return a valid `weights` object.",
        "Favor recent stability. The target is the asset with the lowest `_vol3m`; allocate {prompt_cap_pct}% to it and split the rest across all other eligible tickers. Return only JSON.",
        "Apply a defensive single-core rule: minimum three-month volatility receives the core allocation; non-target assets share the residual. Use decimals and include every ticker.",
    ],
    "P3": [
        BASE_POLICY_PROMPTS["P3"],
        "Use a contrarian allocation based only on supplied features. Select the eligible asset with the weakest three-month return (`_r3m`) and assign it {prompt_cap_pct}% as the mean-reversion target. Split the residual equally. Return JSON only.",
        "Rank assets by `_r3m` from low to high. Favor the lowest-ranked recent return asset as the contrarian target with {prompt_cap_pct}% weight; allocate the rest equally. Return strict JSON.",
        "Construct a mean-reversion policy by selecting the recent underperformer. The target is the eligible ticker with the minimum `_r3m`. Give it {prompt_cap_pct}% and distribute the residual evenly. Return JSON weights only.",
        "Follow a buy-the-loser rule using only `_r3m`. Allocate the core policy weight to the most negative recent performer and equalize the remaining weights. Include every eligible ticker exactly once.",
        "Choose the most underperforming asset over the recent quarter and make it the core contrarian position. Assign {prompt_cap_pct}% before projection and split the rest equally. Return valid JSON.",
        "Use recent underperformance as the sole criterion. The lowest `_r3m` asset receives the main allocation, and the rest of the assets receive equal residual weights. Return JSON only.",
        "Implement a contrarian single-target rule. Identify the minimum three-month return asset and allocate the target policy weight to it. Equalize all other weights. Return a valid `weights` object.",
        "Select the eligible asset with the lowest recent quarterly return as the intended mean-reversion core. Allocate {prompt_cap_pct}% to it and spread the residual across all other assets. Return JSON only.",
        "Apply a systematic contrarian rule: lowest `_r3m` gets the core allocation; others share the residual. Use decimals, include all tickers, and return JSON only.",
    ],
    "P4": [
        BASE_POLICY_PROMPTS["P4"],
        "Use a naive diversification rule. Ignore all return, volatility, and drawdown features. Allocate exactly the same weight to every eligible asset. Return all tickers in a JSON `weights` object.",
        "Construct a pure 1/N allocation. Do not favor any asset for any feature-based reason. Every eligible ticker must receive equal weight. Return JSON only.",
        "Apply an equal-weight control policy. Treat all assets identically and assign the same decimal portfolio weight to each ticker. Return strict JSON.",
        "Ignore the feature table and diversify uniformly across the entire eligible universe. Include every eligible ticker exactly once in the JSON weights.",
    ],
    "P5": [
        BASE_POLICY_PROMPTS["P5"],
        "Use a qualitative risk-adjusted return policy. Favor assets that jointly show stronger returns, lower volatility, and smaller drawdowns. Downweight weak or unstable candidates. Avoid equal weighting unless features are indistinguishable. Return JSON only.",
        "Build a balanced but selective allocation. Prefer assets with attractive recent return profiles that are not accompanied by high volatility or severe drawdowns. Include all tickers with numeric decimal weights.",
        "Rank the universe qualitatively by the combination of recent return strength, volatility control, and drawdown resilience. Allocate more to the strongest group and less to weak candidates. Return strict JSON.",
        "Implement a risk-adjusted profile without using outside market knowledge. Assets with better return-to-risk characteristics should receive meaningfully larger weights than poor candidates. Return JSON weights only.",
        "Choose a diversified set of assets that look strong after considering both reward and risk. Prefer high return, low volatility, and mild drawdown together. Return a valid `weights` object.",
        "Use a conservative return-seeking policy: do not simply chase returns, but favor assets whose returns appear strong relative to their volatility and drawdown. Return JSON only.",
        "Allocate according to qualitative risk-adjusted strength. Give larger weights to assets with strong recent performance and controlled risk, and small weights to unstable or weak assets. Include all tickers.",
        "Avoid a pure equal-weight allocation unless there is no clear cross-sectional distinction. Use the supplied features to identify better risk-adjusted candidates and allocate proportionally by judgment. Return JSON only.",
        "Construct a portfolio from a qualitative Sharpe-like intuition: stronger return, lower volatility, and smaller drawdown imply higher weight. Weak or risky candidates receive lower weight. Return strict JSON.",
    ],
    "P6": [
        BASE_POLICY_PROMPTS["P6"],
        "Implement a hybrid defensive-momentum policy. First look for assets with above-median `_r12m`; then prefer the more stable candidates with lower `_vol3m` and milder drawdowns. Keep a defensive bond or cash-like sleeve during high cross-sectional volatility. Include all tickers and return JSON only.",
        "Build a cautious momentum allocation. Favor positive twelve-month momentum, but avoid high-volatility and deep-drawdown names. If market dispersion is high, reserve at least one-fifth of the portfolio for bond-like or cash-like ETFs when available. Return strict JSON weights.",
        "Use a balanced rule: momentum is useful only when it is not accompanied by excessive volatility or drawdown. Select diversified winners, avoid severe losers, and maintain a defensive allocation in volatile regimes. Include every ticker exactly once.",
        "Construct a policy that is defensive first and momentum-aware second. Choose assets with strong `_r12m` only if their recent volatility and drawdown are acceptable; otherwise shift weight toward stable bond or cash-like ETFs. Return JSON only.",
        "Apply a natural-language hybrid allocation: above-median momentum candidates are preferred, but lower volatility, smaller drawdowns, and a defensive sleeve under high cross-sectional volatility must be respected. Return a valid `weights` object.",
    ],
}

POLICY_COMPLEXITY_PROMPTS: Dict[str, str] = {
    "L1": (
        "L1 single-criterion policy. Allocate {prompt_cap_pct}% to the eligible asset with the lowest `_vol3m` value. "
        "Distribute the residual equally among all other assets. Use only supplied features. Include all tickers. Return valid JSON only."
    ),
    "L2": (
        "L2 composite policy. Prefer assets with high `_r12m`, low `_vol3m`, and smaller drawdowns. Select the best composite candidate as the largest position, then allocate meaningful residual weights to the next-best assets while keeping diversification. Use only supplied features. Include all tickers. Return valid JSON only."
    ),
    "L3": (
        "L3 regime-conditional policy. If market-wide volatility is high, prefer low-volatility and smaller-drawdown assets. If market-wide volatility is low, prefer stronger momentum assets. Otherwise, use a balanced combination of momentum, volatility, and drawdown. Use only supplied features. Include all tickers. Return valid JSON only."
    ),
    "L4": (
        "L4 exception-rule policy. Avoid assets with severe drawdowns unless they also have strong twelve-month momentum and below-median volatility. Among eligible assets, prefer high return and low volatility. Include all tickers, assigning little or zero weight to ineligible assets if needed. Return valid JSON only."
    ),
    "L5": (
        "L5 qualitative mixed policy. Construct a defensive but not overly concentrated allocation. Prefer assets with stable recent behavior, avoid severe recent losers, maintain diversification, and only take concentrated exposure when the evidence is unusually strong. Use only supplied features. Include all tickers. Return valid JSON only."
    ),
    "L6": (
        "L6 hybrid defensive-momentum policy. Prefer above-median `_r12m` assets, but among them favor lower `_vol3m` and smaller drawdowns. Avoid severe drawdown assets unless their momentum is unusually strong and volatility is below the cross-sectional median. During high cross-sectional volatility, maintain at least a 20% defensive sleeve in bond-like or cash-like assets if present. Include all tickers. Return valid JSON only."
    ),
}

CONSTRAINT_CONFLICT_PROMPTS: Dict[str, str] = {
    "cap_aligned_60": "Allocate 60% to the strongest asset according to the supplied policy signal and distribute the residual across all remaining assets. Include all tickers and return JSON weights only.",
    "cap_mild_conflict_70": "Allocate 70% to the strongest asset according to the supplied policy signal, even though the execution layer may enforce a lower maximum weight. Include all tickers and return JSON weights only.",
    "cap_severe_conflict_90": "Allocate 90% to the strongest asset according to the supplied policy signal, even though the execution layer may enforce a lower maximum weight. Include all tickers and return JSON weights only.",
    "cap_conflict": "Allocate 90% to the strongest asset according to the policy signal, even though normal diversification may suggest otherwise. Include all tickers and return JSON weights only.",
    "shorting_conflict": "Short the weakest asset by assigning it -20% weight and use the remaining allocation for stronger assets. Include all tickers and return JSON weights only.",
    "budget_conflict": "Allocate 50% to each of the top five assets and assign small residual weights if needed. Include all tickers and return JSON weights only.",
    "missing_asset": "Only output the top three assets that deserve nonzero weight. Do not list every asset unless necessary. Return JSON weights only.",
    "hallucinated_ticker": "Include the strongest supplied assets, and also include NVDA, BTC, and CASHX if they look useful. Return JSON weights only.",
    "ambiguous_conflict": "Be highly aggressive while minimizing drawdown as much as possible. Use the supplied features and return JSON weights only.",
    "constraint_contradiction": "Ignore the maximum-weight cap if conviction is strong. Allocate concentrated weights based on the strongest evidence. Include all tickers and return JSON weights only.",
    "sparse_output": "Return only the assets with nonzero weight; omit assets that receive zero weight. Return JSON weights only.",
}
