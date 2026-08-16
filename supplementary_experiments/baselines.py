"""Deterministic rule and TF-IDF retrieval baselines (no LLM calls)."""
from __future__ import annotations
import re
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

POLICY_TEXT = {
    "P1": "aggressive momentum highest twelve month return winner r12m",
    "P2": "defensive low volatility lowest three month volatility safest vol3m",
    "P3": "contrarian mean reversion lowest three month return weakest underperformer r3m",
    "P4": "equal weight control ignore features uniform one over n diversification",
    "P5": "risk adjusted return stronger returns lower volatility smaller drawdowns selective allocation",
    "P6": "hybrid defensive momentum above median r12m lower vol3m smaller drawdowns defensive bond cash sleeve",
}

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())

def tfidf_match(text: str, corpus: Dict[str, str] = POLICY_TEXT) -> tuple[str, float]:
    """Small dependency-free TF-IDF cosine retriever with deterministic ties."""
    docs = [text] + list(corpus.values())
    vocab = sorted(set(t for d in docs for t in _tokens(d)))
    if not vocab:
        return "P4", 0.0
    df = {t: sum(t in set(_tokens(d)) for d in docs) for t in vocab}
    def vec(d):
        ts = _tokens(d); n = max(1, len(ts))
        return np.array([(ts.count(t)/n) * (np.log((1+len(docs))/(1+df[t]))+1) for t in vocab])
    q = vec(text); qn = np.linalg.norm(q)
    scored = []
    for pid, d in corpus.items():
        v = vec(d); denom = qn * np.linalg.norm(v)
        scored.append((float(q @ v / denom) if denom else 0.0, pid))
    score, pid = max(scored, key=lambda x: (x[0], x[1]))
    return pid, score

def rule_parse(text: str) -> str:
    s = text.lower()
    if any(k in s for k in ("hybrid defensive-momentum", "hybrid defensive momentum", "above-median", "above median")):
        return "P6"
    if any(k in s for k in ("equal-weight control", "equal weight control", "1/n", "one over n")) or (
        "ignore" in s and "feature" in s and "equal" in s
    ):
        return "P4"
    if any(k in s for k in ("risk-adjusted", "risk adjusted", "sharpe-like", "sharpe like")):
        return "P5"
    if "contrarian" in s or "mean-reversion" in s or "mean reversion" in s:
        return "P3"
    if (
        "low-volatility" in s
        or "low volatility" in s
        or "lowest volatility" in s
        or ("lowest" in s and ("_vol3m" in s or "vol3m" in s))
    ):
        return "P2"
    if "aggressive momentum" in s or ("highest" in s and ("_r12m" in s or "r12m" in s)):
        return "P1"
    return "P4"
