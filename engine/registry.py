# engine/registry.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re

from .strategies import (
    EqStrategy, RiskParityStrategy, LWMVPStrategy,
    Momentum6mStrategy, Trend6mStrategy,
    SharpeWeightedStrategy, SortinoWeightedStrategy,
    LWMVPStrategy, HRPStrategy, BLStrategy, LLMStrategy, CodedPersonaStrategy
)

# --- 기본(비 LLM) 전략 팩토리 ---
STRATEGY_FACTORY: Dict[str, callable] = {
    "EQUAL":     lambda tickers, prices, features, cfg, **kw: EqStrategy(tickers, prices, features, cfg),
    "RiskParity":lambda tickers, prices, features, cfg, **kw: RiskParityStrategy(tickers, prices, features, cfg),
    "MVP":       lambda tickers, prices, features, cfg, **kw: LWMVPStrategy(tickers, prices, features, {**cfg, "cov_method": "sample"}),
    "LW-MVP":    lambda tickers, prices, features, cfg, **kw: LWMVPStrategy(tickers, prices, features, {**cfg, "cov_method": "ledoitwolf"}),
    "HRP":       lambda tickers, prices, features, cfg, **kw: HRPStrategy(tickers, prices, features, cfg),
    "BL":        lambda tickers, prices, features, cfg, **kw: BLStrategy(tickers, prices, features, cfg),
    "MOM6":     lambda tickers, prices, features, cfg, **kw: Momentum6mStrategy(tickers, prices, features, cfg),
    "TRND6":    lambda tickers, prices, features, cfg, **kw: Trend6mStrategy(tickers, prices, features, cfg),
    "SHARPE":    lambda tickers, prices, features, cfg, **kw: SharpeWeightedStrategy(tickers, prices, features, cfg),
    "SORTINO":   lambda tickers, prices, features, cfg, **kw: SortinoWeightedStrategy(tickers, prices, features, cfg),
    # --- Ablation: coded persona executor (no LLM) ---
    "CODED_P1":  lambda tickers, prices, features, cfg, **kw: CodedPersonaStrategy(tickers, prices, features, cfg, persona_id=1),
    "CODED_P2":  lambda tickers, prices, features, cfg, **kw: CodedPersonaStrategy(tickers, prices, features, cfg, persona_id=2),
    "CODED_P3":  lambda tickers, prices, features, cfg, **kw: CodedPersonaStrategy(tickers, prices, features, cfg, persona_id=3),
    "CODED_P4":  lambda tickers, prices, features, cfg, **kw: CodedPersonaStrategy(tickers, prices, features, cfg, persona_id=4),
    "CODED_P5":  lambda tickers, prices, features, cfg, **kw: CodedPersonaStrategy(tickers, prices, features, cfg, persona_id=5),

}

# --- NLPI/LLM 키 파서 ---
# NLPI = LLM-based Natural-Language Policy Interface.
# Backward-compatible LLM keys are still accepted, but all new paper-facing
# names should use NLPI[model|P#].
# 허용 패턴:
#   1) "NLPI" or "LLM"                       -> cfg 그대로 사용
#   2) "NLPI-P3" or "LLM-P3"                 -> cfg + prompt_profile=3
#   3) "NLPI[model_name|P3]" or "LLM[...]"   -> cfg + model_name, prompt_profile
_RX_PN = re.compile(r"^(?:NLPI|LLM)-P(?P<p>\d+)$", re.I)
_RX_BR = re.compile(r"^(?:NLPI|LLM)\[(?P<m>[^|\]]+)\|P(?P<p>\d+)\]$", re.I)

def _parse_llm_key(key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    key = key.strip()
    m = _RX_BR.match(key)
    if m:
        # 대괄호 표기: 모델 + 프롬프트 동시 지정
        model = m.group("m").strip()
        p = int(m.group("p"))
        return {**cfg, "model_name": model, "prompt_profile": p}

    m = _RX_PN.match(key)
    if m:
        # 하이픈 표기: 프롬프트만 지정
        p = int(m.group("p"))
        return {**cfg, "prompt_profile": p}

    # 순수 "LLM" 혹은 기타 → cfg 그대로
    return dict(cfg)

def create_strategy(key: str, tickers, prices, features, cfg: Dict[str, Any], **extras):
    k = str(key).strip()
    if k.upper().startswith("LLM") or k.upper().startswith("NLPI"):
        # LLMStrategy는 few-shot 소스가 추가로 필요할 수 있음
        fs = extras.get("fewshot_or_train_feats")
        llm_cfg = _parse_llm_key(k, cfg)
        return LLMStrategy(tickers, prices, features, fs, llm_cfg)

    if k not in STRATEGY_FACTORY:
        raise KeyError(f"Unknown strategy key: {k}")
    return STRATEGY_FACTORY[k](tickers, prices, features, cfg, **extras)

def build_strategies(keys: List[str], tickers, prices, features, cfg: Dict[str, Any], **extras) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k in keys:
        try:
            out[k] = create_strategy(k, tickers, prices, features, cfg, **extras)
        except KeyError:
            # 알 수 없는 키는 무시 (GUI/CLI에서 실수 방지)
            continue
    return out
