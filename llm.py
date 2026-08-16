# llm.py - FINAL CORRECTED VERSION
from __future__ import annotations
import json, re, time
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, field_validator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import threading
OLLAMA_LOCK = threading.Lock()

# ============================================================
# Diagnostics meta (NEW)
# ============================================================
LAST_CALL_META = {
    "json_valid": 0,
    "repair_used": 0,
    "parse_failed": 0,
    "elapsed_sec": float("nan"),
    "raw_response": "",
    "parsed_response": "",
    "model": "",
    "retry_count": 0,
    "output_length": 0,
    "hallucinated_ticker_count": 0,
    "missing_asset_count": 0,
    "invalid_weight_count": 0,
    "negative_weight_count": 0,
    "equal_fallback": 0,
    "attempt_count": 0,
    "parse_retry_count": 0,
    "network_retry_count": 0,
    "timeout_retry_count": 0,
    "eval_count": 0,
    "eval_duration_sec": 0.0,
    "prompt_eval_count": 0,
    "done_reason": "",
    "attempt_diagnostics": [],
}

def _reset_last_call_meta(model: str = ""):
    LAST_CALL_META.update({
        "json_valid": 0,
        "repair_used": 0,
        "parse_failed": 0,
        "elapsed_sec": float("nan"),
        "raw_response": "",
        "parsed_response": "",
        "model": model or "",
        "retry_count": 0,
        "output_length": 0,
        "hallucinated_ticker_count": 0,
        "missing_asset_count": 0,
        "invalid_weight_count": 0,
        "negative_weight_count": 0,
        "equal_fallback": 0,
        "attempt_count": 0,
        "parse_retry_count": 0,
        "network_retry_count": 0,
        "timeout_retry_count": 0,
        "eval_count": 0,
        "eval_duration_sec": 0.0,
        "prompt_eval_count": 0,
        "done_reason": "",
        "attempt_diagnostics": [],
    })


class OllamaCallError(RuntimeError):
    """Final Ollama failure with diagnostics suitable for checkpoint storage."""

    def __init__(self, message: str, meta: Optional[dict] = None):
        super().__init__(message)
        self.meta = dict(meta or {})


class OllamaHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)

def check_ollama(url_base: str, timeout: float = 2.5) -> Tuple[bool, str]:
    base = url_base.rstrip("/")
    try:
        r = requests.get(base + "/api/tags", timeout=timeout)
        if r.status_code == 200:
            tags = r.json().get("models", [])
            names = ", ".join([m.get("model") or m.get("name") or "" for m in tags]) or "models listed"
            return True, f"Connected ({names})"
    except Exception: pass
    return False, "Disconnected"

class LLMWeights(BaseModel):
    weights: Dict[str, float]
    rationale: Optional[str] = None
    @field_validator("weights")
    @classmethod
    def _guard(cls, w: Dict[str, float]):
        if w is None:
            w = {}

        v: Dict[str, float] = {}
        for k, val in w.items():
            try:
                v[k] = float(max(0.0, val))
            except (ValueError, TypeError):
                v[k] = 0.0

        s = sum(v.values())

        # ✅ 0-sum 방지: LLM이 전부 0을 내는 경우 equal fallback
        if s <= 0 and len(v) > 0:
            n = len(v)
            return {k: 1.0 / n for k in v}

        return {k: x / s for k, x in v.items()}

PROMPT_PROFILES: Dict[int, str] = {
    1: (
        "You are an AGGRESSIVE MOMENTUM policy profile. Your task is to translate a natural-language momentum policy into provisional portfolio weights. "
        "Use only the provided feature values, not outside market knowledge. Identify the eligible asset that is the clearest winner, defined by the highest `_r12m` value. "
        "This is a directional policy profile, so do not default to equal weights when the feature values distinguish a winner. "
        "Allocate {prompt_cap_pct}% of the total portfolio weight to this winner as the language-level policy target. "
        "The hard maximum-weight constraint is {maxw_pct}%. Do not solve the projection yourself; express the language-level target and let the projection layer enforce feasibility before execution. "
        "Distribute the remaining weight equally among all other eligible assets. "
        "Use only tickers from the provided asset list. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only. The `weights` field MUST be a JSON object where keys are tickers and values are numeric portfolio weights expressed as decimals, not percentages."
    ),
    2: (
        "You are a DEFENSIVE LOW-VOLATILITY policy profile. Your task is to translate a natural-language risk-averse policy into provisional portfolio weights. "
        "Use only the provided feature values, not outside market knowledge. Identify the eligible asset that appears safest, defined by the lowest `_vol3m` value. "
        "This is a defensive single-core policy profile, so do not default to equal weights when one asset clearly has the lowest recent volatility. "
        "Allocate {prompt_cap_pct}% of the total portfolio weight to this safest asset as the language-level policy target. "
        "The hard maximum-weight constraint is {maxw_pct}%. Do not solve the projection yourself; express the language-level target and let the projection layer enforce feasibility before execution. "
        "Distribute the remaining weight equally among all other eligible assets. "
        "Use only tickers from the provided asset list. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only. The `weights` field MUST be a JSON object where keys are tickers and values are numeric portfolio weights expressed as decimals, not percentages."
    ),
    3: (
        "You are a CONTRARIAN MEAN-REVERSION policy profile. Your task is to translate a natural-language contrarian policy into provisional portfolio weights. "
        "Use only the provided feature values, not outside market knowledge. Identify the eligible asset with the weakest recent performance, defined by the lowest `_r3m` value. "
        "Interpret this as a mean-reversion policy: the recently underperforming asset is the intended contrarian core position. "
        "This is a directional policy profile, so do not default to equal weights when the feature values distinguish an underperformer. "
        "Allocate {prompt_cap_pct}% of the total portfolio weight to this underperforming asset as the language-level policy target. "
        "The hard maximum-weight constraint is {maxw_pct}%. Do not solve the projection yourself; express the language-level target and let the projection layer enforce feasibility before execution. "
        "Distribute the remaining weight equally among all other eligible assets. "
        "Use only tickers from the provided asset list. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only. The `weights` field MUST be a JSON object where keys are tickers and values are numeric portfolio weights expressed as decimals, not percentages."
    ),
    4: (
        "You are an EQUAL-WEIGHT CONTROL policy profile. Your task is to translate a naive diversification policy into provisional portfolio weights. "
        "Ignore all feature data completely. Do not favor any asset based on return, volatility, drawdown, trend, or asset class. "
        "Allocate the total portfolio weight equally among all eligible assets, using the same weight for each eligible ticker. "
        "This profile is intentionally a control condition; equal weighting is the required behavior. "
        "Use only tickers from the provided asset list. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only. The `weights` field MUST be a JSON object where keys are tickers and values are numeric portfolio weights expressed as decimals, not percentages."
    ),
    5: (
        "You are a RISK-ADJUSTED RETURN policy profile. Your task is to translate a natural-language risk-adjusted return policy into provisional portfolio weights. "
        "Use only the provided feature values, not outside market knowledge. Prefer eligible assets that combine stronger recent returns with lower recent volatility. "
        "First, form a qualitative ranking of assets using the return and volatility features together: assets with higher recent returns and lower recent volatility should be treated as stronger candidates; assets with weak returns, negative returns, high volatility, or unstable profiles should be treated as weaker candidates. "
        "Select a small leading group of assets that appear to have the strongest positive risk-adjusted return profiles, and assign this group meaningfully larger weights than the rest of the universe. "
        "Assign smaller weights to middle-ranked assets and little or minimal weight to weak risk-adjusted-return candidates. "
        "Do not default to equal weights unless the provided features show no meaningful distinction across assets. "
        "Do not intentionally exceed the hard maximum-weight constraint of {maxw_pct}% for any single asset; if necessary, the projection layer will enforce final feasibility before execution. "
        "Use only tickers from the provided asset list. Include every eligible ticker exactly once. "
        "Return strictly valid JSON only. The `weights` field MUST be a JSON object where keys are tickers and values are numeric portfolio weights expressed as decimals, not percentages."
    ),
}

def get_prompt_template(profile_id: int) -> str:
    return PROMPT_PROFILES.get(int(profile_id) if profile_id else 1, PROMPT_PROFILES[1])

def build_fewshot_db(feat_df: pd.DataFrame, assets: List[str], k: int = 8, profile_id: int = 1, prompt_cap_pct: float = 60.0) -> List[tuple]:
    if feat_df is None or feat_df.empty or not assets: return []
    F = feat_df.dropna(axis=0, how="any").copy()
    prompt_cap = max(0.0, min(float(prompt_cap_pct) / 100.0, 1.0))
    if len(F) < k: return []
    
    k = min(k, len(F))
    X = StandardScaler().fit_transform(F.values)
    #km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    
    fewshots: List[tuple] = []
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0: continue
        
        center = km.cluster_centers_[c]; d = np.linalg.norm(X[idx] - center, axis=1)
        representative_idx = idx[np.argmin(d)]; t = F.index[representative_idx]; r = F.loc[t]
        
        label_w = pd.Series(0.0, index=assets)
        try:
            if profile_id == 1: # Aggressive Momentum
                moms = [r.get(f"{a}_r12m", -np.inf) for a in assets]
                if not all(m == -np.inf for m in moms): 
                    top = assets[np.argmax(moms)]
                    label_w[:] = (1.0 - prompt_cap) / (len(assets) - 1)
                    label_w[top] = prompt_cap
            elif profile_id == 2: # Risk-Averse
                vols = [r.get(f"{a}_vol3m", np.inf) for a in assets]
                if not all(v == np.inf for v in vols): 
                    top = assets[np.argmin(vols)]
                    label_w[:] = (1.0 - prompt_cap) / (len(assets) - 1)
                    label_w[top] = prompt_cap
            elif profile_id == 3: # Contrarian
                moms = [r.get(f"{a}_r3m", np.inf) for a in assets]
                if not all(m == np.inf for m in moms): 
                    top = assets[np.argmin(moms)]
                    label_w[:] = (1.0 - prompt_cap) / (len(assets) - 1)
                    label_w[top] = prompt_cap
            elif profile_id == 4: # Equal Weight
                label_w[:] = 1.0 / len(assets)
            elif profile_id == 5: # Sharpe Proxy
                # 1) read 1-month mean and std proxies from features
                mus = pd.Series([r.get(f"{a}_r1m_mean", np.nan) for a in assets], index=assets)
                sigmas = pd.Series([r.get(f"{a}_r1m_std",  np.nan) for a in assets], index=assets)

                # Fallback: if your feature keys are different, use the available ones:
                # mus    = pd.Series([r.get(f"{a}_r1m", np.nan) for a in assets], index=assets)
                # sigmas = pd.Series([r.get(f"{a}_vol1m", np.nan) for a in assets], index=assets)

                # 2) robust cleaning
                sigmas = sigmas.clip(lower=1e-6)

                # if missing values exist, degrade gracefully
                mus = mus.fillna(0.0)
                sigmas = sigmas.fillna(1.0)

                # 3) Sharpe proxy
                sharpe = mus / sigmas

                # 4) clip negatives to zero
                sharpe = sharpe.clip(lower=0.0)

                # 5) normalize; if all zeros => equal weight fallback
                if float(sharpe.sum()) > 0:
                    label_w = sharpe / sharpe.sum()
                else:
                    label_w = pd.Series(1.0 / len(assets), index=assets)        
            else: # P5 (Sharpe-proxy) approximate label for few-shot guidance
                vols = pd.Series([r.get(f"{a}_vol3m", 1.0) for a in assets], index=assets)
                inv_vols = 1.0 / vols.clip(lower=1e-6); label_w = inv_vols / inv_vols.sum()
        except Exception:
            label_w[:] = 1.0 / len(assets)

        if label_w.sum() <= 0: label_w[:] = 1.0 / len(assets)
        fewshots.append((t, r.to_dict(), label_w.to_dict()))
    return fewshots

def render_fewshot_block(fewshots: List[tuple], max_k: int = 4, max_feats: int = 12) -> str:
    lines: List[str] = []
    for i, (t, feat, w) in enumerate(fewshots[:max_k]):
        lines += [f"### EXAMPLE {i+1}", "Features:"]
        cnt = 0
        for k in sorted(feat.keys()):
            if cnt >= max_feats: break
            try: lines.append(f"- {k}: {float(feat[k]):.6f}"); cnt += 1
            except Exception: continue
        lines.append("TargetWeights(JSON): " + json.dumps({k: round(v, 4) for k,v in w.items()}))
    return "\n".join(lines)

def build_prompt(features: dict, assets: List[str], fewshot_block: str, cfg) -> str:
    maxw = int(round(float(cfg.get("max_weight", 0.60)) * 100))
    prompt_cap_pct = float(cfg.get("prompt_cap_pct", maxw))
    turn = int(round(float(cfg.get("turnover_cap", 0.25)) * 100))
    profile_id = cfg.get("prompt_profile", 1)
    
    tmpl = get_prompt_template(profile_id)
    if any(x in tmpl for x in ["{maxw_pct}", "{prompt_cap_pct}", "{turn_pct}"]):
        head = tmpl.format(maxw_pct=maxw, prompt_cap_pct=int(round(prompt_cap_pct)), turn_pct=turn)
    else:
        head = tmpl
    
    cur = ["Current Features:"]
    for k in sorted(features.keys()):
        try: cur.append(f"- {k}: {float(features[k]):.6f}")
        except Exception: pass
    return "\n".join([head, "", fewshot_block or "", "", "\n".join(cur), "Assets: " + ", ".join(assets)])

def _extract_json(text: str) -> dict | None:
    """
    Extract JSON object from LLM output.
    - 1st pass: strict json.loads
    - 2nd pass: light repair (set LAST_CALL_META["repair_used"]=1 if repaired)
    """
    text = (text or "").strip()
    if not text:
        return None

    # Prefer fenced JSON block
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if blocks:
            text = blocks[0].strip()

    # Find first JSON object span by brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    candidate = text[start:end].strip()

    # Pass 1: strict parse
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Pass 2: light repair (conservative)
    repaired = candidate

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Replace single quotes with double quotes ONLY if it looks like JSON-ish
    # (avoid touching already-double-quoted strings too aggressively)
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')

    # Quote unquoted keys: {SPY: 0.5} -> {"SPY": 0.5}
    repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:\s*)', r'\1"\2"\3', repaired)

    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            # mark repair used
            LAST_CALL_META["repair_used"] = 1
            return obj
    except Exception:
        return None

    return None

def _canon_ticker(x: object) -> str:
    """Canonicalize ticker symbols for robust matching (LLM may output 'spy', 'SPY ', etc.)."""
    return re.sub(r"[^A-Za-z0-9._\-]", "", str(x)).upper()

def _post_json(url: str, payload: dict, timeout: Tuple[float, float]) -> dict:
    res = requests.post(url, json=payload, timeout=timeout)
    if res.status_code != 200:
        raise OllamaHTTPError(res.status_code, f"Ollama HTTP {res.status_code}: {res.text[:200]}")
    try: return res.json()
    except Exception: raise RuntimeError(f"Ollama returned non-JSON body: {res.text[:200]}")

def call_llm(prompt: str, url_base: str, model: str, log_fn: Callable,
             timeout: Tuple[float, float] = (30.0, 600.0),
             max_retries: int = 2,
             parse_retries: int = 1,
             timeout_retries: int = 1,
             num_predict: int = 512,
             seed: int = 42,
             temperature: float = 0.0,
             top_p: float = 0.9,
             keep_alive: str = "30m") -> Optional[LLMWeights]:

    _reset_last_call_meta(model)
    t0_all = time.perf_counter()

    base = url_base.rstrip("/")
    allowed_str = prompt.split("Assets:")[-1].strip()
    allowed = [a.strip() for a in allowed_str.split(",") if a.strip()]
    payload = {
        "model": model,
        "stream": False,
        "think": False,  # Qwen thinking 비활성화
        "keep_alive": keep_alive,
        "format": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "properties": {a: {"type": "number"} for a in allowed},
                    "required": allowed,
                    "additionalProperties": False,
                }
            },
            "required": ["weights"],
            "additionalProperties": False,
        },
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "num_predict": int(num_predict),
        }
    }

    parse_used = 0
    network_used = 0
    timeout_used = 0
    attempt = 0
    final_error: Optional[Exception] = None

    while True:

        try:
            with OLLAMA_LOCK:  # 🔥 직렬화 (Metal crash 방지)
                log_fn(f"Calling Ollama chat API (model={model}, attempt={attempt})...")
                data = _post_json(base + "/api/chat", payload, timeout=timeout)

            raw_content = (data.get("message") or {}).get("content", "")
            LAST_CALL_META["raw_response"] = raw_content if isinstance(raw_content, str) else json.dumps(raw_content, ensure_ascii=False)
            LAST_CALL_META["output_length"] = len(LAST_CALL_META["raw_response"])
            LAST_CALL_META["retry_count"] = attempt
            LAST_CALL_META["attempt_count"] = attempt + 1
            LAST_CALL_META["eval_count"] = int(data.get("eval_count", 0) or 0)
            LAST_CALL_META["eval_duration_sec"] = float(data.get("eval_duration", 0) or 0) / 1_000_000_000.0
            LAST_CALL_META["prompt_eval_count"] = int(data.get("prompt_eval_count", 0) or 0)
            LAST_CALL_META["done_reason"] = str(data.get("done_reason", "") or "")

            if isinstance(raw_content, dict):
                parsed = raw_content
            elif isinstance(raw_content, str):
                parsed = _extract_json(raw_content.strip())
            else:
                parsed = None

            if parsed and isinstance(parsed, dict):

                # 1) "weights" 래핑
                if "weights" not in parsed:
                    parsed = {"weights": parsed}

                weights = parsed.get("weights", {})
                if not isinstance(weights, dict):
                    raise ValueError("weights is not dict")

                # 2) allowed ticker set 추출
                # Canonical map: canon -> original (preserve original tickers)
                allowed_map = {_canon_ticker(a): a for a in allowed}
                allowed_set = set(allowed_map.keys())
                raw_keys = {_canon_ticker(k) for k in weights.keys()}
                LAST_CALL_META["hallucinated_ticker_count"] = len(raw_keys - allowed_set)
                LAST_CALL_META["missing_asset_count"] = len(allowed_set - raw_keys)

                # 3) ticker hallucination 제거 + 숫자 캐스팅 (robust matching)
                filtered: Dict[str, float] = {}
                invalid_count = 0
                negative_count = 0
                for k, v in weights.items():
                    ck = _canon_ticker(k)
                    if ck in allowed_set:
                        orig_k = allowed_map[ck]
                        try:
                            fv = float(v)
                            if fv < 0:
                                negative_count += 1
                            filtered[orig_k] = fv
                        except Exception:
                            invalid_count += 1
                            filtered[orig_k] = 0.0
                LAST_CALL_META["invalid_weight_count"] = invalid_count
                LAST_CALL_META["negative_weight_count"] = negative_count

                # ✅ 필터 후 비면: (a) P4/P5는 equal fallback, (b) 그 외는 에러
                # ✅ 필터 후 비면: 무조건 equal fallback (안정화)
                if not filtered:
                    n = len(allowed)
                    if n == 0:
                        raise ValueError("No allowed tickers")
                    filtered = {k: 1.0 / n for k in allowed}
                    LAST_CALL_META["repair_used"] = 1
                    LAST_CALL_META["equal_fallback"] = 1

                # 4) P4: equal 강제 (LLM을 쓰되, 결과는 설계대로 보정)
                if "EQUAL-WEIGHT CONTROL" in prompt or "NAIVE allocator" in prompt:
                    n = len(allowed)
                    filtered = {k: 1.0 / n for k in allowed}
                    LAST_CALL_META["equal_fallback"] = 1

                # 5) P5: Sharpe-proxy sanity (음수→0, 합 0이면 equal)
                if "RISK-ADJUSTED RETURN" in prompt or "SHARPE RATIO MAXIMIZER" in prompt:
                    filtered = {k: max(0.0, float(v)) for k, v in filtered.items()}

                # 6) 정규화 (합 0이면 equal fallback)
                s = sum(filtered.values())
                if s <= 0:
                    n = len(filtered)
                    normalized = {k: 1.0 / n for k in filtered}
                    LAST_CALL_META["equal_fallback"] = 1
                else:
                    normalized = {k: v / s for k, v in filtered.items()}

                LAST_CALL_META["json_valid"] = 1
                LAST_CALL_META["parsed_response"] = json.dumps({"weights": normalized}, ensure_ascii=False)
                LAST_CALL_META["elapsed_sec"] = time.perf_counter() - t0_all
                return LLMWeights(weights=normalized)

            else:
                diag = {
                    "attempt": attempt,
                    "error_type": "parse",
                    "output_length": LAST_CALL_META["output_length"],
                    "eval_count": LAST_CALL_META["eval_count"],
                    "eval_duration_sec": LAST_CALL_META["eval_duration_sec"],
                    "done_reason": LAST_CALL_META["done_reason"],
                }
                LAST_CALL_META["attempt_diagnostics"].append(diag)
                if parse_used < max(0, int(parse_retries)):
                    parse_used += 1
                    LAST_CALL_META["parse_retry_count"] = parse_used
                    log_fn(
                        f"[WARN] LLM attempt {attempt} parse failed "
                        f"(chars={diag['output_length']}, tokens={diag['eval_count']}, "
                        f"done_reason={diag['done_reason'] or 'unknown'}); "
                        f"regenerating {parse_used}/{parse_retries}"
                    )
                    attempt += 1
                    time.sleep(0.4)
                    continue
                final_error = ValueError("Parsing failed")
                break

        except requests.exceptions.Timeout as e:
            final_error = e
            LAST_CALL_META["attempt_diagnostics"].append({
                "attempt": attempt, "error_type": "timeout", "message": str(e)[:500]
            })
            if timeout_used < max(0, int(timeout_retries)):
                timeout_used += 1
                LAST_CALL_META["timeout_retry_count"] = timeout_used
                log_fn(f"[WARN] LLM attempt {attempt} timed out; retrying {timeout_used}/{timeout_retries}")
                attempt += 1
                time.sleep(0.8 * (2 ** (timeout_used - 1)))
                continue
            break

        except (requests.exceptions.ConnectionError, OllamaHTTPError) as e:
            final_error = e
            retryable = not isinstance(e, OllamaHTTPError) or e.status_code in {429, 500, 502, 503, 504}
            LAST_CALL_META["attempt_diagnostics"].append({
                "attempt": attempt, "error_type": "network_or_http", "message": str(e)[:500]
            })
            if retryable and network_used < max(0, int(max_retries)):
                network_used += 1
                LAST_CALL_META["network_retry_count"] = network_used
                log_fn(f"[WARN] LLM attempt {attempt} transport failed: {e}; retrying {network_used}/{max_retries}")
                attempt += 1
                time.sleep(0.8 * (2 ** (network_used - 1)))
                continue
            break

        except Exception as e:
            final_error = e
            LAST_CALL_META["attempt_diagnostics"].append({
                "attempt": attempt, "error_type": type(e).__name__, "message": str(e)[:500]
            })
            break

    LAST_CALL_META["parse_failed"] = int(isinstance(final_error, ValueError))
    LAST_CALL_META["attempt_count"] = attempt + 1
    LAST_CALL_META["retry_count"] = attempt
    LAST_CALL_META["elapsed_sec"] = time.perf_counter() - t0_all
    message = (
        f"{type(final_error).__name__ if final_error else 'OllamaError'}: "
        f"{final_error or 'unknown failure'}; attempts={attempt + 1}, "
        f"parse_retries={parse_used}, timeout_retries={timeout_used}, "
        f"network_retries={network_used}"
    )
    log_fn(f"[ERROR] {message}")
    raise OllamaCallError(message, LAST_CALL_META)
