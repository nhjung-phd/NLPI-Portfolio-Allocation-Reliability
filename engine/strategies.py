# engine/strategies.py - FINAL CORRECTED VERSION
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import time

# llm 모듈 이름을 llm로 유지
import llm
from llm import call_llm, build_prompt, build_fewshot_db, render_fewshot_block


from .cov_utils import estimate_cov, hrp_alloc


def project_capped_simplex(v: pd.Series, maxw: float, tol: float = 1e-10) -> pd.Series:
    """Euclidean projection onto {w: sum(w)=1, 0<=w_i<=maxw}.

    This replaces the old clip-then-normalize repair. The bisection form
    solves the capped-simplex projection and avoids reintroducing cap
    violations through post-normalization.
    """
    x = v.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx = x.index
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float)
    maxw = float(maxw)
    if maxw * n < 1.0 - 1e-12:
        raise ValueError(f"Infeasible cap: max_weight({maxw}) * N({n}) < 1")
    a = x.values.astype(float)
    lo = float(np.min(a - maxw) - 1.0)
    hi = float(np.max(a) + 1.0)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        w = np.clip(a - mid, 0.0, maxw)
        if w.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(a - hi, 0.0, maxw)
    total = float(w.sum())
    if total <= tol:
        w = np.ones(n) / n
    else:
        w = w / total
    # Numerical cleanup only; cap is preserved up to tolerance.
    w = np.clip(w, 0.0, maxw)
    w = w / w.sum()
    return pd.Series(w, index=idx)


def _series_entropy(w: pd.Series) -> float:
    x = w[w > 0].astype(float)
    if len(x) == 0:
        return 0.0
    return float(-(x * np.log(x)).sum())

# ======================================================================
# 공통 베이스
# ======================================================================
class Strategy:
    def __init__(self, tickers: List[str], prices: pd.DataFrame, features: pd.DataFrame, cfg: Dict):
        self.tickers = list(tickers)
        self.prices = prices
        self.features = features
        self.cfg = dict(cfg) if cfg is not None else {}
        px = self.prices.reindex(columns=self.tickers).ffill()
        self.returns = (px.pct_change(fill_method=None)
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0.0))
        
    # ==========================================================
    # Diagnostics helpers (NEW)
    # cfg["_diag_counts"] : dict of counters
    # cfg["_diag_events"] : list of per-step events
    # ==========================================================
    def _diag_init(self):
        if "_diag_counts" not in self.cfg or not isinstance(self.cfg.get("_diag_counts"), dict):
            self.cfg["_diag_counts"] = {
                "json_valid": 0,
                "parse_failed": 0,
                "repair_used": 0,
                "cap_clip": 0,
                "turnover_clip": 0,
                "n_calls": 0,
                "n_steps": 0,
                "sum_proj_l1": 0.0,
                "sum_proj_l2": 0.0,
                "neg_weight_count": 0,
                "over_cap_count": 0,
                "sum_entropy": 0.0,
                "sum_herfindahl": 0.0,
                "equal_collapse": 0,
                "hallucinated_ticker_count": 0,
                "missing_asset_count": 0,
                "invalid_weight_count": 0,
                "negative_weight_count": 0,
                "equal_fallback": 0,
                "sum_latency_sec": 0.0,
                "n_latency": 0,
            }
        if "_diag_events" not in self.cfg or not isinstance(self.cfg.get("_diag_events"), list):
            self.cfg["_diag_events"] = []

    def _diag_emit(self, event: dict):
        # event is a dict; best-effort append
        try:
            self._diag_init()
            self.cfg["_diag_events"].append(event)
        except Exception:
            pass

    def clamp_and_normalize(self, w: pd.Series) -> pd.Series:
        """Project raw/provisional weights onto the capped simplex.

        This is the main reviewer-response change: rather than sequential
        clip-and-normalize repair, the target is projected onto
        {sum(w)=1, 0<=w_i<=wmax}. The method also records projection
        distance and allocation-concentration diagnostics.
        """
        self._diag_init()

        w_in = w.reindex(self.tickers).fillna(0.0).astype(float)
        maxw = float(self.cfg.get("max_weight", 1.0))
        neg_count = int((w_in < 0.0).sum())
        over_count = int((w_in > maxw).sum())

        try:
            w_out = project_capped_simplex(w_in, maxw)
        except Exception:
            # Conservative fallback: equal weights remain feasible whenever maxw*N>=1.
            w_out = pd.Series(1.0 / len(self.tickers), index=self.tickers)

        try:
            diff = w_in.reindex(self.tickers).fillna(0.0) - w_out.reindex(self.tickers).fillna(0.0)
            l1 = float(np.abs(diff.values).sum())
            l2 = float(np.sqrt((diff.values ** 2).sum()))
        except Exception:
            l1, l2 = 0.0, 0.0

        herf = float((w_out ** 2).sum())
        ent = _series_entropy(w_out)
        eq = pd.Series(1.0 / len(self.tickers), index=self.tickers)
        collapse = int(float(np.abs(w_out - eq).sum()) < 1e-4)

        if neg_count > 0 or over_count > 0:
            self.cfg["_diag_counts"]["cap_clip"] += 1
        self.cfg["_diag_counts"]["sum_proj_l1"] += l1
        self.cfg["_diag_counts"]["sum_proj_l2"] += l2
        self.cfg["_diag_counts"]["neg_weight_count"] += neg_count
        self.cfg["_diag_counts"]["over_cap_count"] += over_count
        self.cfg["_diag_counts"]["sum_entropy"] += ent
        self.cfg["_diag_counts"]["sum_herfindahl"] += herf
        self.cfg["_diag_counts"]["equal_collapse"] += collapse
        self.cfg["_diag_counts"]["n_steps"] += 1

        self._diag_emit({
            "event": "projection",
            "proj_l1": l1,
            "proj_l2": l2,
            "cap_clip": int(neg_count > 0 or over_count > 0),
            "neg_weight_count": neg_count,
            "over_cap_count": over_count,
            "entropy": ent,
            "herfindahl": herf,
            "equal_collapse": collapse,
            "sum_weight": float(w_out.sum()),
            "max_weight": float(w_out.max()),
            "min_weight": float(w_out.min()),
        })

        return w_out

    def apply_turnover_cap(self, prev_w: pd.Series, target_w: pd.Series) -> pd.Series:
        """Apply an L1 turnover cap while preserving feasibility.

        Key fix:
        - Turnover should be computed and enforced on *feasible* weights.
        - If both prev and target are feasible (non-neg, sum=1, maxw cap),
        then their convex combination remains feasible; no re-normalization
        is needed and the turnover cap is not violated by post-hoc projection.
        """
        self._diag_init()

        cap = float(self.cfg.get("turnover_cap", 1.0))
        prev_raw = prev_w.reindex(self.tickers).fillna(0.0).astype(float)
        tgt_raw  = target_w.reindex(self.tickers).fillna(0.0).astype(float)

        # Ensure both endpoints are feasible first (deterministic projection)
        prev = self.clamp_and_normalize(prev_raw)
        tgt  = self.clamp_and_normalize(tgt_raw)

        if cap <= 0:
            out = prev
            self._diag_emit({
                "event": "turnover",
                "turnover_l1": 0.0,
                "turnover_cap": cap,
                "turnover_clip": 0,
            })
            return out

        delta = tgt - prev
        tot = float(np.abs(delta).sum())
        clipped = bool(tot > cap)

        if not clipped or tot <= 1e-12:
            out = tgt
        else:
            self.cfg["_diag_counts"]["turnover_clip"] += 1
            alpha = cap / tot
            out = prev + alpha * delta  # convex combination; remains feasible

        self._diag_emit({
            "event": "turnover",
            "turnover_l1": tot,
            "turnover_cap": cap,
            "turnover_clip": int(clipped),
        })

        return out

    
    # ==========================================================
    # Diagnostics summary for GUI
    # ==========================================================
    def diagnostics_summary(self) -> dict:
        self._diag_init()
        c = self.cfg["_diag_counts"]

        # Raw counters
        n_calls = int(c.get("n_calls", 0) or 0)
        n_steps = int(c.get("n_steps", 0) or 0)

        # Many non-LLM strategies never perform JSON parsing/repair.
        # Showing 0.0 can be misread as "parsing failed"; use NaN instead.
        supports_json = bool(getattr(self, "USES_LLM_JSON", False))
        if supports_json and n_calls > 0:
            json_valid = c.get("json_valid", 0) / n_calls
            repair = c.get("repair_used", 0) / n_calls
            parse_fail = c.get("parse_failed", 0) / n_calls
        else:
            json_valid = float("nan")
            repair = float("nan")
            parse_fail = float("nan")

        denom_steps = max(n_steps, 1)
        return {
            "strategy": self.__class__.__name__,
            "JSON_valid_rate": json_valid,
            "Repair_rate": repair,
            "Parse_fail_rate": parse_fail,
            "Avg_projection_L1": c.get("sum_proj_l1", 0.0) / denom_steps,
            "Avg_projection_L2": c.get("sum_proj_l2", 0.0) / denom_steps,
            "Cap_clip_freq": c.get("cap_clip", 0) / denom_steps,
            "Turnover_clip_freq": c.get("turnover_clip", 0) / denom_steps,
            "Avg_entropy": c.get("sum_entropy", 0.0) / denom_steps,
            "Avg_herfindahl": c.get("sum_herfindahl", 0.0) / denom_steps,
            "Equal_collapse_rate": c.get("equal_collapse", 0) / denom_steps,
            "Neg_weight_per_step": c.get("neg_weight_count", 0) / denom_steps,
            "Over_cap_per_step": c.get("over_cap_count", 0) / denom_steps,
            "Hallucinated_ticker_per_call": c.get("hallucinated_ticker_count", 0) / max(n_calls, 1),
            "Missing_asset_per_call": c.get("missing_asset_count", 0) / max(n_calls, 1),
            "Invalid_weight_per_call": c.get("invalid_weight_count", 0) / max(n_calls, 1),
            "Negative_weight_per_call": c.get("negative_weight_count", 0) / max(n_calls, 1),
            "Equal_fallback_rate": c.get("equal_fallback", 0) / max(n_calls, 1),
            "Avg_latency_sec": c.get("sum_latency_sec", 0.0) / max(c.get("n_latency", 0), 1),
            "n_calls": n_calls,
            "n_steps": n_steps,
        }

    def diagnostics_timeseries(self) -> pd.DataFrame:
        self._diag_init()
        return pd.DataFrame(self.cfg["_diag_events"])

    def _log_old(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        log_fn = self.cfg.get("log_fn", print)
        try:
            log_fn(f"[{ts}] [{level}] {msg}")
        except Exception:
            try:
                print(f"[{ts}] [{level}] {msg}")
            except Exception:
                pass
        msg_q = self.cfg.get("msg_q")
        msg_info = self.cfg.get("MSG_INFO", "info")
        if msg_q is not None:
            try:
                msg_q.put((msg_info, msg))
            except Exception:
                pass

    def _log(self, msg: str, level: str | int = "INFO", *, t: int | None = None):
        import time
        LV = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
        def _lv(x):
            if isinstance(x, (int, float)):
                return int(x)
            return LV.get(str(x).upper(), 20)
        level_i = _lv(level)
        min_level = _lv(self.cfg.get("log_level", 20))
        gui_min_level = _lv(self.cfg.get("gui_log_level", min_level))
        log_every = int(self.cfg.get("log_every", 0) or 0)
        if log_every > 0 and t is not None and level_i < 30:
            if (t % log_every) != 0:
                return
        if level_i >= min_level:
            ts = time.strftime("%H:%M:%S")
            label = "INFO"
            if level_i <= 10: label = "DEBUG"
            elif level_i >= 40: label = "ERROR"
            elif level_i >= 30: label = "WARN"
            line = f"[{ts}] [{label}] {msg}"
            log_fn = self.cfg.get("log_fn", print)
            if log_fn is not None:
                try:
                    log_fn(line)
                except Exception:
                    try:
                        print(line)
                    except Exception:
                        pass
        if level_i >= gui_min_level:
            msg_q = self.cfg.get("msg_q")
            if msg_q is not None:
                tag_info = self.cfg.get("MSG_INFO", "info")
                tag_warn = self.cfg.get("MSG_WARN", "warn")
                tag_error = self.cfg.get("MSG_ERROR", "error")
                tag = tag_info if level_i < 30 else (tag_warn if level_i < 40 else tag_error)
                try:
                    msg_q.put((tag, msg))
                except Exception:
                    pass

# ======================================================================
# 유틸: 강건 공분산
# ======================================================================
def _robust_cov(returns: pd.DataFrame, min_obs: int = 60, ridge: float = 1e-6) -> pd.DataFrame:
    r = returns.dropna()
    cols = list(returns.columns)
    k = len(cols)
    if len(r) < max(2, min_obs) or k == 0:
        return pd.DataFrame(np.eye(k), index=cols, columns=cols)
    try:
        from sklearn.covariance import LedoitWolf
        C = pd.DataFrame(LedoitWolf().fit(r.values).covariance_, index=cols, columns=cols)
    except Exception:
        C = r.cov()
    C = C.fillna(0.0).values + np.eye(k) * ridge
    return pd.DataFrame(C, index=cols, columns=cols)

# ======================================================================
# 기존 전략들
# ======================================================================
class EqStrategy(Strategy):
    def target_weights(self, t: int) -> pd.Series:
        return pd.Series(1.0 / len(self.tickers), index=self.tickers)

class RiskParityStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        window = self.returns.iloc[t - self.lookback : t]
        vol = np.sqrt(np.diag(_robust_cov(window).values)).clip(1e-12)
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        return self.clamp_and_normalize(pd.Series(w, index=self.tickers))

class MVPStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        C = _robust_cov(self.returns.iloc[t - self.lookback : t])
        try:
            invC = np.linalg.pinv(C.values)
            ones = np.ones(len(self.tickers))
            x = invC @ ones
            x = np.clip(x, 0, None)
            w = x / x.sum() if x.sum() > 0 else ones / len(ones)
        except np.linalg.LinAlgError:
            w = np.ones(len(self.tickers)) / len(self.tickers)
        return self.clamp_and_normalize(pd.Series(w, index=self.tickers))

class Momentum12mStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 252):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        px = self.prices.reindex(columns=self.tickers)
        ret = (px.iloc[t] / px.iloc[t - self.lookback] - 1.0).fillna(0.0)
        k = max(1, len(self.tickers) // 2)
        winners = ret.nlargest(k).index
        w = pd.Series(0.0, index=self.tickers)
        w.loc[list(winners)] = 1.0 / len(winners)
        return self.clamp_and_normalize(w)
    
class Momentum6mStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 126):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        px = self.prices.reindex(columns=self.tickers)
        ret = (px.iloc[t] / px.iloc[t - self.lookback] - 1.0).fillna(0.0)
        k = max(1, len(self.tickers) // 2)
        winners = ret.nlargest(k).index
        w = pd.Series(0.0, index=self.tickers)
        w.loc[list(winners)] = 1.0 / len(winners)
        return self.clamp_and_normalize(w)    

class Trend10mStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 210):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
        self.sma = self.prices.reindex(columns=self.tickers).rolling(self.lookback).mean()
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        active = self.prices.iloc[t] > self.sma.iloc[t]
        active_tickers = active[active].index.tolist()
        if len(active_tickers) == 0:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        w = pd.Series(0.0, index=self.tickers)
        w.loc[active_tickers] = 1.0 / len(active_tickers)
        return self.clamp_and_normalize(w)

class Trend6mStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 126):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
        self.sma = self.prices.reindex(columns=self.tickers).rolling(self.lookback).mean()
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        active = self.prices.iloc[t] > self.sma.iloc[t]
        active_tickers = active[active].index.tolist()
        if len(active_tickers) == 0:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        w = pd.Series(0.0, index=self.tickers)
        w.loc[active_tickers] = 1.0 / len(active_tickers)
        return self.clamp_and_normalize(w)

class LLMStrategy(Strategy):
    USES_LLM_JSON = True

    def __init__(self, tickers, prices, features, fewshot_or_train_feats, cfg):
        super().__init__(tickers, prices, features, cfg)
        self.url = str(cfg.get("ollama_url", "http://localhost:11434"))
        self.model = str(cfg.get("model_name", "gpt-oss:20b"))
        self.use_ollama = bool(cfg.get("use_ollama", True))
        self.log_fn = cfg.get("log_fn", print)
        profile_id = int(self.cfg.get("prompt_profile", 1))
        self.log_fn(f"[LLMStrategy Init] P{profile_id}, model={self.model}")
        try:
            if isinstance(fewshot_or_train_feats, pd.DataFrame):
                fewshots = build_fewshot_db(fewshot_or_train_feats, self.tickers, k=8, profile_id=profile_id, prompt_cap_pct=float(self.cfg.get("prompt_cap_pct", float(self.cfg.get("max_weight", 0.60))*100.0)))
                self.fewshot_block = render_fewshot_block(fewshots, max_k=4)
            elif isinstance(fewshot_or_train_feats, str):
                self.fewshot_block = fewshot_or_train_feats
            else:
                self.fewshot_block = ""
        except Exception as e:
            self.log_fn(f"[WARN] Few-shot build failed (P{profile_id}): {e}. Using empty few-shot block.")
            self.fewshot_block = ""
        self._fewshot_text = self.fewshot_block
        self.log_fn(f"[LLMStrategy Init] Few-shot block length={len(self.fewshot_block)} (P{profile_id})")

    def _feat_value(self, row, asset: str, suffix: str, default: float) -> float:
        try:
            v = float(row.get(f"{asset}_{suffix}", default))
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def _intended_core_asset(self, row, persona_id: int):
        """Persona-level intended asset for prompt-fidelity tests."""
        if persona_id == 1:
            return max(self.tickers, key=lambda a: self._feat_value(row, a, "r12m", -np.inf))
        if persona_id == 2:
            return min(self.tickers, key=lambda a: self._feat_value(row, a, "vol3m", np.inf))
        if persona_id == 3:
            return min(self.tickers, key=lambda a: self._feat_value(row, a, "r3m", np.inf))
        return None

    def target_weights(self, t: int) -> pd.Series:
        self._log(f"target_weights(t={t}) called")
        feats_row = self.features.iloc[t].to_dict() if len(self.features) > t else {}
        if not self.use_ollama:
            self._log("LLM disabled → equal weight")
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        try:
            self._log(f"Calling LLM for target weights at t={t} (P{self.cfg.get('prompt_profile','?')})")
            prompt = build_prompt(feats_row, self.tickers, self.fewshot_block, self.cfg)
            self._log(f"[prompt.len={len(prompt)}] {prompt[:400]}...")
            try:
                out = call_llm(prompt, self.url, self.model, log_fn=self._log)
            except Exception:
                out = call_llm(prompt, self.url, self.model, log_fn=lambda *_a, **_k: None)

            # ---- Diagnostics: record parse/meta from llm.py (NEW) ----
            # ---- Diagnostics: record parse/meta from llm.py (NEW) ----
            try:
                self._diag_init()
                self.cfg["_diag_counts"]["n_calls"] += 1
                meta = llm.LAST_CALL_META
                self.cfg["_diag_counts"]["json_valid"] += int(meta.get("json_valid", 0))
                self.cfg["_diag_counts"]["parse_failed"] += int(meta.get("parse_failed", 0))
                self.cfg["_diag_counts"]["repair_used"] += int(meta.get("repair_used", 0))

                # Reviewer bugfix: aggregate LLM latency and metadata for
                # model-level computational-efficiency diagnostics.
                try:
                    lat = float(meta.get("elapsed_sec", np.nan))
                except Exception:
                    lat = float("nan")
                if np.isfinite(lat):
                    self.cfg["_diag_counts"]["sum_latency_sec"] += lat
                    self.cfg["_diag_counts"]["n_latency"] += 1

                self.cfg["_diag_counts"]["hallucinated_ticker_count"] += int(meta.get("hallucinated_ticker_count", 0) or 0)
                self.cfg["_diag_counts"]["missing_asset_count"] += int(meta.get("missing_asset_count", 0) or 0)
                self.cfg["_diag_counts"]["invalid_weight_count"] += int(meta.get("invalid_weight_count", 0) or 0)
                self.cfg["_diag_counts"]["negative_weight_count"] += int(meta.get("negative_weight_count", 0) or 0)
                self.cfg["_diag_counts"]["equal_fallback"] += int(meta.get("equal_fallback", 0) or 0)

                self._diag_emit({
                    "event": "llm_call",
                    "model": self.model,
                    "persona": f"P{self.cfg.get('prompt_profile','?')}",
                    "json_valid": int(meta.get("json_valid", 0)),
                    "parse_failed": int(meta.get("parse_failed", 0)),
                    "repair_used": int(meta.get("repair_used", 0)),
                    "elapsed_sec": lat,
                    "output_length": meta.get("output_length", np.nan),
                    "hallucinated_ticker_count": int(meta.get("hallucinated_ticker_count", 0) or 0),
                    "missing_asset_count": int(meta.get("missing_asset_count", 0) or 0),
                    "invalid_weight_count": int(meta.get("invalid_weight_count", 0) or 0),
                    "negative_weight_count": int(meta.get("negative_weight_count", 0) or 0),
                    "equal_fallback": int(meta.get("equal_fallback", 0) or 0),
                    "raw_response": str(meta.get("raw_response", ""))[:4000],
                    "parsed_response": str(meta.get("parsed_response", ""))[:4000],
                })
            except Exception:
                pass

            raw_w: Dict[str, Any] | None = None
            if out is None:
                self._log("[LLM] returned None")
            elif hasattr(out, "weights"):
                raw_w = getattr(out, "weights", None)
            elif isinstance(out, dict):
                raw_w = out.get("weights") if "weights" in out else out
            else:
                self._log(f"[LLM] unexpected type: {type(out)} -> {str(out)[:120]}")
            if not isinstance(raw_w, dict):
                raise RuntimeError(f"weights-not-dict: got {type(raw_w)} -> {str(raw_w)[:120]}")
            safe_dict: Dict[str, float] = {}
            for k, v in raw_w.items():
                try:
                    val = float(v)
                    if val < 0:
                        val = 0.0
                    safe_dict[str(k).strip()] = val
                except Exception:
                    continue
            w = pd.Series(safe_dict, dtype=float).reindex(self.tickers).fillna(0.0)
            self._log(f"[LLM RAW|{self.model}|P{self.cfg.get('prompt_profile','?')}] {safe_dict}")

            pid = int(self.cfg.get("prompt_profile", 1) or 1)
            prompt_cap = float(self.cfg.get("prompt_cap_pct", float(self.cfg.get("max_weight", 0.60))*100.0)) / 100.0
            maxw = float(self.cfg.get("max_weight", 0.60))
            core = self._intended_core_asset(feats_row, pid)

            def _equal_l1(vec: pd.Series) -> float:
                eq = pd.Series(1.0 / len(self.tickers), index=self.tickers)
                return float((vec.reindex(self.tickers).fillna(0.0) - eq).abs().sum())

            raw_top = str(w.idxmax()) if len(w) else ""
            if core is not None:
                raw_core_weight = float(w.get(core, 0.0))
                self._diag_emit({
                    "event": "prompt_fidelity",
                    "model": self.model,
                    "prompt_profile": pid,
                    "stage": "raw",
                    "persona": f"P{pid}",
                    "intended_asset": core,
                    "raw_top_asset": raw_top,
                    "raw_fidelity": int(raw_top == core),
                    "raw_core_weight": raw_core_weight,
                    "prompt_cap": prompt_cap,
                    "hard_max_weight": maxw,
                    "raw_threshold_fidelity": int(raw_core_weight >= min(prompt_cap, 1.0) - 1e-6),
                    "raw_threshold_gap": float(raw_core_weight - prompt_cap),
                })
            elif pid == 4:
                self._diag_emit({
                    "event": "prompt_fidelity", "model": self.model, "prompt_profile": pid, "stage": "raw", "persona": f"P{pid}",
                    "raw_equal_l1": _equal_l1(w),
                    "raw_equal_fidelity": int(_equal_l1(w) <= 1e-3),
                })
            elif pid == 5:
                self._diag_emit({
                    "event": "prompt_fidelity", "model": self.model, "prompt_profile": pid, "stage": "raw", "persona": f"P{pid}",
                    "raw_herfindahl": float((w.reindex(self.tickers).fillna(0.0) ** 2).sum()),
                    "raw_nonnegative": int(float(w.min()) >= -1e-12),
                })

            projected = self.clamp_and_normalize(w)
            proj_top = str(projected.idxmax()) if len(projected) else ""
            if core is not None:
                projected_core_weight = float(projected.get(core, 0.0))
                self._diag_emit({
                    "event": "prompt_fidelity",
                    "model": self.model,
                    "prompt_profile": pid,
                    "stage": "projected",
                    "persona": f"P{pid}",
                    "intended_asset": core,
                    "projected_top_asset": proj_top,
                    "projected_fidelity": int(proj_top == core),
                    "projected_core_weight": projected_core_weight,
                    "prompt_cap": prompt_cap,
                    "hard_max_weight": maxw,
                    "projected_threshold_fidelity": int(projected_core_weight >= min(prompt_cap, maxw) - 1e-6),
                    "projected_threshold_gap": float(projected_core_weight - prompt_cap),
                    "fidelity_loss": int(raw_top == core) - int(proj_top == core),
                    "threshold_fidelity_loss": int(float(w.get(core, 0.0)) >= prompt_cap - 1e-6) - int(projected_core_weight >= min(prompt_cap, maxw) - 1e-6),
                })
            elif pid == 4:
                self._diag_emit({
                    "event": "prompt_fidelity", "model": self.model, "prompt_profile": pid, "stage": "projected", "persona": f"P{pid}",
                    "projected_equal_l1": _equal_l1(projected),
                    "projected_equal_fidelity": int(_equal_l1(projected) <= 1e-3),
                })
            elif pid == 5:
                self._diag_emit({
                    "event": "prompt_fidelity", "model": self.model, "prompt_profile": pid, "stage": "projected", "persona": f"P{pid}",
                    "projected_herfindahl": float((projected.reindex(self.tickers).fillna(0.0) ** 2).sum()),
                    "projected_nonnegative": int(float(projected.min()) >= -1e-12),
                })
            # Reviewer appendix examples: raw -> projected allocation.
            self._diag_emit({
                "event": "allocation_example",
                "model": self.model,
                "persona": f"P{pid}",
                "prompt_cap": prompt_cap,
                "hard_max_weight": maxw,
                "raw_top_asset": raw_top,
                "projected_top_asset": proj_top,
                "raw_weights_json": w.to_json(),
                "projected_weights_json": projected.to_json(),
            })
            return projected
        except Exception as e:
            self._log(f"[ERROR] LLMStrategy P{self.cfg.get('prompt_profile')} failed at t={t}: {e}. Fallback.")
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)

# ======================================================================
# Ablation: Coded policy executor (LLM 제거)
# ======================================================================
class CodedPersonaStrategy(Strategy):
    """Deterministic executor for the same persona rules used by the LLM prompts.

    persona_id:
      1: winner-take-cap (12m momentum, max-feasible 60%)
      2: min-vol (3m vol, max-feasible 60%)
      3: contrarian (3m return loser, min 60%)
      4: equal-weight
      5: sharpe-proxy (r1m/vol3m, floor at 0)
    """

    def __init__(self, tickers: List[str], prices: pd.DataFrame, features: pd.DataFrame, cfg: Dict, persona_id: int = 1):
        super().__init__(tickers, prices, features, cfg)
        self.persona_id = int(persona_id)

    def _remainder_equal(self, core: str, min_w: float) -> pd.Series:
        w = pd.Series(0.0, index=self.tickers)
        w.loc[core] = float(min_w)
        rem = max(0.0, 1.0 - float(min_w))
        others = [a for a in self.tickers if a != core]
        if others:
            w.loc[others] = rem / float(len(others))
        return w

    def target_weights(self, t: int) -> pd.Series:
        if self.features is None or t >= len(self.features):
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)

        row = self.features.iloc[t]

        def feat(asset: str, suffix: str) -> float:
            key = f"{asset}_{suffix}"
            try:
                v = float(row.get(key, 0.0))
                return v if np.isfinite(v) else 0.0
            except Exception:
                return 0.0

        pid = self.persona_id

        if pid == 4:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)

        prompt_cap = max(0.0, min(float(self.cfg.get("prompt_cap_pct", float(self.cfg.get("max_weight", 0.60))*100.0)) / 100.0, 1.0))

        if pid == 1:
            scores = {a: feat(a, "r12m") for a in self.tickers}
            core = max(scores, key=scores.get)
            return self._remainder_equal(core, prompt_cap)

        if pid == 2:
            scores = {a: feat(a, "vol3m") for a in self.tickers}
            core = min(scores, key=scores.get)
            return self._remainder_equal(core, prompt_cap)

        if pid == 3:
            scores = {a: feat(a, "r3m") for a in self.tickers}
            core = min(scores, key=scores.get)
            return self._remainder_equal(core, prompt_cap)

        if pid == 5:
            s = pd.Series({
                a: max(feat(a, "r1m") / max(feat(a, "vol3m"), 1e-12), 0.0)
                for a in self.tickers
            })
            if float(s.sum()) <= 1e-12:
                return pd.Series(1.0 / len(self.tickers), index=self.tickers)
            return self.clamp_and_normalize(s / s.sum())

        return pd.Series(1.0 / len(self.tickers), index=self.tickers)


class SharpeWeightedStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        start = max(0, t - self.lookback)
        R = self.returns.iloc[start:t]
        if R.empty:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        mu = R.mean()
        sigma = R.std().replace(0, np.nan)
        sharpe = (mu / sigma).clip(lower=0).fillna(0)
        if sharpe.sum() <= 0:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        return self.clamp_and_normalize(sharpe / sharpe.sum())

class SortinoWeightedStrategy(Strategy):
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(lookback)
    def target_weights(self, t: int) -> pd.Series:
        start = max(0, t - self.lookback)
        R = self.returns.iloc[start:t]
        if R.empty:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        mu = R.mean()
        dsig = R.clip(upper=0).std().replace(0, np.nan)
        sortino = (mu / dsig).clip(lower=0).fillna(0)
        if sortino.sum() <= 0:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        return self.clamp_and_normalize(sortino / sortino.sum())


class MinCVaRStrategy(Strategy):
    """Simple long-only minimum historical CVaR strategy.

    A practical approximation is used for reviewer-response robustness: assets
    with lower historical 5% CVaR receive higher weights, then the vector is
    projected to the capped simplex.
    """
    def __init__(self, tickers, prices, features, cfg, lookback: int = 126, alpha: float = 0.05):
        super().__init__(tickers, prices, features, cfg)
        self.lookback = int(self.cfg.get("lookback_cvar", lookback))
        self.alpha = float(self.cfg.get("cvar_alpha", alpha))
    def target_weights(self, t: int) -> pd.Series:
        start = max(0, t - self.lookback)
        R = self.returns.iloc[start:t].reindex(columns=self.tickers)
        if len(R) < 20:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        losses = -R
        cvars = {}
        for a in self.tickers:
            x = losses[a].dropna().astype(float)
            if x.empty:
                cvars[a] = np.nan
                continue
            q = x.quantile(1.0 - self.alpha)
            tail = x[x >= q]
            cvars[a] = float(tail.mean()) if len(tail) else float(x.max())
        c = pd.Series(cvars).replace([np.inf, -np.inf], np.nan)
        inv = 1.0 / c.clip(lower=1e-6)
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if float(inv.sum()) <= 1e-12:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        return self.clamp_and_normalize(inv / inv.sum())


class VolatilityTargetTrendStrategy(Strategy):
    """Trend-following with inverse-volatility scaling.

    Assets with positive 6-month return are selected and weighted by inverse
    3-month volatility. If no asset has positive trend, the strategy falls back
    to equal weight.
    """
    def __init__(self, tickers, prices, features, cfg, ret_lookback: int = 126, vol_lookback: int = 63):
        super().__init__(tickers, prices, features, cfg)
        self.ret_lookback = int(ret_lookback)
        self.vol_lookback = int(vol_lookback)
        self.lookback = max(self.ret_lookback, self.vol_lookback)
    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        R = self.returns.iloc[max(0,t-self.ret_lookback):t].reindex(columns=self.tickers)
        V = self.returns.iloc[max(0,t-self.vol_lookback):t].reindex(columns=self.tickers).std().replace(0, np.nan)
        trend = (1.0 + R).prod() - 1.0
        active = trend[trend > 0].index.tolist()
        if not active:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        score = (1.0 / V.reindex(active).clip(lower=1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if float(score.sum()) <= 1e-12:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        w = pd.Series(0.0, index=self.tickers)
        w.loc[active] = score / score.sum()
        return self.clamp_and_normalize(w)

# ======================================================================
# 신규 전략들: LW-MVP / HRP / Black–Litterman
# ======================================================================
class LWMVPStrategy(Strategy):
    """
    Ledoit–Wolf Shrinkage MVP (또는 cov_method='sample'로 일반 MVP 대체 가능)
    cfg:
      - lookback: int (기본 60)
      - cov_method: 'ledoitwolf' | 'sample' (기본 'ledoitwolf')
    """
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60, cov_method: str = None):
        super().__init__(tickers, prices, features, cfg)
        self.lookback  = int(self.cfg.get("lookback", lookback))
        self.cov_method = str(self.cfg.get("cov_method", cov_method or "ledoitwolf")).lower()
        if self.cov_method not in ("ledoitwolf", "sample"):
            self.cov_method = "ledoitwolf"

    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        win = self.returns.iloc[t - self.lookback : t].reindex(columns=self.tickers)
        cov = estimate_cov(win, method=self.cov_method)
        try:
            inv = np.linalg.pinv(cov.values)
            ones = np.ones((cov.shape[0], 1))
            w = inv @ ones
            denom = (ones.T @ inv @ ones).item()
            if denom <= 0:
                w = np.ones((cov.shape[0], 1)) / cov.shape[0]
            else:
                w = w / denom
            w = np.clip(w.ravel(), 0.0, None)
        except Exception:
            w = np.ones(len(self.tickers)) / len(self.tickers)
        return self.clamp_and_normalize(pd.Series(w, index=self.tickers, name="LW-MVP" if self.cov_method=="ledoitwolf" else "MVP"))

class HRPStrategy(Strategy):
    """
    Hierarchical Risk Parity (Lopez de Prado)
    cfg:
      - lookback: int (기본 60)
      - cov_method: 'sample' | 'ledoitwolf' (기본 'sample')
    """
    def __init__(self, tickers, prices, features, cfg, lookback: int = 60, cov_method: str = None):
        super().__init__(tickers, prices, features, cfg)
        self.lookback  = int(self.cfg.get("lookback", lookback))
        self.cov_method = str(self.cfg.get("cov_method", cov_method or "sample")).lower()
        if self.cov_method not in ("ledoitwolf", "sample"):
            self.cov_method = "sample"

    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        win = self.returns.iloc[t - self.lookback : t].reindex(columns=self.tickers)
        cov = estimate_cov(win, method=self.cov_method)
        try:
            w = hrp_alloc(cov).reindex(self.tickers).fillna(0.0).values
            w = np.clip(w, 0.0, None)
        except Exception:
            w = np.ones(len(self.tickers)) / len(self.tickers)
        return self.clamp_and_normalize(pd.Series(w, index=self.tickers, name=f"HRP({self.cov_method})"))

class BLStrategy(Strategy):
    """
    Black–Litterman (간단 구현)
    cfg:
      - lookback: int (기본 60)
      - cov_method: 'ledoitwolf' | 'sample' (기본 'ledoitwolf')
      - bl_delta: float (리스크 기피도 δ, 기본 2.5)
      - bl_tau: float (τ, 기본 0.05)
      - bl_omega_scale: float (Ω 스케일, 기본 1.0)
      - bl_market_weights: dict|list|pd.Series (자산별 시장가중치; 없으면 균등)
      - bl_P: list[list]|np.ndarray|pd.DataFrame  (K x N)
      - bl_Q: list|np.ndarray|pd.Series          (K, )
    참고:
      μ_post = [ (τΣ)^(-1) + P^T Ω^(-1) P ]^(-1) [ (τΣ)^(-1) π + P^T Ω^(-1) Q ],
      π = δ Σ w_mkt,  w ∝ Σ^{-1} μ_post → long-only/합1는 clamp_and_normalize로 처리
    """
    def __init__(self, tickers, prices, features, cfg,
                 lookback: int = 60,
                 cov_method: str = None):
        super().__init__(tickers, prices, features, cfg)
        self.lookback   = int(self.cfg.get("lookback", lookback))
        self.cov_method = str(self.cfg.get("cov_method", cov_method or "ledoitwolf")).lower()
        if self.cov_method not in ("ledoitwolf", "sample"):
            self.cov_method = "ledoitwolf"
        self.delta       = float(self.cfg.get("bl_delta", 2.5))
        self.tau         = float(self.cfg.get("bl_tau", 0.05))
        self.omega_scale = float(self.cfg.get("bl_omega_scale", 1.0))
        self._mw_raw     = self.cfg.get("bl_market_weights", None)
        self._P_raw      = self.cfg.get("bl_P", None)
        self._Q_raw      = self.cfg.get("bl_Q", None)

    @staticmethod
    def _to_array(x) -> Optional[np.ndarray]:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (list, tuple)):
            return np.array(x, dtype=float)
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.values.astype(float)
        return None

    def _parse_market_weights(self, assets: List[str]) -> np.ndarray:
        N = len(assets)
        if self._mw_raw is None:
            return np.ones(N) / N
        if isinstance(self._mw_raw, pd.Series):
            w = self._mw_raw.reindex(assets).fillna(0.0).values
        elif isinstance(self._mw_raw, dict):
            w = pd.Series(self._mw_raw).reindex(assets).fillna(0.0).values
        elif isinstance(self._mw_raw, (list, tuple, np.ndarray)):
            w = np.asarray(self._mw_raw, dtype=float)
            if w.shape[0] != N:
                w = np.ones(N) / N
        else:
            w = np.ones(N) / N
        s = w.sum()
        if s <= 0:
            return np.ones(N)/N
        return np.clip(w, 0.0, None) / s

    def _parse_PQ(self, N: int):
        P = self._to_array(self._P_raw)
        Q = self._to_array(self._Q_raw)
        if P is None or Q is None:
            return None, None
        P = np.atleast_2d(P.astype(float))
        Q = np.atleast_1d(Q.astype(float)).reshape(-1, 1)
        if P.shape[1] != N or Q.shape[0] != P.shape[0]:
            # 차원 불일치면 뷰를 사용하지 않음
            return None, None
        return P, Q

    @staticmethod
    def _pinv(A: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(A)

    def target_weights(self, t: int) -> pd.Series:
        if t < self.lookback:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        assets = list(self.tickers)
        win = self.returns.iloc[t - self.lookback : t].reindex(columns=assets)
        cov = estimate_cov(win, method=self.cov_method).loc[assets, assets]
        Sigma = cov.values
        N = Sigma.shape[0]

        # 시장가중치
        w_mkt = self._parse_market_weights(assets)

        # 균형수익 π
        pi = self.delta * (Sigma @ w_mkt)

        # Priors
        tauSigma = self.tau * Sigma

        # 뷰
        P, Q = self._parse_PQ(N)

        if P is None or Q is None:
            mu_post = pi
        else:
            # Omega: diag(P τΣ P^T) * omega_scale
            Omega = np.diag(np.diag(P @ tauSigma @ P.T)) * max(self.omega_scale, 1e-12)
            inv_tauSigma = self._pinv(tauSigma)
            inv_Omega = self._pinv(Omega)
            M = inv_tauSigma + P.T @ inv_Omega @ P
            b = inv_tauSigma @ pi.reshape(-1, 1) + P.T @ inv_Omega @ Q
            mu_post = self._pinv(M) @ b
            mu_post = mu_post.ravel()

        # 최종 가중치: w ∝ Σ^{-1} μ_post
        try:
            invSigma = self._pinv(Sigma)
            w = invSigma @ mu_post
            w = np.clip(w, 0.0, None)
        except Exception:
            w = np.ones(N) / N

        return self.clamp_and_normalize(pd.Series(w, index=assets, name="Black-Litterman"))
