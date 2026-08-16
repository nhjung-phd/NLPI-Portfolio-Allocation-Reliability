# engine/cov_utils.py
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.cluster import AgglomerativeClustering

# --- 공분산 추정 ---
def estimate_cov(returns: pd.DataFrame, method: str = "sample"):
    """
    returns: (T x N) 수익률 DataFrame
    method: 'sample' | 'ledoitwolf'
    """
    X = returns.values
    if method == "ledoitwolf":
        lw = LedoitWolf().fit(X)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    # sample
    cov = np.cov(X, rowvar=False, ddof=1)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

# --- HRP 보조 유틸 ---
def correl_dist(corr: np.ndarray) -> np.ndarray:
    # López de Prado (2016) 거리 변환: d_ij = sqrt(0.5 * (1 - corr_ij))
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    return dist

def seriation(Z, cur_index):
    # 군집 결과를 순서화(퀘이지 대각화 목적), 간단 DFS
    if cur_index < Z.shape[0]:
        return [cur_index]
    left = int(Z[cur_index - Z.shape[0], 0])
    right = int(Z[cur_index - Z.shape[0], 1])
    return seriation(Z, left) + seriation(Z, right)

def _hierarchical_order(corr: pd.DataFrame) -> list:
    # sklearn Agglomerative를 쓰되, linkage info를 numpy 형태로 재구성
    # (간단 구현: 'ward' 대신 'average' 권장. affinity='precomputed'를 위해 거리 사용)
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage
    dist = correl_dist(corr.values)
    Z = linkage(squareform(dist, checks=False), method='average')
    # root index는 2N-2
    order = seriation(Z, Z.shape[0] + corr.shape[0] - 2)
    return order

def get_quasi_diag(cov: pd.DataFrame) -> list:
    corr = cov.corr()
    order = _hierarchical_order(corr)
    return list(cov.index[order])

def get_ivp(cov: pd.DataFrame) -> np.ndarray:
    # inverse-variance weights
    iv = 1. / np.diag(cov.values)
    w = iv / iv.sum()
    return w

def hrp_alloc(cov: pd.DataFrame) -> pd.Series:
    # Recursive bisection
    items = list(cov.index)
    sort_ix = get_quasi_diag(cov)
    w = pd.Series(1.0, index=sort_ix)

    def _cluster_var(cov_sub):
        w_ivp = get_ivp(cov_sub)
        cvar = np.dot(w_ivp, np.dot(cov_sub.values, w_ivp))
        return cvar

    def _split_weights(items_sorted):
        if len(items_sorted) == 1:
            return
        split = len(items_sorted) // 2
        left_items = items_sorted[:split]
        right_items = items_sorted[split:]
        cov_left = cov.loc[left_items, left_items]
        cov_right = cov.loc[right_items, right_items]
        var_left = _cluster_var(cov_left)
        var_right = _cluster_var(cov_right)
        alpha = 1.0 - var_left / (var_left + var_right)
        w[left_items] *= alpha
        w[right_items] *= (1.0 - alpha)
        _split_weights(left_items)
        _split_weights(right_items)

    _split_weights(sort_ix)
    # 정규화
    w = w / w.sum()
    return w.reindex(items)
