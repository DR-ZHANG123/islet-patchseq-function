"""Stage 09b：胞吐的有符号值与左删失模型。

HumanIslets 发布的细胞层面胞吐（ephys_cell.csv）已把负值截为 0，原始有符号值不存在。
可利用恒等式 总胞吐 = 早期 + 晚期 部分恢复：
  - 晚期 = 0 且总胞吐 > 0 时，有符号晚期 = 总 − 早期（≤ 0，可观测）；
  - 总胞吐 = 0 时，有符号总值 ≤ 0（左删失于 0），有符号晚期 ≤ −早期（早期 > 0 时）或 ≤ 0（左删失）。
有符号总胞吐无法恢复（总值为 0 的细胞只知道 ≤ 0）。
模型：随机截距 Tobit（左删失，Gauss–Hermite 积分），在 asinh 尺度（兼容负值、正值端近似对数）与原始尺度各拟合一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

from t2d_models import design


def signed_audit(raw: pd.DataFrame, tol: float) -> dict:
    """统计发布文件中负值个数与 总 = 早 + 晚 恒等式的成立情况（证明截断发生在发布之前）。"""
    t, a, l = (raw[c] for c in ("total_exocytosis_fF_pF_cell", "early_exocytosis_fF_pF_cell",
                                "late_exocytosis_fF_pF_cell"))
    ok = t.notna() & a.notna() & l.notna()
    d = (t - a - l)[ok].abs()
    allpos = ok & (t > 0) & (a > 0) & (l > 0)
    return {"n_cells_file": int(len(raw)), "n_negative_total": int((t < 0).sum()), "n_negative_early": int((a < 0).sum()),
            "n_negative_late": int((l < 0).sum()), "n_zero_total": int((t == 0).sum()),
            "n_zero_early": int((a == 0).sum()), "n_zero_late": int((l == 0).sum()),
            "identity_holds_all_positive": float((d[allpos[ok]] < tol).mean()),
            "identity_fails_with_zero_component": int(((d >= tol) & ((a[ok] == 0) | (l[ok] == 0))).sum())}


def signed_values(df: pd.DataFrame, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (值, 是否删失)。删失细胞的值为上界。df 需含 exocytosis / early_exocytosis / late_exocytosis（≥ 0）。"""
    t, a, l = df.exocytosis.values, df.early_exocytosis.values, df.late_exocytosis.values
    if endpoint == "exocytosis":
        return t.copy(), t == 0
    if endpoint == "late_exocytosis":
        v, cens = l.copy(), np.zeros(len(l), bool)
        rec = (l == 0) & (t > 0)
        v[rec] = np.minimum(t[rec] - a[rec], 0.0)
        c0 = (l == 0) & (t == 0)
        v[c0] = -a[c0]                     # 上界：−早期（早期为 0 时即 0）
        cens[c0] = True
        return v, cens
    if endpoint == "early_exocytosis":
        v, cens = a.copy(), np.zeros(len(a), bool)
        rec = (a == 0) & (t > 0)
        v[rec] = np.minimum(t[rec] - l[rec], 0.0)
        c0 = (a == 0) & (t == 0)
        v[c0] = -l[c0]
        cens[c0] = True
        return v, cens
    raise KeyError(endpoint)


def ri_tobit(y: np.ndarray, cens: np.ndarray, X: np.ndarray, groups: np.ndarray, nodes: int = 20) -> dict:
    """随机截距 Tobit：y* = Xβ + s·u_d + σε；观测细胞贡献密度，删失细胞贡献 P(y* ≤ 上界)。"""
    _, g = np.unique(groups, return_inverse=True)
    D = g.max() + 1
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    logw = np.log(w / np.sqrt(2 * np.pi))
    G = sp.csr_matrix((np.ones(len(y)), (g, np.arange(len(y)))), shape=(D, len(y)))
    obs = ~cens

    def nll(theta):
        beta, s, sig = theta[:-2], np.exp(theta[-2]), np.exp(theta[-1])
        r = (y[:, None] - (X @ beta)[:, None] - s * z[None, :]) / sig
        ll = np.where(obs[:, None], -0.5 * r ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sig), log_ndtr(r))
        return -float(logsumexp(G @ ll + logw[None, :], axis=1).sum())

    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    sd0 = max(float(np.std(y - X @ b0)), 1e-3)
    th0 = np.r_[b0, np.log(0.5 * sd0), np.log(sd0)]
    bounds = [(None, None)] * X.shape[1] + [(np.log(sd0) - 8, np.log(sd0) + 3)] * 2
    res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
    return {"beta": res.x[:-2], "sd_u": float(np.exp(res.x[-2])), "sigma": float(np.exp(res.x[-1])),
            "converged": bool(res.success)}


def tobit_effects(df: pd.DataFrame, spec: dict) -> dict:
    """有符号终点的 T2D 效应：asinh 尺度系数，以及原始尺度的潜变量均值差（fF/pF）和 T2D/ND 潜变量均值比。"""
    X, names = design(df)
    j = names.index("t2d")
    cens = df.cens.values.astype(bool)
    out = {}
    fa = ri_tobit(np.arcsinh(df.ys.values), cens, X, df.donor.values, spec["gh_nodes"])
    out["tobit_asinh_coef"] = fa["beta"][j]
    fr = ri_tobit(df.ys.values, cens, X, df.donor.values, spec["gh_nodes"])
    out["tobit_raw_diff"] = fr["beta"][j]
    nd = df.status.values == "ND"
    m_nd = float(np.mean(X[nd] @ fr["beta"]))
    out["tobit_raw_logratio"] = np.log((m_nd + fr["beta"][j]) / m_nd) if m_nd > 0 and m_nd + fr["beta"][j] > 0 \
        else np.nan
    return out
