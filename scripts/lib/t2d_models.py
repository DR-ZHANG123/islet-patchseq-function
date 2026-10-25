"""Stage 09 用到的统计模型：随机截距线性混合模型（REML）、随机截距 logistic 模型（Gauss–Hermite 积分）、
加权分位数回归（线性规划精确解），以及设计矩阵构建。全部为确定性算法，便于按供体重抽样与置换时重复调用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import expit, logsumexp


def design(df: pd.DataFrame, covariates: tuple = ()) -> tuple[np.ndarray, list]:
    """截距、t2d（第 2 列）、糖尿病前期、葡萄糖与来源中心哑变量、标准化的连续/二值协变量；去掉常数列与共线列。"""
    cols = {"intercept": np.ones(len(df)), "t2d": df.t2d.values, "pre": df.pre.values}
    for var in ("glucose", "site"):
        levels = sorted(df[var].unique(), key=str)
        for lv in levels[1:]:
            cols[f"{var}[{lv}]"] = (df[var].values == lv).astype(float)
    for cv in covariates:
        v = df[cv].values.astype(float)
        cols[cv] = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
    names, mats = [], []
    for nm, v in cols.items():
        if nm != "intercept" and np.ptp(v) == 0:
            continue
        trial = np.column_stack(mats + [v])
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            mats.append(v)
            names.append(nm)
    if "t2d" not in names:
        raise ValueError("T2D indicator not estimable in this subset")
    return np.column_stack(mats), names


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, g = np.unique(groups, return_inverse=True)
    return g, np.bincount(g)


def ri_lmm(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """随机截距线性混合模型 y = Xβ + u_d + e，REML 轮廓似然对 λ = τ²/σ² 一维优化。"""
    g, n_d = _group_index(groups)
    N, p = X.shape

    def transformed(lam):
        theta = 1.0 - 1.0 / np.sqrt(1.0 + n_d * lam)
        ybar = np.bincount(g, y) / n_d
        xbar = np.column_stack([np.bincount(g, X[:, j]) / n_d for j in range(p)])
        return y - theta[g] * ybar[g], X - theta[g, None] * xbar[g]

    def neg_reml(loglam):
        lam = np.exp(loglam)
        ys, xs = transformed(lam)
        beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        rss = float(np.sum((ys - xs @ beta) ** 2))
        logdet = np.linalg.slogdet(xs.T @ xs)[1]
        return 0.5 * ((N - p) * np.log(rss / (N - p)) + np.sum(np.log1p(n_d * lam)) + logdet)

    opt = minimize_scalar(neg_reml, bounds=(-12.0, 6.0), method="bounded", options={"xatol": 1e-4})
    lam = float(np.exp(opt.x))
    ys, xs = transformed(lam)
    beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
    sigma2 = float(np.sum((ys - xs @ beta) ** 2) / (N - p))
    return {"beta": beta, "sigma2": sigma2, "tau2": lam * sigma2}


def ri_logit(y: np.ndarray, X: np.ndarray, groups: np.ndarray, nodes: int = 30, ridge: float = 0.01) -> dict:
    """随机截距 logistic 模型 logit P(y=1) = Xβ + s·u_d，u_d ~ N(0,1)；非自适应 Gauss–Hermite 积分求边际似然。
    对非截距系数加很弱的 L2 罚（ridge）以防小子集中的完全分离。"""
    g, _ = _group_index(groups)
    D = g.max() + 1
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    logw = np.log(w / np.sqrt(2 * np.pi))
    G = sp.csr_matrix((np.ones(len(y)), (g, np.arange(len(y)))), shape=(D, len(y)))
    pen = np.r_[0.0, np.full(X.shape[1] - 1, ridge)]

    def f(theta):
        beta, s = theta[:-1], np.exp(theta[-1])
        eta = (X @ beta)[:, None] + s * z[None, :]
        ll = y[:, None] * eta - np.logaddexp(0.0, eta)
        S = G @ ll + logw[None, :]
        lse = logsumexp(S, axis=1)
        post = np.exp(S - lse[:, None])[g]
        r = (y[:, None] - expit(eta)) * post
        gb = X.T @ r.sum(1) - 2 * pen * beta
        gs = s * float(np.sum(r * z[None, :]))
        return -(lse.sum() - np.sum(pen * beta ** 2)), -np.r_[gb, gs]

    m0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    th0 = np.r_[np.log(m0 / (1 - m0)), np.zeros(X.shape[1] - 1), np.log(0.5)]
    bounds = [(None, None)] * X.shape[1] + [(-6.0, 2.5)]
    res = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    return {"beta": res.x[:-1], "sd_u": float(np.exp(res.x[-1])), "converged": bool(res.success)}


def wquantreg(y: np.ndarray, X: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """加权线性分位数回归：min Σ w_i ρ_q(y_i − x_iβ)，HiGHS 内点法 + crossover 得精确顶点解。"""
    n, p = X.shape
    A = sp.hstack([sp.csr_matrix(X), sp.eye(n), -sp.eye(n)]).tocsc()
    c = np.r_[np.zeros(p), w * q, w * (1 - q)]
    res = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs-ipm")
    if res.status != 0:
        raise ValueError(f"quantile regression failed: {res.message}")
    return res.x[:p]


def weighted_mean_by(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg 校正（忽略 NaN）。"""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / np.arange(1, len(v) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(v)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out
