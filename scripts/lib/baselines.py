"""对照方法：预测基因集 + kNN（按 Camunas-Soler 等 2020 的思路在训练折内重建）、弹性网、PLS、梯度提升树。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _rank_rows(a: np.ndarray) -> np.ndarray:
    r = np.apply_along_axis(rankdata, 1, a)
    r = r - r.mean(1, keepdims=True)
    return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)


def knn_predictive_set(xlog: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                       n_genes: int, k: int, min_mean: float) -> np.ndarray:
    """训练折内：按与终点的 Spearman 相关绝对值选预测基因集；测试细胞与训练细胞在该基因集上的
    Spearman 相似度取前 k 个近邻，预测为近邻终点均值。"""
    xt = xlog[tr]
    expr = np.where(xt.mean(0) > min_mean)[0]
    rx = np.apply_along_axis(rankdata, 0, xt[:, expr])
    rx = rx - rx.mean(0)
    ry = rankdata(y[tr]) - (len(tr) + 1) / 2
    rho = (rx * ry[:, None]).sum(0) / (np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum()) + 1e-12)
    sel = expr[np.argsort(-np.abs(np.nan_to_num(rho)))[:n_genes]]
    a = _rank_rows(xlog[np.ix_(tr, sel)])
    b = _rank_rows(xlog[np.ix_(te, sel)])
    sim = b @ a.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    return y[tr][nn].mean(1)


def elastic_net(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, inner: list) -> np.ndarray:
    m = ElasticNetCV(l1_ratio=0.5, n_alphas=12, cv=inner, max_iter=3000, tol=1e-3, n_jobs=1)
    return m.fit(f[tr], y[tr]).predict(f[te])


def pls(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, inner: list, comps=(2, 4, 6, 8, 10)) -> np.ndarray:
    errs = []
    for c in comps:
        e = 0.0
        for itr, iva in inner:
            p = PLSRegression(c, scale=False).fit(f[tr][itr], y[tr][itr]).predict(f[tr][iva]).ravel()
            e += ((p - y[tr][iva]) ** 2).sum()
        errs.append(e)
    c = comps[int(np.argmin(errs))]
    return PLSRegression(c, scale=False).fit(f[tr], y[tr]).predict(f[te]).ravel()


def gradient_boosting(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                      random_state=seed)
    return m.fit(f[tr], y[tr]).predict(f[te])


def inner_splits(donor_tr: np.ndarray, k: int, seed: int) -> list:
    ud = np.unique(donor_tr)
    rng = np.random.default_rng(seed)
    perm = dict(zip(ud, rng.permutation(len(ud))))
    fold = np.array([perm[d] % k for d in donor_tr])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]
