"""带截距的岭回归，一次 SVD 求解整条惩罚路径。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgePath:
    x_mean: np.ndarray
    y_mean: float
    vt: np.ndarray        # (k, p)
    s: np.ndarray         # (k,)
    uty: np.ndarray       # (k,)

    def coef(self, alpha: float) -> np.ndarray:
        d = self.s / (self.s ** 2 + alpha)
        return self.vt.T @ (d * self.uty)

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        return (x - self.x_mean) @ self.coef(alpha) + self.y_mean

    def predict_path(self, x: np.ndarray, alphas) -> np.ndarray:
        """返回 (n, len(alphas)) 预测矩阵。"""
        proj = (x - self.x_mean) @ self.vt.T            # (n, k)
        d = self.s[None, :] / (self.s[None, :] ** 2 + np.asarray(alphas)[:, None])  # (a, k)
        return proj @ (d * self.uty[None, :]).T + self.y_mean


def fit_path(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> RidgePath:
    """weights：样本权重（例如按供体平衡）；用 sqrt(w) 缩放后做普通最小二乘的 SVD。"""
    if weights is None:
        weights = np.ones(len(y))
    w = weights / weights.mean()
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = float((w * y).sum() / w.sum())
    sw = np.sqrt(w)
    xc = (x - xm) * sw[:, None]
    yc = (y - ym) * sw
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return RidgePath(xm, ym, vt, s, u.T @ yc)


def fit_augmented(x: np.ndarray, y: np.ndarray, xc: np.ndarray, yc: np.ndarray, lam: float,
                  weights: np.ndarray | None = None) -> RidgePath:
    """供体内差分监督：最小化 ||y - Xb||² + lam·||yc - Xc b||² + alpha||b||²。
    Xc、yc 为比较组内去均值后的特征与终点；截距只由第一项决定（Xc、yc 组内均值为 0）。"""
    if lam <= 0:
        return fit_path(x, y, weights)
    if weights is None:
        weights = np.ones(len(y))
    w = weights / weights.mean()
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = float((w * y).sum() / w.sum())
    sw = np.sqrt(w)
    a = np.vstack([(x - xm) * sw[:, None], np.sqrt(lam) * xc * sw[:, None]])
    b = np.concatenate([(y - ym) * sw, np.sqrt(lam) * yc * sw])
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return RidgePath(xm, ym, vt, s, u.T @ b)
