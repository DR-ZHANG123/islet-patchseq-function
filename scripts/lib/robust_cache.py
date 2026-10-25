"""进程内计算复用：岭回归的 SVD 只取决于特征矩阵（与终点无关），梯度提升树只取决于（特征, 终点, 种子）。

置换零模型、扩展惩罚网格与留一参照在同一外层折上反复拟合同一特征矩阵；把 SVD 与梯度提升树按内容哈希缓存，
结果与逐次重新计算逐位一致（同一 LAPACK 调用、同一随机种子），只省去重复计算。
install() 替换 context_model 模块中的 fit_path / fit_augmented 与 ContextModel._fit_boost。
"""
from __future__ import annotations

import hashlib

import numpy as np

import context_model as cm
from ridge import RidgePath

_SVD: dict = {}
_HGB: dict = {}
_ORIG_BOOST = cm.ContextModel._fit_boost


def _h(*arrs) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrs:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _decomp(key, build):
    if key not in _SVD:
        _SVD[key] = build()
    return _SVD[key]


def fit_path(x, y, weights=None):
    w = np.ones(len(y)) if weights is None else weights
    w = w / w.mean()

    def build():
        xm = (w[:, None] * x).sum(0) / w.sum()
        sw = np.sqrt(w)
        u, s, vt = np.linalg.svd((x - xm) * sw[:, None], full_matrices=False)
        return xm, u, s, vt

    xm, u, s, vt = _decomp(("p", _h(x, weights)), build)
    ym = float((w * y).sum() / w.sum())
    return RidgePath(xm, ym, vt, s, u.T @ ((y - ym) * np.sqrt(w)))


def fit_augmented(x, y, xc, yc, lam, weights=None):
    if lam <= 0:
        return fit_path(x, y, weights)
    w = np.ones(len(y)) if weights is None else weights
    w = w / w.mean()
    sw = np.sqrt(w)

    def build():
        xm = (w[:, None] * x).sum(0) / w.sum()
        a = np.vstack([(x - xm) * sw[:, None], np.sqrt(lam) * xc * sw[:, None]])
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        return xm, u, s, vt

    xm, u, s, vt = _decomp(("a", lam, _h(x, xc, weights)), build)
    ym = float((w * y).sum() / w.sum())
    b = np.concatenate([(y - ym) * sw, np.sqrt(lam) * yc * sw])
    return RidgePath(xm, ym, vt, s, u.T @ b)


def _fit_boost(self, z, g1, y, idx):
    if not self.use_boosting:
        return None
    key = (self.seed, self.nb, _h(z[idx, :self.nb], g1[idx], y[idx]))
    if key not in _HGB:
        _HGB[key] = _ORIG_BOOST(self, z, g1, y, idx)
    return _HGB[key]


def install() -> None:
    cm.fit_path = fit_path
    cm.fit_augmented = fit_augmented
    cm.ContextModel._fit_boost = _fit_boost


def clear(svd: bool = True, hgb: bool = True) -> None:
    if svd:
        _SVD.clear()
    if hgb:
        _HGB.clear()
