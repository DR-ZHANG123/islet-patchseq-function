"""供体层面模型：以供体同类型细胞的伪 bulk 表达预测该供体的平均电生理表型。

- 伪 bulk：同供体、同类型全部细胞计数求和 → log1p(CPM)。
- 供体层面终点：细胞终点先在每个葡萄糖浓度内标准化（训练细胞的均值与标准差），再按供体取均值（≥ min_cells 个细胞）。
- 跨队列应用时，基因在各自队列内跨供体标准化（队列内 z 分数），模型因此只依赖供体之间的相对差异。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ridge import fit_path


def pseudobulk(counts: np.ndarray, donors: np.ndarray) -> pd.DataFrame:
    """counts：细胞 × 基因（整数）；返回供体 × 基因的 log1p(CPM)。"""
    df = pd.DataFrame(counts).groupby(donors).sum()
    lib = df.sum(1).replace(0, 1)
    return np.log1p(df.div(lib, axis=0) * 1e6)


def donor_targets(y: np.ndarray, donor: np.ndarray, glucose: np.ndarray, train_mask: np.ndarray,
                  min_cells: int) -> pd.DataFrame:
    """葡萄糖浓度内标准化（只用训练细胞的统计量）后按供体平均。"""
    z = np.full(len(y), np.nan)
    for g in np.unique(glucose):
        m = glucose == g
        ref = y[m & train_mask & ~np.isnan(y)]
        if len(ref) < 5:
            continue
        z[m] = (y[m] - ref.mean()) / (ref.std() + 1e-8)
    df = pd.DataFrame({"donor": donor, "z": z}).dropna()
    agg = df.groupby("donor").z.agg(["mean", "size"]).rename(columns={"mean": "y", "size": "n"})
    return agg[agg.n >= min_cells]


def select_genes(pb_train: pd.DataFrame, n_genes: int, min_logcpm: float) -> np.ndarray:
    mean = pb_train.mean(0)
    var = pb_train.var(0)
    cand = var[mean > min_logcpm].sort_values(ascending=False)
    return cand.index[:n_genes].values


def cohort_z(pb: pd.DataFrame, genes, ref: pd.DataFrame | None = None) -> np.ndarray:
    """基因在参照队列（缺省为自身）内跨供体标准化。"""
    ref = pb if ref is None else ref
    m = ref[genes].mean(0)
    s = ref[genes].std(0).replace(0, 1) + 1e-6
    return ((pb[genes] - m) / s).values


def fit_donor_model(x: np.ndarray, y: np.ndarray, w: np.ndarray, alphas, k_inner: int, seed: int):
    """岭回归，惩罚系数由供体层面的内层 k 折交叉验证选择；返回 (RidgePath, alpha)。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % k_inner
    err = np.zeros(len(alphas))
    for f in range(k_inner):
        tr, va = fold != f, fold == f
        path = fit_path(x[tr], y[tr], w[tr])
        err += (w[va][:, None] * (path.predict_path(x[va], alphas) - y[va][:, None]) ** 2).sum(0)
    alpha = float(np.asarray(alphas)[int(np.argmin(err))])
    return fit_path(x, y, w), alpha
