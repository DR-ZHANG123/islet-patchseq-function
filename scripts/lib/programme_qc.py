"""Stage 08b：控制质控协变量后的供体内基因–终点关联。

与 Stage 08 的供体内关联相同（比较组 = 供体 × 葡萄糖，组内取秩并去均值，组内置换估计零分布标准差，
z = r / SD_null，正态尾 P 值，BH 校正），只是表达秩与终点秩在相关之前都对质控协变量（组内去均值、标准化）
做线性回归取残差，即偏相关；置换在残差化后的终点上于比较组内进行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from statsmodels.stats.multitest import multipletests

from robust_qc import qc_features



def centered_ranks(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """与 Stage 08 相同：比较组内取秩（除以组大小）并去均值。"""
    out = np.zeros_like(a, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        r = rankdata(a[m], axis=0) / m.sum()
        out[m] = r - r.mean(0)
    return out


def covariate_matrix(td, cfg: dict, ct: str, endpoint: str) -> tuple[np.ndarray, list]:
    """质控协变量（log 检出基因数、log 计数、线粒体比例、本类激素比例、他类激素 log1p CPM）；
    离子电流类终点另加 log 细胞大小。"""
    pc = cfg["programme_qc"]
    q = qc_features(td, cfg, ct)
    cols = []
    for c in pc["qc_covariates"]:
        if c == "ambient":
            cols += [k for k in q.columns if k.startswith("ambient_")]
        else:
            cols.append(c)
    m = q[cols].copy()
    if endpoint in pc["size_endpoints"]:
        m["log_size"] = np.log(td.cells["size"].values.astype(float))
        cols.append("log_size")
    return m.values.astype(float), cols


def _residualize(a: np.ndarray, Q: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(Q, a, rcond=None)
    return a - Q @ coef


def partial_within_assoc(x, y, qmat, groups, n_perm, rng):
    """返回 (r, z, p)；x 为细胞 × 基因表达，y 为终点，qmat 为细胞 × 协变量。"""
    keep = pd.Series(groups).map(pd.Series(groups).value_counts()).values >= 4
    keep &= np.isfinite(qmat).all(1)
    x, y, g, qm = x[keep], y[keep], groups[keep], qmat[keep]
    xr, yr = centered_ranks(x, g), centered_ranks(y[:, None], g)[:, 0]
    qc = qm - pd.DataFrame(qm).groupby(g).transform("mean").values
    qc = (qc - qc.mean(0)) / (qc.std(0) + 1e-12)
    Q = np.c_[np.ones(len(qc)), qc]
    xr, yr = _residualize(xr, Q), _residualize(yr, Q)
    nx = np.sqrt((xr ** 2).sum(0)) + 1e-12
    r = (xr * yr[:, None]).sum(0) / (nx * np.sqrt((yr ** 2).sum()))
    # 置换后的终点按批组成矩阵，用矩阵乘法计算零分布（与逐次循环等价，随机数消耗顺序相同）
    null = np.zeros((n_perm, x.shape[1]))
    idx_by_g = [np.where(g == k)[0] for k in np.unique(g)]
    batch = 100
    for b0 in range(0, n_perm, batch):
        yp = np.repeat(yr[:, None], min(batch, n_perm - b0), axis=1)
        for j in range(yp.shape[1]):
            for ix in idx_by_g:
                yp[ix, j] = yr[rng.permutation(ix)]
        null[b0:b0 + yp.shape[1]] = (yp.T @ xr) / (nx[None, :] * np.sqrt((yp ** 2).sum(0))[:, None])
    z = r / (null.std(0) + 1e-12)
    return r, z, 2 * norm.sf(np.abs(z)), int(keep.sum())


def compare_genes(orig: pd.DataFrame, adj: pd.DataFrame, genes: list, fdr: float = 0.05) -> pd.DataFrame:
    """特征基因调整前后：r、z、FDR，符号是否改变、显著性（FDR < 0.05）是否改变。"""
    o = orig.set_index("gene").loc[genes, ["within_r", "within_z", "within_fdr"]]
    a = adj.set_index("gene").loc[genes, ["within_r", "within_z", "within_fdr"]]
    t = o.add_suffix("_original").join(a.add_suffix("_qc_adjusted")).reset_index()
    t["sign_changed"] = np.sign(t.within_z_original) != np.sign(t.within_z_qc_adjusted)
    so, sa = t.within_fdr_original < fdr, t.within_fdr_qc_adjusted < fdr
    t["significance_change"] = np.select([so & ~sa, ~so & sa], ["lost", "gained"], "unchanged")
    return t


def compare_terms(orig: pd.DataFrame, adj: pd.DataFrame, terms: list, fdr: float = 0.05) -> pd.DataFrame:
    """hallmark 条目调整前后的 NES 与 FDR。"""
    def tidy(e):
        e = e.assign(fdr=pd.to_numeric(e["FDR q-val"], errors="coerce"), NES=pd.to_numeric(e.NES, errors="coerce"))
        return e.set_index("Term").reindex(terms)[["NES", "fdr"]]
    t = tidy(orig).add_suffix("_original").join(tidy(adj).add_suffix("_qc_adjusted")).reset_index()
    t = t.rename(columns={"index": "Term"})
    t["sign_changed"] = np.sign(t.NES_original) != np.sign(t.NES_qc_adjusted)
    so, sa = t.fdr_original < fdr, t.fdr_qc_adjusted < fdr
    t["significance_change"] = np.select([so & ~sa, ~so & sa], ["lost", "gained"], "unchanged")
    return t


def bh(p: np.ndarray) -> np.ndarray:
    return multipletests(p, method="fdr_bh")[1]
