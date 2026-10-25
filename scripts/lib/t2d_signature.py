"""Stage 09d：β 细胞 T2D 转录特征是否携带胞吐缺陷。

特征：每一折只用折外开发供体（不含 HPAP）的供体汇总表达（各供体全部 β 细胞计数求和 → log2 CPM），
对 ND 与 T2D 做逐基因线性模型（截距 + T2D + 来源中心），取 t 统计量最高与最低的各 n 个基因；
被打分的供体从不参与其所用特征的推导。细胞得分 = 上调基因 z 值均值 − 下调基因 z 值均值，
z 值以训练折 ND 细胞的均值与 SD 标准化。
中介：同 Stage 09c 的供体等权线性模型（截距 + T2D + 糖尿病前期 + 葡萄糖 + 中心），比较加入中介变量
（特征得分、ND 训练的预测值或两者）前后的 T2D 系数；间接效应 = β_total − β_direct。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from data import donor_folds
from t2d_effects import donor_weights
from t2d_models import design, ri_lmm

MEDIATORS = {"signature": ("s",), "predicted": ("p",), "both": ("s", "p")}


# ---------- 特征推导与打分 ----------

def pseudobulk(counts: sp.csr_matrix, donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ud, inv = np.unique(donor, return_inverse=True)
    agg = sp.csr_matrix((np.ones(len(donor)), (inv, np.arange(len(donor)))), shape=(len(ud), len(donor)))
    return ud, np.asarray((agg @ counts).todense(), float)


def de_tstats(pb: np.ndarray, t2d: np.ndarray, site: np.ndarray, min_cpm: float) -> tuple[np.ndarray, np.ndarray]:
    """供体汇总 log2 CPM 上的逐基因 OLS：返回 T2D 系数的 t 统计量与基因保留掩码。"""
    lib = pb.sum(1, keepdims=True)
    cpm = pb / lib * 1e6
    keep = cpm.mean(0) >= min_cpm
    y = np.log2(cpm[:, keep] + 1.0)
    cols = [np.ones(len(t2d)), t2d.astype(float)]
    for lv in sorted(set(site))[1:]:
        v = (site == lv).astype(float)
        if np.ptp(v) > 0:
            cols.append(v)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(t2d) - X.shape[1], 1)
    s2 = (res ** 2).sum(0) / dof
    cov = np.linalg.pinv(X.T @ X)
    t = beta[1] / np.sqrt(np.maximum(s2 * cov[1, 1], 1e-12))
    return t, keep


def derive_signature(counts, donor, status, site, train_donors, n_genes: int, min_cpm: float):
    """训练供体（ND 与 T2D）上的上调与下调基因索引（相对于全部基因）。"""
    m = np.isin(donor, train_donors) & np.isin(status, ["ND", "T2D"])
    ud, pb = pseudobulk(counts[np.where(m)[0]], donor[m])
    dstat = pd.Series(status[m], index=donor[m]).groupby(level=0).first().loc[ud].values
    dsite = pd.Series(site[m], index=donor[m]).groupby(level=0).first().loc[ud].values
    t, keep = de_tstats(pb, dstat == "T2D", dsite, min_cpm)
    idx = np.where(keep)[0]
    order = np.argsort(t)
    return idx[order[::-1][:n_genes]], idx[order[:n_genes]], pd.Series(t, index=idx)


def score_cells(xlog: np.ndarray, ref: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    genes = np.r_[up, down]
    mu = xlog[np.ix_(ref, genes)].mean(0)
    sd = xlog[np.ix_(ref, genes)].std(0)
    sd[sd == 0] = 1.0
    z = (xlog[:, genes] - mu) / sd
    return z[:, :len(up)].mean(1) - z[:, len(up):].mean(1)


def fold_scores(td, status, site, dev, n_genes, min_cpm, nf, seed, keep_sets: bool = False):
    """留出供体打分：每折用折外开发供体推导特征，给折内细胞打分。返回得分（及各折基因集）。"""
    ok = np.isin(status, ["ND", "preT2D", "T2D"])
    folds = np.full(len(status), -1)
    folds[ok] = donor_folds(td.donor[ok], nf, seed)
    s = np.full(len(status), np.nan)
    sets = []
    for f in range(nf):
        tr_cells = ok & (folds != f) & dev
        up, down, t = derive_signature(td.counts, td.donor, status, site, np.unique(td.donor[tr_cells]),
                                       n_genes, min_cpm)
        ref = np.where(tr_cells & (status == "ND"))[0]
        te = np.where(ok & (folds == f))[0]
        s[te] = score_cells(td.xlog, ref, up, down)[te]
        if keep_sets:
            sets.append((f, up, down))
    return s, sets


# ---------- 中介与关联统计 ----------

def _wls_coef(X, w, v, j):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return float(beta[j])


def mediation(df: pd.DataFrame, lmm: bool = False) -> dict:
    """β_total 与加入各中介变量后的 β_direct（供体等权 WLS；lmm=True 时另用随机截距模型）。"""
    w = donor_weights(df.donor.values)
    X, names = design(df)
    j = names.index("t2d")
    out = {"total": _wls_coef(X, w, df.y.values, j), "score_t2d": _wls_coef(X, w, df.s.values, j)}
    for k, cov in MEDIATORS.items():
        Xm, nm = design(df, cov)
        out[f"direct_{k}"] = _wls_coef(Xm, w, df.y.values, nm.index("t2d"))
        out[f"indirect_{k}"] = out["total"] - out[f"direct_{k}"]
        out[f"prop_{k}"] = out[f"indirect_{k}"] / out["total"] if out["total"] != 0 else np.nan
    if lmm:
        tot = float(ri_lmm(df.y.values, X, df.donor.values)["beta"][j])
        out["lmm_total"] = tot
        for k, cov in MEDIATORS.items():
            Xm, nm = design(df, cov)
            d = float(ri_lmm(df.y.values, Xm, df.donor.values)["beta"][nm.index("t2d")])
            out[f"lmm_indirect_{k}"] = tot - d
            out[f"lmm_prop_{k}"] = (tot - d) / tot if tot != 0 else np.nan
    return out


def association(df: pd.DataFrame) -> dict:
    """得分与终点：供体内（比较组内去均值）Pearson r（全部供体、仅 ND）与供体均值 Spearman ρ。"""
    d = df.assign(yc=df.y - df.groupby("g").y.transform("mean"), sc=df.s - df.groupby("g").s.transform("mean"))
    out = {}
    for lab, sub in (("all", d), ("ND", d[d.status == "ND"])):
        ok = len(sub) > 3 and sub.yc.std() > 0 and sub.sc.std() > 0
        out[f"within_r_{lab}"] = float(np.corrcoef(sub.yc, sub.sc)[0, 1]) if ok else np.nan
        m = sub.groupby("donor")[["y", "s"]].mean()
        out[f"donor_rho_{lab}"] = float(spearmanr(m.y, m.s)[0]) if len(m) >= 4 else np.nan
    return out


def safe(fn, df, **kw):
    try:
        return fn(df, **kw)
    except (ValueError, np.linalg.LinAlgError):
        return {}
