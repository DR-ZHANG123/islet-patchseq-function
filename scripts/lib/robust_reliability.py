"""测量可靠性与功效（Stage 04d）。

- 供体均值可靠性：(1) 单因素方差分析 ICC(1)（非平衡 n0 校正）→ 每位供体 Spearman–Brown 可靠性 n·ICC/(1+(n−1)ICC)；
  (2) 供体内随机 split-half：两半细胞的供体均值跨供体相关，Spearman–Brown 校正到全部细胞 2r/(1+r)。
  完美预测真实供体均值时，可观察到的供体层面相关上限约为 sqrt(可靠性)。
- 细胞内一致性：总胞吐 = 早期 + 晚期（原始尺度）。两部分的 Cronbach α = 4·cov(早, 晚)/var(总)（比较组内去均值），
  在误差不相关时是总胞吐可靠性的下界；sqrt(α) 为供体内相关上限的保守估计。
- 功效：在实际的比较组结构与实测终点上模拟与实测值真实相关为 r 的预测，按供体重抽样得到 95% 置信区间，
  功效 = 区间下界 > 0 的比例；按供体权重向量化计算合并的供体内 Pearson。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def glucose_std(y: np.ndarray, glucose: np.ndarray) -> np.ndarray:
    z = np.full(len(y), np.nan)
    for g in np.unique(glucose):
        m = (glucose == g) & ~np.isnan(y)
        z[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-12)
    return z


def icc1(z: np.ndarray, donor: np.ndarray) -> float:
    df = pd.DataFrame({"z": z, "d": donor}).dropna()
    g = df.groupby("d").z
    n = g.size().values.astype(float)
    k, N = len(n), n.sum()
    ssb = (n * (g.mean().values - df.z.mean()) ** 2).sum()
    ssw = ((df.z - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)))


def sb(r, k=2.0):
    r = np.asarray(r, float)
    return k * r / (1 + (k - 1) * r)


def split_halves(z, donor, min_cells, n_split, rng):
    """每位供体（≥ min_cells 个细胞）n_split 次随机对半：返回 (供体, 半1均值矩阵, 半2均值矩阵, n)。"""
    df = pd.DataFrame({"z": z, "d": donor}).dropna()
    ds, h1, h2, ns = [], [], [], []
    for d, s in df.groupby("d"):
        v = s.z.values
        if len(v) < min_cells:
            continue
        half = len(v) // 2
        perms = np.argsort(rng.random((n_split, len(v))), axis=1)
        a = v[perms[:, :half]].mean(1)
        b = v[perms[:, half:]].mean(1)
        ds.append(d); h1.append(a); h2.append(b); ns.append(len(v))
    return np.array(ds), np.array(h1), np.array(h2), np.array(ns)


def _colcorr(a, b, rank=False):
    """逐列（随机划分）跨供体相关；a, b 形状 (供体, 划分)。"""
    if rank:
        a = np.apply_along_axis(rankdata, 0, a)
        b = np.apply_along_axis(rankdata, 0, b)
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def split_half_reliability(h1, h2, n_boot, rng, rank=False, boot_cols=50):
    """点估计：所有划分的 split-half 相关取平均后做 Spearman–Brown；置信区间按供体重抽样。"""
    point = float(sb(np.mean(_colcorr(h1, h2, rank))))
    D, S = h1.shape
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, D, D)
        cols = rng.integers(0, S, boot_cols)
        bs.append(sb(np.nanmean(_colcorr(h1[np.ix_(idx, cols)], h2[np.ix_(idx, cols)], rank))))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)


def centred(v, group):
    return v - pd.Series(v).groupby(group).transform("mean").values


def alpha_two(e, l, group):
    """两部分 Cronbach α（比较组内去均值）：4·cov(e, l) / var(e + l)。"""
    ec, lc = centred(e, group), centred(l, group)
    t = ec + lc
    return float(4 * np.mean(ec * lc) / np.mean(t ** 2))


def donor_sums(yc, pc, donor):
    df = pd.DataFrame({"yp": yc * pc, "yy": yc ** 2, "pp": pc ** 2, "d": donor}).groupby("d").sum()
    return df.yp.values, df.yy.values, df.pp.values


def boot_within(yc, pc, donor, n_boot, rng):
    """合并的供体内 Pearson 的按供体重抽样分布（多项权重向量化）。"""
    syp, syy, spp = donor_sums(yc, pc, donor)
    D = len(syp)
    w = rng.multinomial(D, np.full(D, 1 / D), size=n_boot).astype(float)
    return (w @ syp) / np.sqrt((w @ syy) * (w @ spp))


def power_curve(y, group, donor, r_grid, sims, n_boot, rng):
    """y 为实测终点（已去缺失）。返回每个真实 r 的功效与模拟得到的平均观测 r。"""
    yc = centred(y, group)
    keep = pd.Series(group).map(pd.Series(group).value_counts()).values >= 2
    yc, grp, dn = yc[keep], np.asarray(group)[keep], np.asarray(donor)[keep]
    ys = yc / yc.std()
    rows = []
    for r in r_grid:
        hit, rs = 0, []
        for _ in range(sims):
            p = r * ys + np.sqrt(1 - r ** 2) * rng.standard_normal(len(ys))
            pc = centred(p, grp)
            rs.append(np.sum(ys * pc) / np.sqrt(np.sum(ys ** 2) * np.sum(pc ** 2)))
            lo = np.percentile(boot_within(ys, pc, dn, n_boot, rng), 2.5)
            hit += lo > 0
        rows.append({"true_r": r, "power": hit / sims, "mean_observed_r": float(np.mean(rs)),
                     "n_cells": int(len(ys)), "n_groups": int(len(np.unique(grp))),
                     "n_donors": int(len(np.unique(dn)))})
    return pd.DataFrame(rows)


def min_detectable(curve: pd.DataFrame, target: float) -> float:
    """功效曲线上达到目标功效的最小真实 r（线性插值）；达不到返回 NaN。"""
    c = curve.sort_values("true_r")
    above = c[c.power >= target]
    if above.empty:
        return np.nan
    i = above.index[0]
    pos = list(c.index).index(i)
    if pos == 0:
        return float(c.true_r.iloc[0])
    r0, p0 = c.true_r.iloc[pos - 1], c.power.iloc[pos - 1]
    r1, p1 = c.true_r.iloc[pos], c.power.iloc[pos]
    return float(r0 + (target - p0) * (r1 - r0) / (p1 - p0 + 1e-12))
