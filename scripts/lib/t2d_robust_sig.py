"""Stage 09f 用函数：每次供体重抽样都重新推导留出供体 T2D 特征的中介区间，以及 ND 训练模型供体内精度的诊断。

重抽样：在状态 × 中心层内有放回抽取供体；在抽到的不同供体中重新分折，每折用折外开发供体推导特征
（与 Stage 09d 相同的供体汇总 t 统计量），为折内细胞打分；再在重抽样数据（重复供体视为不同簇）上
计算总效应、加入细胞层面得分及其供体均值后的直接效应。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import donor_folds
from t2d_conditioned import boot_df
from t2d_robust import donor_mean, t2d_coef
from t2d_signature import derive_signature, score_cells


def rederived_scores(td, status, site, dev, donors: np.ndarray, n_genes: int, min_cpm: float, nf: int,
                     seed: int) -> np.ndarray:
    """只用给定供体集合推导留出特征并打分；集合外细胞为 NaN。"""
    ok = np.isin(td.donor, donors) & np.isin(status, ["ND", "preT2D", "T2D"])
    folds = np.full(len(status), -1)
    folds[ok] = donor_folds(td.donor[ok], nf, seed)
    s = np.full(len(status), np.nan)
    for f in range(nf):
        tr = ok & (folds != f) & dev
        up, down, _ = derive_signature(td.counts, td.donor, status, site, np.unique(td.donor[tr]), n_genes, min_cpm)
        ref = np.where(tr & (status == "ND"))[0]
        te = np.where(ok & (folds == f))[0]
        s[te] = score_cells(td.xlog, ref, up, down)[te]
    return s


def _effects(b: pd.DataFrame, s: np.ndarray) -> dict:
    sB = donor_mean(s, b.donor.values)
    tot = t2d_coef(b)
    out = {"total": tot, "score_t2d": t2d_coef(b, y=s)}
    for lab, M in (("signature", s[:, None]), ("signature_between", sB[:, None])):
        d = t2d_coef(b, M)
        out[f"prop_{lab}"] = (tot - d) / tot if tot != 0 else np.nan
    return out


def boot_rederived(frames: dict, td, status, site, dev, n_genes, min_cpm, nf, n_rep: int, seed: int) -> list[dict]:
    """n_rep 次重抽样；frames 为 {(终点, 部分): 细胞表}，各表共用同一次供体抽样。"""
    rng = np.random.default_rng(seed)
    key0 = next(iter(frames))
    rows = []
    for r in range(n_rep):
        b0 = boot_df(frames[key0], rng)
        orig = frames[key0].donor.values[b0.orig_row.values]
        clusters = pd.Series(orig, index=b0.donor.values).groupby(level=0).first()
        s_all = rederived_scores(td, status, site, dev, np.unique(clusters.values), n_genes, min_cpm, nf, seed + r)
        for key, df in frames.items():
            parts = []
            for c, d in clusters.items():
                sub = df[df.donor == d]
                if len(sub):
                    parts.append(sub.assign(donor=c))
            b = pd.concat(parts, ignore_index=True)
            s = s_all[b.cell_row.values]
            keep = np.isfinite(s)
            b, s = b[keep].reset_index(drop=True), s[keep]
            try:
                rows.append({"rep": r, "endpoint": key[0], "part": key[1], **_effects(b, s)})
            except (ValueError, np.linalg.LinAlgError):
                continue
    return rows


def nd_within_diag(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """ND 训练模型在各状态中的供体内 r，供体重抽样区间与组内置换零分布。"""
    rng = np.random.default_rng(seed)
    out = {}
    for st, g in pred.groupby("status"):
        r = float(np.corrcoef(g.yc, g.pc)[0, 1])
        donors = g.donor.unique()
        pos = {d: np.where(g.donor.values == d)[0] for d in donors}
        bs = []
        for _ in range(n_boot):
            idx = np.concatenate([pos[d] for d in rng.choice(donors, len(donors), replace=True)])
            bs.append(np.corrcoef(g.yc.values[idx], g.pc.values[idx])[0, 1])
        null = []
        yc = g.yc.values
        grp = g.g.values
        order = {k: np.where(grp == k)[0] for k in np.unique(grp)}
        for _ in range(n_boot):
            yp = yc.copy()
            for ix in order.values():
                yp[ix] = yc[rng.permutation(ix)]
            null.append(np.corrcoef(yp, g.pc.values)[0, 1])
        null = np.asarray(null)
        out[st] = {"r": r, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                   "p_two_sided_within_perm": float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))),
                   "n_cells": len(g), "n_donors": len(donors)}
    return out
