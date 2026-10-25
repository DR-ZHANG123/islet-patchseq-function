"""三个层面的预测评价：细胞层面、供体均值层面、供体内层面。

- 细胞层面：所有测试细胞上实测值与预测值的 Spearman 相关、R²、MAE。
- 供体均值层面：实测值与预测值各自在葡萄糖浓度内标准化后按供体取均值（≥ min_group 个细胞），
  计算供体均值之间的 Spearman 相关（每位供体一个值）。未提供供体与葡萄糖时退化为按比较组计算。
- 供体内层面：比较组内去均值后的实测值与预测值的 Pearson 相关（合并所有组，按细胞加权），
  以及逐组 Spearman 相关的均值（组内 ≥ min_within 个细胞）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _safe(fn, a, b) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:   # 数值噪声级的方差视为常数
        return np.nan
    return float(fn(a, b)[0]) if fn is pearsonr else float(fn(a, b).statistic)


def donor_means(y, p, donor, glucose, min_group: int = 3) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p, "d": donor, "gl": glucose})
    for c in ("y", "p"):
        df[c] = df.groupby("gl")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12) if len(v) > 2 else v * np.nan)
    gm = df.dropna().groupby("d").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return gm[gm.n >= min_group]


def three_level(y: np.ndarray, p: np.ndarray, group: np.ndarray, min_group: int = 3,
                min_within: int = 5, donor=None, glucose=None) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    df = pd.DataFrame({"y": y, "p": p, "g": group})
    out = {"n_cells": len(y), "cell_spearman": _safe(spearmanr, y, p),
           "cell_r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "cell_mae": float(np.abs(y - p).mean())}
    if donor is not None and glucose is not None:
        gm = donor_means(y, p, donor, glucose, min_group)
    else:
        gm = df.groupby("g").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
        gm = gm[gm.n >= min_group]
    out["donor_spearman"] = _safe(spearmanr, gm.y.values, gm.p.values)
    out["n_groups"] = int(len(gm))
    size = df.groupby("g").y.transform("size")
    keep = size >= 2
    yc = (df.y - df.groupby("g").y.transform("mean"))[keep]
    pc = (df.p - df.groupby("g").p.transform("mean"))[keep]
    out["within_pearson"] = _safe(pearsonr, yc.values, pc.values)
    per = [spearmanr(s.y, s.p).statistic for _, s in df.groupby("g")
           if len(s) >= min_within and s.p.std() > 0 and s.y.std() > 0]
    out["within_spearman_mean"] = float(np.nanmean(per)) if per else np.nan
    out["n_within_groups"] = len(per)
    return out


def within_group_scores(y, p, group, min_within: int = 5) -> pd.DataFrame:
    """逐比较组的供体内 Spearman，用于配对比较与分层（如按糖代谢分组）。"""
    df = pd.DataFrame({"y": y, "p": p, "g": group})
    rows = []
    for g, s in df.groupby("g"):
        if len(s) >= min_within and s.p.std() > 0 and s.y.std() > 0:
            rows.append({"group": g, "n": len(s), "spearman": spearmanr(s.y, s.p).statistic})
    return pd.DataFrame(rows)
