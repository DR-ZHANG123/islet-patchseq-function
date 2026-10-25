"""Stage 09c：T2D 的功能差异是否体现在转录组预测值中。

每个细胞有实测值 y、转录组预测值 p（只用 ND 供体训练的对称留出供体预测）与残差 r = y − p，均在建模尺度
（log1p 或 log）上。对三者分别拟合同一线性模型：截距 + T2D + 糖尿病前期 + 葡萄糖 + 来源中心，细胞权重 1/n_d
（每位供体等权，相当于供体均值层面的比较）。模型对结果线性，故 β_y = β_p + β_r 严格成立，
"预测值捕获比例" = β_p / β_y。
胞吐类终点另做"仅正值反应"部分（y > 0 的细胞），对应 Stage 09 的正值几何均值框架。
推断：状态 × 中心分层的供体重抽样（区间），中心内 ND/T2D 供体标签置换（P 值），与 Stage 09 相同。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from t2d_effects import donor_weights
from t2d_models import design

EXO = ("exocytosis", "early_exocytosis", "late_exocytosis")


def _coef(X: np.ndarray, w: np.ndarray, v: np.ndarray, j: int) -> float:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return float(beta[j])


def _within_r(sub: pd.DataFrame) -> float:
    if len(sub) < 3 or sub.yc.std() == 0 or sub.pc.std() == 0:
        return np.nan
    return float(np.corrcoef(sub.yc, sub.pc)[0, 1])


def _donor_rho(sub: pd.DataFrame) -> float:
    m = sub.groupby("donor")[["y", "p"]].mean()
    if len(m) < 4:
        return np.nan
    return float(spearmanr(m.y, m.p)[0])


def part_stats(df: pd.DataFrame) -> dict:
    """一个数据部分（全部细胞或仅正值细胞）的全部统计量。df.donor 为簇编号。"""
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    out = {f"beta_{k}": _coef(X, w, df[k].values, j) for k in ("y", "p", "r")}
    out["captured_fraction"] = out["beta_p"] / out["beta_y"] if out["beta_y"] != 0 else np.nan
    for s in ("ND", "T2D"):
        sub = df[df.status == s]
        out[f"within_r_{s}"] = _within_r(sub)
        out[f"donor_rho_{s}"] = _donor_rho(sub)
    out["within_r_diff"] = out["within_r_T2D"] - out["within_r_ND"]
    return out


def parts(df: pd.DataFrame, endpoint: str) -> dict[str, pd.DataFrame]:
    out = {"all cells": df}
    if endpoint in EXO:
        pos = df[df.y > 0].copy()
        pos = pos[pos.groupby("g").y.transform("size") >= 2]
        pos["yc"] = pos.y - pos.groupby("g").y.transform("mean")
        pos["pc"] = pos.p - pos.groupby("g").p.transform("mean")
        out["positive responses"] = pos
    return out


def _boot(df: pd.DataFrame, rng) -> pd.DataFrame:
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    idx, cid, k = [], [], 0
    pos = {d: np.where(df.donor.values == d)[0] for d in dtab.index}
    for _, g in dtab.groupby(["status", "site"]):
        for d in rng.choice(g.index.values, len(g), replace=True):
            idx.append(pos[d])
            cid.append(np.full(len(pos[d]), k))
            k += 1
    b = df.iloc[np.concatenate(idx)].copy()
    b["donor"] = np.concatenate(cid)
    return b


def _perm(df: pd.DataFrame, rng) -> pd.DataFrame:
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    lab = dtab.status.copy()
    for _, s in dtab[dtab.status.isin(["ND", "T2D"])].groupby("site"):
        lab.loc[s.index] = rng.permutation(s.status.values)
    p = df.copy()
    p["status"] = df.donor.map(lab).values
    p["t2d"] = (p.status == "T2D").astype(float).values
    return p


def _safe(df):
    try:
        return part_stats(df)
    except (ValueError, np.linalg.LinAlgError):
        return {}


def infer(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    """点估计、供体重抽样 95% 区间与中心内标签置换 P 值（captured_fraction 与相关只给区间）。"""
    rng = np.random.default_rng(seed)
    point = part_stats(df)
    boots = pd.DataFrame([_safe(_boot(df, rng)) for _ in range(n_boot)], columns=list(point))
    tested = ("beta_y", "beta_p", "beta_r", "within_r_diff")
    perms = pd.DataFrame([{k: v for k, v in _safe(_perm(df, rng)).items() if k in tested}
                          for _ in range(n_perm)], columns=list(tested))
    rows = []
    for k, v in point.items():
        b = boots[k].replace([np.inf, -np.inf], np.nan).dropna().values
        row = {"quantity": k, "estimate": v, "lo": np.percentile(b, 2.5) if len(b) else np.nan,
               "hi": np.percentile(b, 97.5) if len(b) else np.nan, "n_boot": len(b), "p_perm": np.nan}
        if k in tested and np.isfinite(v):
            p = perms[k].dropna().values
            row["p_perm"] = (1 + np.sum(np.abs(p) >= abs(v) - 1e-12)) / (1 + len(p))
        rows.append(row)
    return pd.DataFrame(rows)


def nd_sd(df: pd.DataFrame) -> float:
    """ND 细胞在葡萄糖内去均值后的实测值 SD，用于把系数换算为 SD 单位。"""
    nd = df[df.status == "ND"]
    return float((nd.y - nd.groupby("glucose").y.transform("mean")).std())
