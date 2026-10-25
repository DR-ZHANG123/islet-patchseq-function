"""Stage 09e 用函数：以细胞自身转录组为条件后的 T2D 效应。

总效应：供体等权线性模型（截距 + T2D + 糖尿病前期 + 葡萄糖 + 来源中心）的 T2D 系数（同 Stage 09c 的 β_y）。
直接效应：同一模型再加入 K 个表达主成分（高变基因、标准化 log 表达，ND/糖尿病前期/T2D 全部细胞上无监督计算，
不使用结局与糖尿病状态）；转录组可解释部分 = 总效应 − 直接效应，可解释比例 = 可解释部分 / 总效应。
交叉拟合（敏感性）：以供体为单位分折，岭回归在训练供体上由全部高变基因与协变量预测结局与 T2D 指示，
留出供体的两个残差做供体等权回归（Robinson 残差回归）；残差固定后按供体重抽样。
半合成检验：去掉实测 T2D 效应后，令其中比例 π 改由留出供体推导的 T2D 转录特征得分携带（总效应不变），
再用同一方法估计可解释比例，检验该方法能否识别由转录状态携带的缺陷。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.utils.extmath import randomized_svd

from t2d_effects import donor_weights
from t2d_models import design

EXO = ("exocytosis", "early_exocytosis", "late_exocytosis")


# ---------- 转录组表示 ----------

def hvg_matrix(xlog: np.ndarray, n_hvg: int) -> np.ndarray:
    """按方差取高变基因，逐基因标准化（均值 0、SD 1）。"""
    v = xlog.var(0)
    hv = np.argsort(v)[::-1][:n_hvg]
    z = xlog[:, hv] - xlog[:, hv].mean(0)
    sd = z.std(0)
    sd[sd == 0] = 1.0
    return z / sd


def pcs(z: np.ndarray, n_pc: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u, s, _ = randomized_svd(z, n_pc, random_state=seed)
    return u * s, s ** 2 / (z ** 2).sum()


# ---------- 数据部分 ----------

def parts(df: pd.DataFrame, endpoint: str, min_group: int = 2) -> dict[str, pd.DataFrame]:
    """全部细胞；胞吐类终点另取正值反应（对数尺度，系数取指数即几何均值比）。"""
    out = {"all cells": df}
    if endpoint in EXO:
        pos = df[df.y > 0].copy()
        pos["y"] = np.log(np.expm1(pos.y.values))
        out["positive responses"] = pos[pos.groupby("g").y.transform("size") >= min_group].reset_index(drop=True)
    return out


def _wls(X: np.ndarray, w: np.ndarray, v: np.ndarray, j: int) -> float:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return float(beta[j])


def pc_effects(df: pd.DataFrame, pc: np.ndarray, ks: list[int], y: np.ndarray | None = None) -> dict:
    """总效应与加入前 K 个主成分后的直接效应；y 缺省为 df.y。"""
    y = df.y.values if y is None else y
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    tot = _wls(X, w, y, j)
    out = {"total": tot}
    p = pc[df.cell_row.values]
    for k in ks:
        d = _wls(np.column_stack([X, p[:, :k]]), w, y, j)
        out[f"direct_pc{k}"] = d
        out[f"indirect_pc{k}"] = tot - d
        out[f"prop_pc{k}"] = (tot - d) / tot if tot != 0 else np.nan
    return out


# ---------- 交叉拟合（岭回归，全部高变基因） ----------

def _ridge_resid(F_tr, v_tr, F_te, v_te, alphas):
    m = RidgeCV(alphas=alphas).fit(F_tr, v_tr)
    return v_te - m.predict(F_te)


def crossfit(df: pd.DataFrame, z: np.ndarray, folds: dict, alphas: np.ndarray,
             y: np.ndarray | None = None, rt: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """按供体分折的结局残差与 T2D 指示残差（协变量与高变基因作为岭回归特征）。rt 给定时只重算结局残差。"""
    y = df.y.values if y is None else y
    X, names = design(df)
    C = X[:, [i for i, n in enumerate(names) if n not in ("intercept", "t2d")]]
    F = np.column_stack([z[df.cell_row.values], C * 10.0])  # 协变量放大 10 倍，使其几乎不受罚
    f = df.donor.map(folds).values
    ry = np.zeros(len(df))
    rt_new = np.zeros(len(df)) if rt is None else rt
    for k in np.unique(f):
        tr, te = f != k, f == k
        ry[te] = _ridge_resid(F[tr], y[tr], F[te], y[te], alphas)
        if rt is None:
            rt_new[te] = _ridge_resid(F[tr], df.t2d.values[tr], F[te], df.t2d.values[te], alphas)
    return ry, rt_new


def dml_theta(ry: np.ndarray, rt: np.ndarray, donor: np.ndarray) -> float:
    w = donor_weights(donor)
    return float((w * rt * ry).sum() / (w * rt ** 2).sum())


# ---------- 半合成 ----------

def synthetic_y(df: pd.DataFrame, pi: float) -> np.ndarray:
    """去掉实测 T2D 效应，再令比例 π 由特征得分 s 携带、1 − π 直接由 T2D 指示携带；总效应不变。"""
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    tot = _wls(X, w, df.y.values, j)
    ds = _wls(X, w, df.s.values, j)
    s = df.s.values - np.average(df.s.values, weights=w)
    return df.y.values - pi * tot * df.t2d.values + pi * tot * s / ds


# ---------- 重抽样 ----------

def boot_df(df: pd.DataFrame, rng) -> pd.DataFrame:
    """状态 × 中心分层的供体重抽样（重复抽到的供体视为不同簇）。"""
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    pos = {d: np.where(df.donor.values == d)[0] for d in dtab.index}
    idx, cid, k = [], [], 0
    for _, g in dtab.groupby(["status", "site"]):
        for d in rng.choice(g.index.values, len(g), replace=True):
            idx.append(pos[d])
            cid.append(np.full(len(pos[d]), f"b{k}"))
            k += 1
    b = df.iloc[np.concatenate(idx)].copy()
    b["orig_row"] = np.concatenate(idx)
    b["donor"] = np.concatenate(cid)
    return b.reset_index(drop=True)


def perm_df(df: pd.DataFrame, rng) -> pd.DataFrame:
    """来源中心内 ND/T2D 供体标签置换。"""
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    lab = dtab.status.copy()
    for _, s in dtab[dtab.status.isin(["ND", "T2D"])].groupby("site"):
        lab.loc[s.index] = rng.permutation(s.status.values)
    p = df.copy()
    p["status"] = df.donor.map(lab).values
    p["t2d"] = (p.status == "T2D").astype(float).values
    return p


def summarise(point: dict, boots: list[dict], perms: list[dict] | None, tested: tuple) -> list[dict]:
    b = pd.DataFrame(boots)
    pm = pd.DataFrame(perms) if perms else None
    rows = []
    for q, v in point.items():
        bb = b[q].replace([np.inf, -np.inf], np.nan).dropna().values if q in b else np.array([])
        row = {"quantity": q, "estimate": v, "lo": np.percentile(bb, 2.5) if len(bb) else np.nan,
               "hi": np.percentile(bb, 97.5) if len(bb) else np.nan, "n_boot": len(bb), "p_perm": np.nan,
               "n_perm": 0}
        if pm is not None and q in tested and q in pm and np.isfinite(v):
            pv = pm[q].dropna().values
            row["p_perm"] = (1 + np.sum(np.abs(pv) >= abs(v) - 1e-12)) / (1 + len(pv))
            row["n_perm"] = len(pv)
        rows.append(row)
    return rows
