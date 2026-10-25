"""Stage 09g 用函数：ND 训练模型供体内负相关的来源诊断、对数尺度 Ca2+ 进入的测量误差校正、胰岛胰岛素含量。

ND 训练诊断：与 Stage 09c 相同的对称留出供体分折（ND/糖尿病前期/T2D 供体同分折，每折只用折外 ND 供体训练），
以岭回归为学习器分三种训练目标：(1) 细胞值（混合供体间与供体内差异），(2) 比较组（供体 × 葡萄糖）内去均值的
细胞值（只学供体内关联），(3) 比较组均值（只学供体间关联）。在留出供体中计算供体内相关。若 (3) 在供体内给出
负相关而 (2) 不然，则负相关来自供体间与供体内方向相反的表达–胞吐关联。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from data import donor_folds
from features import fit_scaler, glucose_onehot, group_center


def _inner_alpha(Z, y, donors, alphas, k, seed) -> float:
    """训练供体内按供体分组的内层交叉验证选择岭参数（最小均方误差）。"""
    f = donor_folds(donors, k, seed)
    err = np.zeros(len(alphas))
    for j in range(k):
        tr, te = f != j, f == j
        if te.sum() == 0 or tr.sum() == 0:
            continue
        for i, a in enumerate(alphas):
            m = Ridge(alpha=a).fit(Z[tr], y[tr])
            err[i] += np.sum((y[te] - m.predict(Z[te])) ** 2)
    return float(alphas[int(np.argmin(err))])


def nd_trained_variants(td, cfg: dict, endpoint: str, seed: int) -> pd.DataFrame:
    """返回留出供体细胞的结局与三种训练目标的预测（均为比较组内去均值后的值）。"""
    k = td.endpoints.index(endpoint)
    y = td.y[:, k]
    st = td.cells.status.values
    ok = np.isin(st, ["ND", "preT2D", "T2D"]) & ~np.isnan(y)
    nf, alphas = cfg["cv"]["outer_folds"], np.asarray(cfg["model"]["alphas"], float)
    folds = np.full(len(y), -1)
    folds[ok] = donor_folds(td.donor[ok], nf, seed)
    g1 = glucose_onehot(td.glucose)
    pred = {v: np.full(len(y), np.nan) for v in ("pooled", "within", "between")}
    for f in range(nf):
        tr = np.where(ok & (folds != f) & (st == "ND"))[0]
        te = np.where(ok & (folds == f))[0]
        sc = fit_scaler(td.xlog[tr], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
        Z = np.c_[sc.transform(td.xlog), g1]
        grp, don = td.group[tr], td.donor[tr]
        yc, Zc = group_center(y[tr], grp), group_center(Z[tr], grp)
        gm = pd.DataFrame(Z[tr]).groupby(grp).mean()
        ym = pd.Series(y[tr]).groupby(grp).mean().loc[gm.index].values
        gdon = pd.Series(don).groupby(grp).first().loc[gm.index].values
        for v, (A, b, dd) in {"pooled": (Z[tr], y[tr], don), "within": (Zc, yc, don),
                              "between": (gm.values, ym, gdon)}.items():
            a = _inner_alpha(A, b, dd, alphas, cfg["cv"]["inner_folds"], seed)
            pred[v][te] = Ridge(alpha=a).fit(A, b).predict(Z[te])
    df = pd.DataFrame({"y": y, "g": td.group, "status": st, "donor": td.donor} | pred)[ok]
    df = df[df.groupby("g").y.transform("size") >= 2].copy()
    df["yc"] = group_center(df.y.values, df.g.values)
    for v in ("pooled", "within", "between"):
        df[f"pc_{v}"] = group_center(df[v].values, df.g.values)
    return df.assign(endpoint=endpoint)


def within_r_boot(df: pd.DataFrame, col: str, n_boot: int, rng) -> dict:
    r = float(np.corrcoef(df.yc, df[col])[0, 1])
    donors = df.donor.unique()
    pos = {d: np.where(df.donor.values == d)[0] for d in donors}
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[d] for d in rng.choice(donors, len(donors), replace=True)])
        bs.append(np.corrcoef(df.yc.values[idx], df[col].values[idx])[0, 1])
    return {"r": r, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "n_cells": len(df),
            "n_donors": len(donors)}


# ---------- 胰岛胰岛素含量（Alberta Diabetes Institute IsletCore 分离记录） ----------

def insulin_content(iso: pd.DataFrame, donors: pd.DataFrame, n_boot: int, n_perm: int, rng) -> list[dict]:
    """每 IEQ 胰岛素含量与胰岛素/DNA 比：T2D 与 ND 供体几何均值比（状态内重抽样区间、状态置换 P）。"""
    d = iso.merge(donors[["donor", "status"]], left_on="record_id", right_on="donor")
    d = d[d.status.isin(["ND", "T2D"])]
    rows = []
    for col in ("insulinperieq", "insulindnaratio", "insulincontent"):
        v = d[[col, "status"]].dropna()
        v = v[v[col] > 0]
        lv, t = np.log(v[col].values), (v.status == "T2D").values
        diff = lv[t].mean() - lv[~t].mean()
        bs = [rng.choice(lv[t], t.sum()).mean() - rng.choice(lv[~t], (~t).sum()).mean() for _ in range(n_boot)]
        perm = []
        for _ in range(n_perm):
            p = rng.permutation(t)
            perm.append(lv[p].mean() - lv[~p].mean())
        rows.append({"measure": col, "ratio": float(np.exp(diff)), "lo": float(np.exp(np.percentile(bs, 2.5))),
                     "hi": float(np.exp(np.percentile(bs, 97.5))),
                     "p_perm": float((1 + np.sum(np.abs(perm) >= abs(diff))) / (1 + n_perm)),
                     "n_nd": int((~t).sum()), "n_t2d": int(t.sum())})
    return rows
