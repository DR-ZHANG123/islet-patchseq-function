"""Stage 09：转录组–功能对应关系在 T2D 中是否保留。

对称设计：该细胞类型的全部供体（ND、糖尿病前期、T2D）分到同一组留出供体折；每一折只用折外的 ND 供体训练，
预测折内所有供体。因此 T2D 与糖尿病前期供体从不进入训练，且三组的预测都来自同样规模的训练集。
精度：比较组（供体 × 葡萄糖）内去均值后的 Pearson 相关（供体内精度）。
差值：T2D − ND（以及糖尿病前期 − ND），在各状态内按供体重抽样；另给出按来源中心分层的差值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from context_model import ContextModel
from data import donor_folds
from features import context_mean, fit_scaler, glucose_onehot

MAIN_SITES = ["ADI IsletCore", "HPAP"]


def predict_symmetric(td, cfg, k: int, seed: int) -> pd.DataFrame:
    y = td.y[:, k]
    st = td.cells.status.values
    ok = np.isin(st, ["ND", "preT2D", "T2D"]) & ~np.isnan(y)
    pred = np.full(len(y), np.nan)
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    g1 = glucose_onehot(td.glucose)
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    nf = cfg["cv"]["outer_folds"]
    folds = np.full(len(y), -1)
    folds[ok] = donor_folds(td.donor[ok], nf, seed)
    for f in range(nf):
        tr = np.where(ok & (folds != f) & (st == "ND"))[0]
        te = np.where(ok & (folds == f))[0]
        sc = fit_scaler(td.xlog[tr], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
        z = sc.transform(td.xlog)
        mu = context_mean(z, td.donor)
        m = ContextModel(mcfg, True, cfg["model"]["n_boost_features"], seed).fit(z, mu, g1, y, td.donor, td.group,
                                                                                tr, cells_of)
        pred[te] = m.predict(z, mu, g1, td.donor, te, cells_of)["context_model"]
    site = td.cells.site.where(td.cells.site.isin(MAIN_SITES), "Other").values
    df = pd.DataFrame({"y": y, "p": pred, "g": td.group, "status": st, "donor": td.donor, "site": site})[ok]
    df = df[df.groupby("g").y.transform("size") >= 2].copy()
    df["yc"] = df.y - df.groupby("g").y.transform("mean")
    df["pc"] = df.p - df.groupby("g").p.transform("mean")
    return df.assign(endpoint=td.endpoints[k])


def _r(yc, pc):
    if len(yc) < 3 or np.std(yc) == 0 or np.std(pc) == 0:
        return np.nan
    return float(np.corrcoef(yc, pc)[0, 1])


def _stats(df: pd.DataFrame) -> dict:
    out = {f"r_{s}": _r(g.yc.values, g.pc.values) for s, g in df.groupby("status")}
    for s in ("T2D", "preT2D"):
        out[f"diff_{s}"] = out.get(f"r_{s}", np.nan) - out.get("r_ND", np.nan)
    num, den = 0.0, 0.0
    for site in MAIN_SITES:
        sub = df[df.site == site]
        nd, t2 = sub[sub.status == "ND"], sub[sub.status == "T2D"]
        if nd.donor.nunique() >= 3 and t2.donor.nunique() >= 3:
            w = t2.donor.nunique()
            num += w * (_r(t2.yc.values, t2.pc.values) - _r(nd.yc.values, nd.pc.values))
            den += w
    out["diff_T2D_centre_stratified"] = num / den if den else np.nan
    return out


def coupling_inference(df: pd.DataFrame, n_boot: int, margin: float, seed: int) -> pd.DataFrame:
    """点估计、95% 区间、90% 区间（等效性：以 margin 为界的双单侧检验，基于重抽样分布）。"""
    rng = np.random.default_rng(seed)
    pos = {d: np.where(df.donor.values == d)[0] for d in df.donor.unique()}
    strata = [g.donor.unique() for _, g in df.groupby(["status", "site"])]
    point = _stats(df)
    boots = []
    for _ in range(n_boot):
        idx, cid = [], []
        for k, dd in enumerate(strata):
            for j, d in enumerate(rng.choice(dd, len(dd), replace=True)):
                idx.append(pos[d])
                cid.append(np.full(len(pos[d]), f"{k}_{j}"))
        b = df.iloc[np.concatenate(idx)].assign(donor=np.concatenate(cid))
        boots.append(_stats(b))
    boots = pd.DataFrame(boots)
    rows = []
    for key, v in point.items():
        b = boots[key].dropna().values
        row = {"quantity": key, "estimate": v, "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
               "lo90": np.percentile(b, 5), "hi90": np.percentile(b, 95), "n_boot": len(b)}
        if key.startswith("diff"):
            row["p_two_sided"] = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / (1 + len(b)))
            row["p_equivalence"] = max((b <= -margin).mean(), (b >= margin).mean())
            row["margin"] = margin
        rows.append(row)
    st = df.groupby("status").agg(n_cells=("y", "size"), n_donors=("donor", "nunique"))
    out = pd.DataFrame(rows)
    out["n_cells"] = out.quantity.map(lambda q: st.n_cells.get(q[2:], np.nan) if q.startswith("r_") else np.nan)
    out["n_donors"] = out.quantity.map(lambda q: st.n_donors.get(q[2:], np.nan) if q.startswith("r_") else np.nan)
    return out
