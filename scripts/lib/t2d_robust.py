"""Stage 09f 用函数：T2D 胞吐差异以转录组为条件的分解、阳性对照模拟、安慰剂、Ca2+ 测量误差校正。

记号同 Stage 09e：总效应 = 供体等权线性模型（截距 + T2D + 糖尿病前期 + 葡萄糖 + 来源中心）的 T2D 系数；
直接效应 = 再加入条件变量后的 T2D 系数；携带比例 = (总 − 直接) / 总。
分解：把每个细胞的主成分拆成供体均值（供体间部分）与供体内偏差（供体内部分）。供体内部分与任何供体层面
变量正交，因此只能通过葡萄糖分组影响 T2D 系数；转录组若在供体层面携带缺陷，应由供体均值主成分吸收。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from t2d_conditioned import _wls, boot_df
from t2d_effects import donor_weights
from t2d_models import design


def donor_mean(a: np.ndarray, donor: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return pd.Series(a).groupby(donor).transform("mean").values
    return pd.DataFrame(a).groupby(donor).transform("mean").values


def t2d_coef(df: pd.DataFrame, extra: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
    y = df.y.values if y is None else y
    X, names = design(df)
    if extra is not None:
        X = np.column_stack([X, extra])
    return _wls(X, donor_weights(df.donor.values), y, names.index("t2d"))


def conditioning_sets(df: pd.DataFrame, pc: np.ndarray, kb: list[int], kw: int) -> dict[str, np.ndarray]:
    """条件变量集合：全部主成分、供体内部分、供体间部分（前 k 个）、两者同时（Mundlak）、特征得分及其供体均值。"""
    P = pc[df.cell_row.values]
    PB = donor_mean(P, df.donor.values)
    PW = P - PB
    sets = {f"full{kw}": P[:, :kw], f"within{kw}": PW[:, :kw]}
    for k in kb:
        sets[f"between{k}"] = PB[:, :k]
        sets[f"mundlak{k}"] = np.column_stack([PB[:, :k], PW[:, :kw]])
    sB = donor_mean(df.s.values, df.donor.values)
    sets["signature"] = df.s.values[:, None]
    sets["signature_between"] = sB[:, None]
    sets["signature_within"] = (df.s.values - sB)[:, None]
    if df.s_ext.notna().all():
        sets["external_signature"] = df.s_ext.values[:, None]
        sets["external_signature_between"] = donor_mean(df.s_ext.values, df.donor.values)[:, None]
    return sets


def shares(df: pd.DataFrame, pc: np.ndarray, kb: list[int], kw: int, y: np.ndarray | None = None,
           only: list[str] | None = None) -> dict:
    tot = t2d_coef(df, None, y)
    out = {"total": tot}
    for name, M in conditioning_sets(df, pc, kb, kw).items():
        if only is not None and name not in only:
            continue
        d = t2d_coef(df, M, y)
        out[f"direct_{name}"] = d
        out[f"prop_{name}"] = (tot - d) / tot if tot != 0 else np.nan
    return out


def summarise_boot(point: dict, boots: list[dict]) -> list[dict]:
    b = pd.DataFrame(boots).replace([np.inf, -np.inf], np.nan)
    rows = []
    for q, v in point.items():
        bb = b[q].dropna().values if q in b else np.array([])
        rows.append({"quantity": q, "estimate": v, "lo": np.percentile(bb, 2.5) if len(bb) else np.nan,
                     "hi": np.percentile(bb, 97.5) if len(bb) else np.nan, "n_boot": len(bb)})
    return rows


def boot_shares(df, pc, kb, kw, n_boot, rng) -> list[dict]:
    point = shares(df, pc, kb, kw)
    boots = [shares(boot_df(df, rng), pc, kb, kw) for _ in range(n_boot)]
    return summarise_boot(point, boots)


# ---------- 阳性对照模拟 ----------

def carriers(df: pd.DataFrame, rhos: list[float], seed: int) -> dict[str, np.ndarray]:
    """缺陷的携带者：细胞层面特征、其供体均值、供体均值 + 部分供体内特征 + 非转录噪声、20 基因小特征。"""
    s = df.s.values
    sB = donor_mean(s, df.donor.values)
    sW = s - sB
    eps = np.random.default_rng(seed).standard_normal(len(s)) * sW.std()
    out = {"signature_cell": s, "signature_donor": sB, "small_geneset_cell": df.s_small.values}
    for r in rhos:
        out[f"signature_noisy_rho{r:g}"] = sB + r * sW + np.sqrt(max(1 - r ** 2, 0.0)) * eps
    return out


def synth_y(df: pd.DataFrame, c: np.ndarray, pi: float) -> np.ndarray:
    """去掉实测 T2D 效应，令比例 π 改由携带者 c 承担（总效应不变；c 的 T2D 差须不为零）。"""
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    tot = _wls(X, w, df.y.values, j)
    dc = _wls(X, w, c, j)
    cc = c - np.average(c, weights=w)
    return df.y.values - pi * tot * df.t2d.values + pi * tot * cc / dc


def within_r(df: pd.DataFrame, y: np.ndarray, x: np.ndarray) -> float:
    g = df.g.values
    yc = y - pd.Series(y).groupby(g).transform("mean").values
    xc = x - pd.Series(x).groupby(g).transform("mean").values
    return float(np.corrcoef(yc, xc)[0, 1])


def simulate(df, pc, carrier: np.ndarray, pis, estimators, kb, kw, n_boot, rng) -> list[dict]:
    """各 π 下的携带比例估计（点估计与供体重抽样区间；重抽样中携带者按原细胞取值，合成结局重算）。"""
    rows = []
    for pi in pis:
        ys = synth_y(df, carrier, pi)
        pt = shares(df, pc, kb, kw, y=ys, only=estimators)
        boots = []
        for _ in range(n_boot):
            b = boot_df(df, rng)
            boots.append(shares(b, pc, kb, kw, y=synth_y(b, carrier[b.orig_row.values], pi), only=estimators))
        b = pd.DataFrame(boots).replace([np.inf, -np.inf], np.nan)
        for e in estimators:
            q = f"prop_{e}"
            bb = b[q].dropna().values
            rows.append({"pi": pi, "estimator": e, "estimate": pt[q], "lo": np.percentile(bb, 2.5),
                         "hi": np.percentile(bb, 97.5), "share_boot_above_0": float(np.mean(bb > 0)),
                         "within_r_y_signature": within_r(df, ys, df.s.values),
                         "within_r_y_carrier": within_r(df, ys, carrier)})
    return rows


# ---------- 安慰剂：供体均值主成分在供体间置换 ----------

def placebo(df: pd.DataFrame, pc: np.ndarray, kw: int, n: int, rng) -> dict:
    P = pc[df.cell_row.values][:, :kw]
    donors = df.donor.values
    PB = donor_mean(P, donors)
    PW = P - PB
    ud = np.unique(donors)
    first = {d: np.where(donors == d)[0][0] for d in ud}
    tot = t2d_coef(df)
    obs = (tot - t2d_coef(df, P)) / tot
    vals = []
    for _ in range(n):
        perm = dict(zip(ud, rng.permutation(ud)))
        PBp = np.vstack([PB[first[perm[d]]] for d in donors])
        vals.append((tot - t2d_coef(df, PBp + PW)) / tot)
    vals = np.asarray(vals)
    return {"observed_share": obs, "placebo_mean": float(vals.mean()), "placebo_lo": np.percentile(vals, 2.5),
            "placebo_hi": np.percentile(vals, 97.5), "p_placebo_ge_observed": float((1 + np.sum(vals >= obs)) / (1 + n)),
            "n_placebo": n}


# ---------- Ca2+ 进入的测量误差：回归校准 ----------

def _group_center(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    return v - pd.Series(v).groupby(g).transform("mean").values


def ca_models(df: pd.DataFrame, lams: list[float], col: str = "ca") -> dict:
    """同一组细胞上：Ca2+ 进入的 T2D 差、未调整与调整 Ca2+ 进入后的胞吐 T2D 系数，以及按供体内信度 λ 的回归校准。"""
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    x = df[col].values
    out = {"total": _wls(X, w, df.y.values, j), "ca_t2d": _wls(X, w, x, j)}
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], x * sw, rcond=None)
    xhat = X @ beta
    e = x - xhat
    var_within = np.var(_group_center(x, df.g.values))
    for lam in lams:
        lam_r = max(1.0 - (1.0 - lam) * var_within / np.var(e), 0.05)
        xs = xhat + lam_r * e
        out[f"direct_ca_lam{lam:g}"] = _wls(np.column_stack([X, xs]), w, df.y.values, j)
    xb = donor_mean(x, df.donor.values)
    out["direct_ca_between"] = _wls(np.column_stack([X, xb]), w, df.y.values, j)
    out["direct_ca_between_within"] = _wls(np.column_stack([X, xb, x - xb]), w, df.y.values, j)
    return out


def ca_reliability(df: pd.DataFrame, col: str = "ca", col_early: str = "ca_early") -> dict:
    """供体内（比较组内去均值）Ca2+ 进入与早期 Ca2+ 电流的相关：两次同源测量一致性的参考值。"""
    a = _group_center(df[col].values, df.g.values)
    b = _group_center(df[col_early].values, df.g.values)
    ok = np.isfinite(a) & np.isfinite(b)
    return {"within_r_ca_entry_early_ca": float(np.corrcoef(a[ok], b[ok])[0, 1]), "n_cells": int(ok.sum())}


def boot_generic(fn, df: pd.DataFrame, n_boot: int, rng, **kw) -> list[dict]:
    point = fn(df, **kw)
    boots = [fn(boot_df(df, rng), **kw) for _ in range(n_boot)]
    return summarise_boot(point, boots)


# ---------- 供体内结局置换零分布与外部特征的 T2D 差 ----------

def within_perm_null(df: pd.DataFrame, pc: np.ndarray, ks: list[int], n: int, rng) -> list[dict]:
    """结局在比较组（供体 × 葡萄糖）内置换、主成分不变：携带比例的零分布（保留供体层面结构，去掉细胞层面对应）。"""
    tot = t2d_coef(df)
    P = pc[df.cell_row.values]
    idx = [np.where(df.g.values == k)[0] for k in np.unique(df.g.values)]
    obs = {k: (tot - t2d_coef(df, P[:, :k])) / tot for k in ks}
    null = {k: [] for k in ks}
    for _ in range(n):
        y = df.y.values.copy()
        for ix in idx:
            y[ix] = df.y.values[rng.permutation(ix)]
        for k in ks:
            null[k].append((tot - t2d_coef(df, P[:, :k], y)) / tot)
    rows = []
    for k in ks:
        v = np.asarray(null[k])
        rows.append({"n_pcs": k, "observed_share": obs[k], "null_median": float(np.median(v)),
                     "null_lo": np.percentile(v, 2.5), "null_hi": np.percentile(v, 97.5),
                     "p_null_ge_observed": float((1 + np.sum(v >= obs[k])) / (1 + n)), "n_perm": n})
    return rows


def score_effect(df: pd.DataFrame, col: str) -> dict:
    """得分的 T2D 系数（供体等权）。两种单位：ND 细胞（葡萄糖内去均值）得分 SD（同 Stage 09d），ND 供体均值的 SD。"""
    s = df[col].values
    b = t2d_coef(df, y=s)
    nd = df.assign(v=s)[df.status == "ND"]
    sd_cell = float((nd.v - nd.groupby("glucose").v.transform("mean")).std())
    sd_donor = float(nd.groupby("donor").v.mean().std())
    return {"score_t2d": b, "score_t2d_nd_cell_sd": b / sd_cell, "score_t2d_nd_donor_mean_sd": b / sd_donor}
