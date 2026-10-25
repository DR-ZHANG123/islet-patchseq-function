"""Stage 04j：胞吐的 Ca²⁺ 可解释部分与 Ca²⁺–分泌偶联效率。

分解（log1p 尺度）：E = b·X + r。X 为细胞自身实测的 Ca²⁺ 内流、早/晚期 Ca²⁺ 电流与细胞大小（与终点相同的变换），
b 为比较组（供体 × 葡萄糖）内去均值后的合并最小二乘斜率，只在训练细胞上估计；b·X 为 Ca²⁺ 可解释部分，
r = E − b·X 为偶联效率（给定 Ca²⁺ 内流、电流与大小时多出或少出的胞吐），二者都保留供体层面差异。
比值形式（敏感性）：log(E_raw / Ca²⁺ 内流_raw)，只用两者均 > 0 的细胞，不需拟合；其对照 positive_* 为同一批细胞上的 log E_raw。
所有派生终点与对应原始终点都限制在 X 完整的同一批细胞上。
T2D 比较：偶联效率（斜率由 ND 细胞估计）与比值的随机截距线性混合模型 T2D 系数，推断同 Stage 09。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features import group_center
from t2d_models import design, ri_lmm
from t2d_resample import _bootstrap_frame, _permuted_frame, _positions, summarize

SHORT = {"exocytosis": "total", "early_exocytosis": "early", "late_exocytosis": "late"}


def derived_names(cc: dict) -> list:
    out = []
    for t in cc["targets"]:
        s = SHORT[t]
        out += [f"ca_explained_{s}", f"coupling_{s}", f"ratio_{s}"]
    return out + [f"positive_{SHORT[t]}" for t in cc["targets"]]   # 比值的对照：同一批细胞上的 log 胞吐（不除以 Ca²⁺ 内流）


def augment(td, cc: dict):
    """返回只含 X 完整细胞的终点矩阵（原始终点 + 派生终点占位）与终点名列表；派生列由 fill_derived 按折填充。"""
    cols = [td.endpoints.index(v) for v in cc["covariates"]]
    complete = ~np.isnan(td.y[:, cols]).any(1)
    raw = list(dict.fromkeys([cc["ratio_entry"]] + cc["covariates"] + cc["targets"]))
    y = np.full((len(td.donor), len(raw)), np.nan)
    for j, ep in enumerate(raw):
        y[complete, j] = td.y[complete, td.endpoints.index(ep)]
    names = derived_names(cc)
    y = np.c_[y, np.full((len(td.donor), len(names)), np.nan)]
    return replace(td, y=y, endpoints=raw + names), complete


def slopes(td, cc: dict, idx: np.ndarray, target: str) -> np.ndarray:
    """训练细胞（idx）上 E 对 X 的组内去均值合并斜率。"""
    xi = [td.endpoints.index(v) for v in cc["covariates"]]
    e = td.y[idx, td.endpoints.index(target)]
    x = td.y[idx][:, xi]
    ok = ~np.isnan(e) & ~np.isnan(x).any(1)
    g = td.group[idx][ok]
    b, *_ = np.linalg.lstsq(group_center(x[ok], g), group_center(e[ok], g), rcond=None)
    return b


def ratio(td, cc: dict, target: str, divide: bool = True) -> np.ndarray:
    e = td.cells[target].values.astype(float)
    ca = td.cells[cc["ratio_entry"]].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(e) - (np.log(ca) if divide else 0.0)
    out[~((e > 0) & (ca > 0))] = np.nan
    return out


def fill_derived(td, cc: dict, complete: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """用训练细胞估计 b，填充全部细胞的派生终点（完整病例之外为 NaN）。返回 (终点矩阵, {目标: b})。"""
    y = td.y.copy()
    xi = [td.endpoints.index(v) for v in cc["covariates"]]
    bs = {}
    for t in cc["targets"]:
        s = SHORT[t]
        b = slopes(td, cc, train_idx, t)
        bs[t] = b
        expl = td.y[:, xi] @ b
        e = td.y[:, td.endpoints.index(t)]
        y[:, td.endpoints.index(f"ca_explained_{s}")] = np.where(complete, expl, np.nan)
        y[:, td.endpoints.index(f"coupling_{s}")] = np.where(complete, e - expl, np.nan)
        y[:, td.endpoints.index(f"ratio_{s}")] = np.where(complete, ratio(td, cc, t), np.nan)
        y[:, td.endpoints.index(f"positive_{s}")] = np.where(complete, ratio(td, cc, t, divide=False), np.nan)
    return y, bs


def decomposition_summary(td, cc: dict, complete: np.ndarray, ct: str) -> pd.DataFrame:
    """全部开发细胞上的分解描述：斜率、组内 R²、各部分的供体间方差占比（描述性，不用于建模）。"""
    idx = np.where(complete)[0]
    y, bs = fill_derived(td, cc, complete, idx)
    rows = []
    for t in cc["targets"]:
        s = SHORT[t]
        e = y[idx, td.endpoints.index(t)]
        ok = ~np.isnan(e)
        g = td.group[idx][ok]
        ec = group_center(e[ok], g)
        rc = group_center(y[idx, td.endpoints.index(f"coupling_{s}")][ok], g)
        rec = {"cell_type": ct, "target": t, "n_cells": int(ok.sum()), "n_donors": int(len(np.unique(td.donor[idx][ok]))),
               "within_r2_ca": float(1 - (rc ** 2).sum() / (ec ** 2).sum())}
        for part in (t, f"ca_explained_{s}", f"coupling_{s}"):
            v = y[idx, td.endpoints.index(part)][ok]
            dm = pd.Series(v).groupby(td.donor[idx][ok]).transform("mean").values
            rec[f"between_donor_share_{part.replace('_' + s, '') if part != t else 'measured'}"] = float(
                np.var(dm) / np.var(v))
        rec.update({f"b_{c}": float(v) for c, v in zip(cc["covariates"], bs[t])})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- T2D 比较 ----------
def t2d_frame(cells: pd.DataFrame, ct: str, cc: dict, sites: list, statuses: list) -> pd.DataFrame:
    """全部 Patch-seq 供体（含外部队列）的细胞表；X 与终点同 Stage 01 的变换；偶联效率的斜率只用 ND 细胞估计。"""
    from features import transform_endpoint
    c = cells[cells.qc_pass & (cells.cell_type == ct) & cells.status.isin(statuses)]
    c = c.dropna(subset=cc["covariates"] + cc["targets"]).reset_index(drop=True)
    tf = {"size": "log"}
    df = pd.DataFrame({"donor": c.donor.values, "status": c.status.values, "glucose": c.glucose.values,
                       "site": c.site.where(c.site.isin(sites), "Other").values})
    df["t2d"] = (df.status == "T2D").astype(float)
    df["pre"] = (df.status == "preT2D").astype(float)
    x = np.column_stack([transform_endpoint(c[v].values, tf.get(v, "log1p")) for v in cc["covariates"]])
    grp = (c.donor + "|" + c.glucose.astype(str)).values
    nd = df.status.values == "ND"
    for t in cc["targets"]:
        s = SHORT[t]
        e = transform_endpoint(c[t].values, "log1p")
        b, *_ = np.linalg.lstsq(group_center(x[nd], grp[nd]), group_center(e[nd], grp[nd]), rcond=None)
        df[f"coupling_{s}"] = e - x @ b
        ev, ca = c[t].values.astype(float), c[cc["ratio_entry"]].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"ratio_{s}"] = np.where((ev > 0) & (ca > 0), np.log(ev) - np.log(ca), np.nan)
    df["calcium_entry"] = x[:, cc["covariates"].index(cc["ratio_entry"])]
    return df


def lmm_effect(df: pd.DataFrame, col: str) -> dict:
    d = df[np.isfinite(df[col].values)]
    X, names = design(d)
    return {"t2d_coef": ri_lmm(d[col].values, X, d.donor.values)["beta"][names.index("t2d")]}


def _safe(df, col):
    try:
        return lmm_effect(df, col)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {}


def _chunk(df, col, kind, n, seed):
    rng = np.random.default_rng(seed)
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    pos = _positions(df)
    strata = [g.index.values for _, g in dtab.groupby(["status", "site"])]
    return [_safe(_bootstrap_frame(df, pos, strata, rng) if kind == "boot" else _permuted_frame(df, dtab, rng), col)
            for _ in range(n)]


def t2d_test(df: pd.DataFrame, col: str, n_boot: int, n_perm: int, seed: int, n_jobs: int,
             chunk: int = 50) -> pd.DataFrame:
    """T2D 系数（log 尺度）：状态 × 中心分层供体重抽样置信区间，中心内供体标签置换 P 值。"""
    df = df.reset_index(drop=True)
    point = lmm_effect(df, col)
    tasks = [("boot", min(chunk, n_boot - i)) for i in range(0, n_boot, chunk)] + \
            [("perm", min(chunk, n_perm - i)) for i in range(0, n_perm, chunk)]
    seeds = np.random.SeedSequence(seed).spawn(len(tasks))
    res = Parallel(n_jobs=n_jobs)(delayed(_chunk)(df, col, k, n, int(s.generate_state(1)[0]))
                                  for (k, n), s in zip(tasks, seeds))
    boots = pd.DataFrame([r for (k, _), rr in zip(tasks, res) if k == "boot" for r in rr], columns=list(point))
    perms = pd.DataFrame([r for (k, _), rr in zip(tasks, res) if k == "perm" for r in rr], columns=list(point))
    out = summarize(point, boots, perms)
    d = df[np.isfinite(df[col].values)]
    st = d.groupby("donor").status.first()
    out["scale"] = "log ratio" if col.startswith("ratio") else "log1p difference"   # exp → 比值（偶联效率为 1 + 胞吐 的比值）
    out["ratio_t2d_vs_nd"] = np.exp(out.estimate)
    out["ratio_lo"], out["ratio_hi"] = np.exp(out.lo), np.exp(out.hi)
    return out.assign(measure=col, n_cells=len(d), n_nd_donors=int((st == "ND").sum()),
                      n_t2d_donors=int((st == "T2D").sum()))
