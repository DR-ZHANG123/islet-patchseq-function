"""Stage 09：单个终点的 T2D 效应量（两部分模型、分位数比、供体内离散度、高功能细胞比例）。

输入为细胞表（原始尺度终点 y ≥ 0、供体/簇编号、状态、葡萄糖、来源中心、协变量）；每个函数都可在按供体重抽样或
置换后的数据上重复调用。比值类效应在对数尺度返回，汇总时再取指数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import digamma

from repro import ROOT
from t2d_models import design, ri_lmm, ri_logit, weighted_mean_by, wquantreg

STATUSES = ["ND", "preT2D", "T2D"]
MAIN_SITES = ["ADI IsletCore", "HPAP"]


def load_cells(cell_type: str, endpoint: str) -> pd.DataFrame:
    """原始尺度终点（负值记为 0，与主分析的终点定义一致）及供体协变量。"""
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    donors = pd.read_csv(ROOT / "results/01_dataset/donors.tsv", sep="\t").set_index("donor")
    c = cells[cells.qc_pass & (cells.cell_type == cell_type) & cells.status.isin(STATUSES)]
    c = c.dropna(subset=[endpoint])
    df = pd.DataFrame({"y": c[endpoint].clip(lower=0).values, "donor": c.donor.values, "status": c.status.values,
                       "glucose": c.glucose.values, "site": c.site.where(c.site.isin(MAIN_SITES), "Other").values})
    df = df.join(donors[["age", "sex", "bmi", "clinical"]], on="donor")
    df["sex_male"] = (df.sex == "Male").astype(float).where(df.sex.notna())
    df["t2d"] = (df.status == "T2D").astype(float)
    df["pre"] = (df.status == "preT2D").astype(float)
    # 与既往 SD 尺度结果可比：log1p 后在每个葡萄糖浓度内标准化
    z = np.log1p(df.y) if endpoint != "size" else np.log(df.y)
    df["z"] = z.groupby(df.glucose).transform(lambda v: (v - v.mean()) / v.std())
    return df.reset_index(drop=True)


def add_isolation(df: pd.DataFrame, covariates: list) -> pd.DataFrame:
    """HumanIslets 分离与培养变量（仅 ADI IsletCore 供体有记录）。"""
    raw = ROOT / "data/raw/humanislets"
    iso = pd.read_csv(raw / "isolation.csv").set_index("record_id")
    don = pd.read_csv(raw / "donor.csv").set_index("record_id")
    tab = pd.DataFrame({"coldischemiatime": iso.coldischemiatime,
                        "predistributionculturetime": don.predistributionculturetime,
                        "isolation_year": pd.to_datetime(iso.isolationdate, errors="coerce").dt.year})
    return df.join(tab[covariates], on="donor")


def donor_weights(donor: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(donor, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def _within_sd(pos: pd.DataFrame, min_df: int) -> float:
    """供体内（同葡萄糖条件内）log 正值的离散度：偏差校正的 log 方差对 T2D 回归，返回 log SD 比。"""
    pos = pos.assign(ly=np.log(pos.y.values))
    pos["res"] = pos.ly - pos.groupby(["donor", "glucose"]).ly.transform("mean")
    d = pos.groupby("donor").agg(ss=("res", lambda v: float(np.sum(v ** 2))), n=("res", "size"),
                                 k=("glucose", "nunique"), t2d=("t2d", "first"), pre=("pre", "first"),
                                 site=("site", "first")).reset_index()
    d["df"] = d.n - d.k
    d = d[d.df >= min_df]
    if d.t2d.sum() < 2 or (d.t2d == 0).sum() < 2:
        return np.nan
    half = d.df.values / 2.0
    lv = np.log(d.ss.values / d.df.values) - (digamma(half) - np.log(half))
    X, names = design(d.assign(glucose=0))
    beta, *_ = np.linalg.lstsq(X, lv, rcond=None)
    return 0.5 * float(beta[names.index("t2d")])


def _wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(cw, q)])


def _shares(df: pd.DataFrame, thr_q: float, ratio: float, min_nd: int) -> dict:
    """高于同葡萄糖、同中心 ND 细胞 thr_q 分位数的细胞比例（每位供体等权），以及按比例缩放预测的 T2D 比例。"""
    out = {}
    df = df[df.status.isin(["ND", "T2D"])]
    parts = []
    for (_, _), s in df.groupby(["glucose", "site"]):
        nd = s[s.status == "ND"]
        if len(nd) < min_nd or (s.status == "T2D").sum() == 0:
            continue
        wn = donor_weights(nd.donor.values)
        thr = _wquantile(nd.y.values, wn, thr_q)
        scaled = weighted_mean_by((nd.y.values * ratio > thr).astype(float), wn)
        parts.append(s.assign(high=(s.y > thr).astype(float), scaled=scaled))
    if not parts:
        return {}
    e = pd.concat(parts)
    e["w"] = donor_weights(e.donor.values)
    nd, t2 = e[e.status == "ND"], e[e.status == "T2D"]
    rho = weighted_mean_by((t2.y > 0).values.astype(float), t2.w.values) / \
        weighted_mean_by((nd.y > 0).values.astype(float), nd.w.values)
    obs_t, obs_n = weighted_mean_by(t2.high.values, t2.w.values), weighted_mean_by(nd.high.values, nd.w.values)
    exp_t = rho * weighted_mean_by(t2.scaled.values, t2.w.values)
    tag = f"{int(round(thr_q * 100))}"
    out[f"share{tag}_nd"] = obs_n
    out[f"share{tag}_t2d"] = obs_t
    out[f"share{tag}_expected"] = exp_t
    out[f"share{tag}_diff"] = obs_t - obs_n
    out[f"share{tag}_excess"] = obs_t - exp_t
    return out


def effects(df: pd.DataFrame, spec: dict) -> dict:
    """一个终点的全部 T2D 效应量。df.donor 为簇编号（重抽样时重复供体各自成簇）。"""
    cov = tuple(spec.get("covariates", ()))
    res = {}
    X, names = design(df, cov)
    j = names.index("t2d")
    res["mean_sd"] = ri_lmm(df.z.values, X, df.donor.values)["beta"][j]
    nz = (df.y.values > 0).astype(float)
    if spec["zero_part"] and (1 - nz).sum() >= spec["min_zero_cells"]:
        res["nonzero_logor"] = ri_logit(nz, X, df.donor.values, spec["gh_nodes"], spec["logit_ridge"])["beta"][j]
    pos = df[df.y > 0]
    Xp, npos = design(pos, cov)
    jp = npos.index("t2d")
    res["positive_logratio"] = ri_lmm(np.log(pos.y.values), Xp, pos.donor.values)["beta"][jp]
    res["sd_logratio"] = _within_sd(pos, spec["min_sd_df"])
    weigh = (lambda d: donor_weights(d)) if spec["weighting"] == "donor" else (lambda d: np.ones(len(d)))
    ly_all = np.log(np.maximum(df.y.values, spec["floor"]))
    w_all, w_pos = weigh(df.donor.values), weigh(pos.donor.values)
    for q in spec["qs_all"]:
        res[f"q{int(q * 100)}_all_logratio"] = wquantreg(ly_all, X, w_all, q)[j]
    for q in spec["qs_pos"]:
        res[f"q{int(q * 100)}_pos_logratio"] = wquantreg(np.log(pos.y.values), Xp, w_pos, q)[jp]
    lo, hi = (int(q * 100) for q in spec["tail"])
    for part in ("all", "pos"):
        a, b = res.get(f"q{hi}_{part}_logratio"), res.get(f"q{lo}_{part}_logratio")
        if a is not None and b is not None:
            res[f"tail_{part}_logratio"] = a - b
    for t in spec["thresholds"]:
        res.update(_shares(df, t, float(np.exp(res["positive_logratio"])), spec["min_nd_threshold_cells"]))
    return res
