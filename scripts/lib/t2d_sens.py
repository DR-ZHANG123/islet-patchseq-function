"""Stage 09b：两部分模型的敏感性分析（反应阈值、细胞层面协变量调整、等效性界）与通用的按供体重抽样。

与 Stage 09 共用设计矩阵（T2D、糖尿病前期、葡萄糖、来源中心 + 供体随机截距）、随机截距 logistic / LMM 和
状态 × 中心分层的供体重抽样、中心内 ND/T2D 供体标签置换；区别只在于效应函数可替换。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from repro import ROOT
from t2d_effects import MAIN_SITES, STATUSES, donor_weights
from t2d_models import design, ri_lmm, ri_logit, weighted_mean_by
from t2d_resample import _bootstrap_frame, _permuted_frame, _positions, summarize
from t2d_signed import signed_values, tobit_effects

EXO = ["exocytosis", "early_exocytosis", "late_exocytosis"]


def load_beta(cell_type: str, cov_transform: dict) -> pd.DataFrame:
    """细胞表：三个胞吐终点（负值已在发布数据中截为 0）、细胞层面协变量（按配置变换）、供体状态与中心。"""
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    c = cells[cells.qc_pass & (cells.cell_type == cell_type) & cells.status.isin(STATUSES)]
    c = c.dropna(subset=EXO).reset_index(drop=True)
    df = pd.DataFrame({"donor": c.donor.values, "status": c.status.values, "glucose": c.glucose.values,
                       "site": c.site.where(c.site.isin(MAIN_SITES), "Other").values})
    for ep in EXO:
        df[ep] = c[ep].clip(lower=0).values
    for v, how in cov_transform.items():
        x = c[v].values.astype(float)
        df[f"cov_{v}"] = np.log(x) if how == "log" else np.log1p(np.clip(x, 0, None))
    df["t2d"] = (df.status == "T2D").astype(float)
    df["pre"] = (df.status == "preT2D").astype(float)
    return df


def responder_share_diff(df: pd.DataFrame) -> float:
    """反应细胞比例（每位供体等权）T2D − ND，未调整协变量的描述性差值。"""
    s = df[df.status.isin(["ND", "T2D"])]
    w = donor_weights(s.donor.values)
    r = (s.y.values > 0).astype(float)
    t = s.status.values == "T2D"
    return weighted_mean_by(r[t], w[t]) - weighted_mean_by(r[~t], w[~t])


def two_part(df: pd.DataFrame, spec: dict) -> dict:
    """阈值 c 下的两部分模型：反应 = y > c 的比值比；正值部分 = y > c 细胞的 log y 之 T2D 系数（REML）。"""
    cov = tuple(spec.get("covariates", ()))
    c = spec["threshold"]
    d = df.assign(y=np.where(df.y.values > c, df.y.values, 0.0))
    X, names = design(d, cov)
    j = names.index("t2d")
    nz = (d.y.values > 0).astype(float)
    res = {}
    if min(nz.sum(), (1 - nz).sum()) >= spec["min_zero_cells"]:
        res["nonzero_logor"] = ri_logit(nz, X, d.donor.values, spec["gh_nodes"], spec["logit_ridge"])["beta"][j]
    pos = d[d.y > 0]
    Xp, npos = design(pos, cov)
    res["positive_logratio"] = ri_lmm(np.log(pos.y.values), Xp, pos.donor.values)["beta"][npos.index("t2d")]
    res["responder_share_diff"] = responder_share_diff(d)
    return res


EFFECTS = {"two_part": two_part, "tobit": tobit_effects}


def _safe(fn, df, spec):
    try:
        return fn(df, spec)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {}


def _chunk(df, kind, spec, n, seed):
    fn = EFFECTS[spec["effect"]]
    rng = np.random.default_rng(seed)
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    pos = _positions(df)
    strata = [g.index.values for _, g in dtab.groupby(["status", "site"])]
    return [_safe(fn, _bootstrap_frame(df, pos, strata, rng) if kind == "boot" else _permuted_frame(df, dtab, rng),
                  spec) for _ in range(n)]


def resample_fn(df: pd.DataFrame, spec: dict, n_boot: int, n_perm: int, seed: int, n_jobs: int,
                chunk: int = 50) -> pd.DataFrame:
    """点估计 + 供体重抽样置信区间 + 中心内供体标签置换 P 值（种子由 SeedSequence 派生，与并行度无关）。"""
    df = df.reset_index(drop=True)
    point = EFFECTS[spec["effect"]](df, spec)
    tasks = [("boot", min(chunk, n_boot - i)) for i in range(0, n_boot, chunk)] + \
            [("perm", min(chunk, n_perm - i)) for i in range(0, n_perm, chunk)]
    seeds = np.random.SeedSequence(seed).spawn(len(tasks))
    res = Parallel(n_jobs=n_jobs)(delayed(_chunk)(df, k, spec, n, int(s.generate_state(1)[0]))
                                  for (k, n), s in zip(tasks, seeds))
    boots = pd.DataFrame([r for (k, _), rr in zip(tasks, res) if k == "boot" for r in rr], columns=list(point))
    perms = pd.DataFrame([r for (k, _), rr in zip(tasks, res) if k == "perm" for r in rr], columns=list(point))
    out = summarize(point, boots, perms)
    out.loc[out.effect.str.contains("share_diff"), "scale"] = "difference"
    out.loc[out.effect.isin(["tobit_asinh_coef", "tobit_raw_diff"]), "scale"] = "difference"
    st = df.groupby("donor").status.first()
    return out.assign(n_cells=len(df), n_donors=len(st), n_nd_donors=int((st == "ND").sum()),
                      n_t2d_donors=int((st == "T2D").sum()))


def equivalence(tab: pd.DataFrame, or_margins: list, share_margins: list) -> pd.DataFrame:
    """等效性界（双单侧检验，α = 0.05 → 90% 置信区间）：最小可宣称等效的对称界，以及给定界下是否等效。"""
    rows = []
    for _, r in tab.iterrows():
        if r.effect == "nonzero_odds_ratio":
            bound = max(r.hi90, 1 / r.lo90)
            rec = {f"equivalent_within_OR_{m:g}": bool(1 / m < r.lo90 and r.hi90 < m) for m in or_margins}
            unit = "odds ratio (symmetric on log scale)"
        elif r.effect == "responder_share_diff":
            bound = max(abs(r.lo90), abs(r.hi90))
            rec = {f"equivalent_within_{m:g}": bool(-m < r.lo90 and r.hi90 < m) for m in share_margins}
            unit = "difference in responder share (proportion)"
        else:
            continue
        rows.append({**{k: r[k] for k in ("analysis", "endpoint", "threshold", "effect", "estimate", "lo", "hi",
                                          "lo90", "hi90", "p_perm") if k in r},
                     "smallest_equivalence_bound": bound, "bound_unit": unit, **rec})
    return pd.DataFrame(rows)
