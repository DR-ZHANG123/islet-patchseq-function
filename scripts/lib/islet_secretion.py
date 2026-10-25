"""Stage 11：供体层面，Patch-seq 单细胞功能与同一供体胰岛胰岛素分泌（灌流与静态 GSIS）的关系。

单细胞汇总（每种细胞类型，每位供体等权）：
  - 各终点（建模尺度：log1p / log）先在同一葡萄糖浓度内对全部细胞去均值，再取供体均值；
  - 胞吐正值反应：log(原始值)，只用 > 0 的细胞；反应比例：原始值 > 0 的比例（同样按葡萄糖去均值）；
  - 偶联效率与 Ca²⁺ 可解释部分：与 Stage 04j 相同的分解 E = b·X + r（X = Ca²⁺ 内流、早/晚期 Ca²⁺ 电流、大小），
    b 为比较组（供体 × 葡萄糖）内去均值后的合并斜率，此处用全部完整细胞估计（描述性汇总，不做预测）；
  - 转录组预测值：Stage 03 留出供体交叉验证中岭–提升集成模型（cell_branch）的逐细胞预测，五次重复取平均。
胰岛分泌（HumanIslets）：灌流 AUC（3 mM 基线、6/15 mM 葡萄糖、KCl、亮氨酸、油酸/棕榈酸），取 log；
  灌流葡萄糖刺激倍数 = log(AUC 15 mM / AUC 3 mM)；静态 GSIS 取 2.8→16.7 mM 方案各重复的 log 刺激指数与
  16.7 mM 下分泌占含量百分比的 log，供体内取中位数。
统计：供体间 Spearman ρ；按糖尿病状态（T2D、糖尿病前期）调整的偏 Spearman（秩对状态哑变量回归后的残差相关）；
  供体重抽样（按状态分层）95% 区间；置换 P（偏相关：状态内置换胰岛分泌值）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from features import group_center

EXO = ("exocytosis", "early_exocytosis", "late_exocytosis")
SHORT = {"exocytosis": "total", "early_exocytosis": "early", "late_exocytosis": "late"}


# ---------- 单细胞功能的供体汇总 ----------

def _glucose_centre(v: np.ndarray, glucose: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.groupby(glucose).transform("mean")).values


def _donor_mean(v: np.ndarray, donor: np.ndarray, glucose: np.ndarray, min_cells: int) -> pd.Series:
    ok = ~np.isnan(v)
    c = _glucose_centre(v[ok], glucose[ok])
    s = pd.Series(c).groupby(donor[ok])
    m = s.mean()
    return m[s.size() >= min_cells]


def coupling_parts(td, cc: dict) -> dict[str, np.ndarray]:
    """Stage 04j 的分解（全部完整细胞估计斜率）：返回 {ca_explained_<s>, coupling_<s>}（log1p 尺度）。"""
    xi = [td.endpoints.index(v) for v in cc["covariates"]]
    x = td.y[:, xi]
    out = {}
    for t in cc["targets"]:
        e = td.y[:, td.endpoints.index(t)]
        ok = ~np.isnan(e) & ~np.isnan(x).any(1)
        b, *_ = np.linalg.lstsq(group_center(x[ok], td.group[ok]), group_center(e[ok], td.group[ok]), rcond=None)
        expl = np.full(len(e), np.nan)
        expl[ok] = x[ok] @ b
        out[f"ca_explained_{SHORT[t]}"] = expl
        out[f"coupling_{SHORT[t]}"] = e - expl
    return out


def cell_function(td, cc: dict, pred: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    """一种细胞类型的供体层面功能汇总（行 = 供体，列 = 特征）。"""
    feats = {}
    for k, ep in enumerate(td.endpoints):
        feats[ep] = td.y[:, k]
    for ep in EXO:
        raw = td.cells[ep].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f"positive_{SHORT[ep]}"] = np.where(raw > 0, np.log(raw), np.nan)
        feats[f"responding_{SHORT[ep]}"] = np.where(np.isnan(raw), np.nan, (raw > 0).astype(float))
    feats |= coupling_parts(td, cc)
    p = pred.groupby(["cell_id", "endpoint"]).pred.mean().unstack()
    p = p.reindex(td.cells.cell_id.values)
    for ep in p.columns:
        feats[f"predicted_{ep}"] = p[ep].values.astype(float)
    cols = {f: _donor_mean(v, td.donor, td.glucose, min_cells) for f, v in feats.items()}
    return pd.DataFrame(cols)


# ---------- 胰岛分泌汇总 ----------

def islet_secretion(peri: pd.DataFrame, gsis: pd.DataFrame, sc: dict) -> pd.DataFrame:
    out = peri.set_index("record_id")[sc["perifusion_columns"]].astype(float)
    out = np.log(out.where(out > 0))
    out.columns = [f"peri_{c}" for c in out.columns]
    out["peri_glucose_fold"] = out["peri_auc_gluc_15mmgluc"] - out["peri_auc_baseline_3mmgluc"]
    g = gsis[gsis.gluc_conc_group == sc["gsis_protocol"]].copy()
    for c in ("stim_index", "second_insulin_percent"):
        v = g[c].astype(float)
        g[c] = np.log(v.where((v > 0) & np.isfinite(v)))
    gm = g.groupby("record_id")[["stim_index", "second_insulin_percent"]].median()
    gm.columns = ["gsis_log_stim_index", "gsis_log_secreted_percent"]
    return out.join(gm, how="outer")


# ---------- 相关与推断 ----------

def _status_design(status: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(status))]
    for s in ("T2D", "preT2D"):
        v = (status == s).astype(float)
        if np.ptp(v) > 0:
            cols.append(v)
    return np.column_stack(cols)


def _resid(v: np.ndarray, X: np.ndarray) -> np.ndarray:
    b, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ b


def rho(x: np.ndarray, y: np.ndarray, status: np.ndarray | None = None) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if status is not None:
        X = _status_design(status)
        rx, ry = _resid(rx, X), _resid(ry, X)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def infer_pair(x, y, status, n_boot: int, n_perm: int, seed: int) -> dict:
    """一对供体层面变量：未调整与按状态调整的 ρ、分层重抽样区间、置换 P（调整时状态内置换）。"""
    rng = np.random.default_rng(seed)
    out = {"n_donors": len(x), "n_t2d": int((status == "T2D").sum()), "n_pre": int((status == "preT2D").sum())}
    strata = {s: np.where(status == s)[0] for s in np.unique(status)}
    for lab, st in (("rho", None), ("rho_adj", status)):
        est = rho(x, y, st)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ix, len(ix)) for ix in strata.values()])
            boots.append(rho(x[idx], y[idx], None if st is None else st[idx]))
        perms = []
        for _ in range(n_perm):
            yp = y.copy()
            if st is None:
                yp = rng.permutation(y)
            else:
                for ix in strata.values():
                    yp[ix] = rng.permutation(y[ix])
            perms.append(rho(x, yp, st))
        b = np.array(boots, float)
        b = b[np.isfinite(b)]
        p = np.array(perms, float)
        out |= {lab: est, f"{lab}_lo": np.percentile(b, 2.5), f"{lab}_hi": np.percentile(b, 97.5),
                f"{lab}_p": (1 + np.sum(np.abs(p) >= abs(est) - 1e-12)) / (1 + len(p))}
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(len(pv))
    res[order] = np.minimum(ranked, 1.0)
    q[ok] = res
    return q


def status_difference(y: np.ndarray, status: np.ndarray, n_boot: int, n_perm: int, seed: int) -> dict:
    """胰岛分泌（log 尺度）T2D − ND 均值差：状态内供体重抽样区间与 ND/T2D 标签置换 P。"""
    rng = np.random.default_rng(seed)
    nd, t2 = y[status == "ND"], y[status == "T2D"]
    est = t2.mean() - nd.mean()
    boots = [rng.choice(t2, len(t2)).mean() - rng.choice(nd, len(nd)).mean() for _ in range(n_boot)]
    pool = np.r_[nd, t2]
    perms = []
    for _ in range(n_perm):
        p = rng.permutation(pool)
        perms.append(p[len(nd):].mean() - p[:len(nd)].mean())
    perms = np.array(perms)
    return {"difference_log": est, "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
            "ratio": np.exp(est), "ratio_lo": np.exp(np.percentile(boots, 2.5)),
            "ratio_hi": np.exp(np.percentile(boots, 97.5)),
            "p_perm": (1 + np.sum(np.abs(perms) >= abs(est) - 1e-12)) / (1 + n_perm),
            "n_nd": len(nd), "n_t2d": len(t2)}
