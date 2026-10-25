"""Stage 06d 的估计与按供体重抽样置信区间。

一个“单元”（细胞类型 × 终点 × 模型）由若干个重复组成（外部 1 个，开发交叉验证 5 个）；
点估计与每次重抽样的值都先在每个重复内计算，再跨重复取平均。重抽样单位是供体：
有放回抽取供体，重复抽到的供体视为不同供体（比较组编码加偏移），同一组抽样用于全部重复。
另含跨平台供体层面比较：只用两种输入都有的供体，配对重抽样得到 Spearman 差值的置信区间。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calib_metrics import METRICS, SHIFT, SHIFT_CI, point_metrics, shift_by_glucose

warnings.filterwarnings("ignore", category=RuntimeWarning)   # 某葡萄糖浓度无细胞时的全 NaN 平均


def _subset(a: dict, idx: np.ndarray, gnew: np.ndarray) -> dict:
    out = {k: v[idx] for k, v in a.items() if k != "g"}
    out["g"] = gnew
    return out


def _one(reps, inv, levels, min_within, min_level):
    """单次（原始或重抽样）计算：返回 (指标向量, {葡萄糖: 分解向量})，均为跨重复平均。"""
    met, sh = [], {lv: [] for lv in ["all"] + list(levels)}
    for a in reps:
        m = point_metrics(a["y"], a["p"], a["base"], a["base_g"], a["g"], inv, min_within)
        met.append([m[k] for k in METRICS])
        for lv, d in shift_by_glucose(a["y"], a["p"], a["base"], a["glucose"], a["donor"], levels, *min_level).items():
            sh[lv].append([np.nan] * len(SHIFT) if d is None else [d[k] for k in SHIFT])
    return np.nanmean(met, 0), {lv: np.nanmean(v, 0) for lv, v in sh.items()}


def evaluate_unit(reps, inv, levels, n_boot, seed, min_within=2, min_level=(10, 3)):
    """返回 (指标字典含 _lo/_hi, 分解行列表)。min_level：报告某葡萄糖浓度所需的 (最少细胞数, 最少供体数)；
    原始数据中该浓度不达标时整行不报告（重抽样中偶然达标的值也不用）。"""
    pt_m, pt_s = _one(reps, inv, levels, min_within, min_level)
    donors = np.unique(np.concatenate([a["donor"] for a in reps]))
    pos = [{d: np.where(a["donor"] == d)[0] for d in np.unique(a["donor"])} for a in reps]
    rng = np.random.default_rng(seed)
    bm, bs = [], {lv: [] for lv in pt_s}
    for _ in range(n_boot):
        pick = rng.choice(donors, len(donors), replace=True)
        boot = []
        for a, ps in zip(reps, pos):
            gmax = a["g"].max() + 1
            parts = [(ps[d], j) for j, d in enumerate(pick) if d in ps]
            idx = np.concatenate([p for p, _ in parts])
            gnew = np.concatenate([a["g"][p] + j * gmax for p, j in parts])
            boot.append(_subset(a, idx, gnew))
        m, s = _one(boot, inv, levels, min_within, min_level)
        bm.append(m)
        for lv in s:
            bs[lv].append(s[lv])
    lo, hi = np.nanpercentile(np.array(bm), [2.5, 97.5], axis=0)
    met = {"n_donors": len(donors), "n_cells": int(np.mean([len(a["y"]) for a in reps])), "n_repeats": len(reps)}
    for k, name in enumerate(METRICS):
        met |= {name: pt_m[k], f"{name}_lo": lo[k], f"{name}_hi": hi[k]}
    rows = []
    for lv, v in pt_s.items():
        if not np.isfinite(v[0]):
            continue
        r = {"glucose": lv} | dict(zip(SHIFT, v))
        blo, bhi = np.nanpercentile(np.array(bs[lv]), [2.5, 97.5], axis=0)
        for name in SHIFT_CI:
            k = SHIFT.index(name)
            r |= {f"{name}_lo": blo[k], f"{name}_hi": bhi[k]}
        rows.append(r)
    return met, rows


def _rho(y, p):
    return spearmanr(y, p).statistic if np.std(y) > 0 and np.std(p) > 0 else np.nan


def common_donor_comparison(pred: pd.DataFrame, summary: pd.DataFrame, inputs, n_boot, seed) -> pd.DataFrame:
    """跨平台供体层面比较：每个细胞类型 × 终点只用两种输入都有的供体；差值 = inputs[0] − inputs[1]。"""
    rng = np.random.default_rng(seed)
    a_in, b_in = inputs
    rows = []
    for (ct, ep), s in pred.groupby(["cell_type", "endpoint"], sort=False):
        wa = s[s.input == a_in].set_index("donor")
        wb = s[s.input == b_in].set_index("donor")
        common = sorted(set(wa.index) & set(wb.index))
        y = wa.loc[common, "measured"].values
        pa, pb = wa.loc[common, "predicted"].values, wb.loc[common, "predicted"].values
        ra, rb = _rho(y, pa), _rho(y, pb)
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(common), len(common))
            boot.append([_rho(y[i], pa[i]), _rho(y[i], pb[i])])
        boot = np.array(boot)
        diff = boot[:, 0] - boot[:, 1]
        ok = np.isfinite(diff)
        full = summary[(summary.cell_type == ct) & (summary.endpoint == ep)].set_index("input")
        rows.append({"cell_type": ct, "endpoint": ep, "n_common_donors": len(common),
                     "rho_patchseq": ra, "rho_patchseq_lo": np.nanpercentile(boot[:, 0], 2.5),
                     "rho_patchseq_hi": np.nanpercentile(boot[:, 0], 97.5),
                     "rho_droplet": rb, "rho_droplet_lo": np.nanpercentile(boot[:, 1], 2.5),
                     "rho_droplet_hi": np.nanpercentile(boot[:, 1], 97.5),
                     "diff": ra - rb, "diff_lo": np.percentile(diff[ok], 2.5), "diff_hi": np.percentile(diff[ok], 97.5),
                     "boot_frac_diff_le0": float(np.mean(diff[ok] <= 0)),
                     "n_full_patchseq": int(full.loc[a_in, "n_donors"]),
                     "rho_full_patchseq": full.loc[a_in, "spearman"],
                     "n_full_droplet": int(full.loc[b_in, "n_donors"]),
                     "rho_full_droplet": full.loc[b_in, "spearman"]})
    return pd.DataFrame(rows)
