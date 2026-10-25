"""学习曲线的按留出供体重抽样推断（Stage 04g）。

每次抽样 d 的 10 个外层折合起来覆盖全部开发供体，因此同一组重抽样供体可同时用于所有 (比例, 抽样)：
每次重抽样先在每个 (比例, 抽样) 上算三个层面的指标，再在抽样间取平均得到该比例的曲线值；
斜率为曲线值对平均训练供体数的最小二乘斜率（以每 10 位训练供体计），差值为最大比例 − 最小比例。
置信区间取重抽样分位数，P 为双侧重抽样 P 值（下限按 1/n_boot 报告）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import robust_eval as re_

LEVELS = re_.LEVELS


def _slope(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """c：(n_frac, n_level)；返回每个层面的斜率（忽略缺失）。"""
    out = np.full(c.shape[1], np.nan)
    for j in range(c.shape[1]):
        m = ~np.isnan(c[:, j])
        if m.sum() >= 2:
            out[j] = np.polyfit(x[m], c[m, j], 1)[0]
    return out


def _p_two_sided(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())) if len(v) else np.nan


def boot_curve(s: pd.DataFrame, n_train: dict, n_boot: int, seed: int, min_group: int, label: dict):
    """s：某 细胞类型 × 终点 × 模型 的逐细胞留出预测（列 fraction, draw, cell_id, y, pred, donor, glucose, group）。
    n_train：比例 → 平均训练供体数。返回 (曲线表, 趋势表, 逐抽样点估计表)。"""
    fracs = sorted(s.fraction.unique())
    draws = sorted(s.draw.unique())
    preps = {(f, d): re_._prepare(g, min_group) for (f, d), g in s.groupby(["fraction", "draw"])}
    donors = np.array(sorted(s.donor.unique()))
    x = np.array([n_train[f] for f in fracs]) / 10.0

    def curve(pick):
        per = {k: re_._stats(p, pick) for k, p in preps.items()}
        c = np.array([np.nanmean([per[(f, d)] for d in draws if (f, d) in per], axis=0) for f in fracs])
        return c, per

    point, per_point = curve(donors)
    rng = np.random.default_rng(seed)
    boots = np.array([curve(rng.choice(donors, len(donors), replace=True))[0] for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    rows = []
    for i, f in enumerate(fracs):
        draw_vals = np.array([per_point[(f, d)] for d in draws if (f, d) in per_point])
        for j, lev in enumerate(LEVELS):
            rows.append(label | {"fraction": f, "n_train_donors": n_train[f], "level": lev,
                                 "estimate": point[i, j], "lo": lo[i, j], "hi": hi[i, j],
                                 "draw_sd": float(np.nanstd(draw_vals[:, j], ddof=1)) if len(draw_vals) > 1 else np.nan,
                                 "n_draws": len(draw_vals), "n_donors": len(donors)})
    sl_pt, sl_b = _slope(x, point), np.array([_slope(x, b) for b in boots])
    df_pt, df_b = point[-1] - point[0], boots[:, -1] - boots[:, 0]
    trend = []
    for j, lev in enumerate(LEVELS):
        sl_lo, sl_hi = np.nanpercentile(sl_b[:, j], [2.5, 97.5])
        d_lo, d_hi = np.nanpercentile(df_b[:, j], [2.5, 97.5])
        trend.append(label | {"level": lev, "slope_per_10_donors": sl_pt[j], "slope_lo": sl_lo, "slope_hi": sl_hi,
                              "slope_p_boot": _p_two_sided(sl_b[:, j]),
                              "diff_max_vs_min": df_pt[j], "diff_lo": d_lo, "diff_hi": d_hi,
                              "diff_p_boot": _p_two_sided(df_b[:, j]), "fraction_min": fracs[0],
                              "fraction_max": fracs[-1], "n_boot": n_boot})
    per_rows = [label | {"fraction": f, "draw": d, **dict(zip(LEVELS, v))} for (f, d), v in per_point.items()]
    return pd.DataFrame(rows), pd.DataFrame(trend), pd.DataFrame(per_rows)
