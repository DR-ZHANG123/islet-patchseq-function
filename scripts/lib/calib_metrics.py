"""绝对尺度预测的校准指标（Stage 06d）。

所有指标在分析尺度上计算（细胞大小取 log，半失活电压为原始 mV，其余终点取 log1p），
MAE 另在原始测量尺度上给出（预测与基线经逆变换回到原始单位）。

- R²（三种参照）：1 − SSE / Σ(y − b)²，b 分别为
  (a) 测试集均值（现有口径，部署时不可得）；
  (b) 训练集均值（可部署的常数基线）；
  (c) 训练集在同一葡萄糖浓度下的均值。
  (b)(c) 为负表示模型的平方误差大于直接报告训练均值。
- 校准斜率与截距：实测值对预测值的最小二乘回归（斜率 1、截距 0 为理想）。
- 供体内斜率：比较组（供体 × 葡萄糖）内去均值后实测值对预测值的回归斜率。
- 误差分解（Theil）：MSE = (p̄ − ȳ)² + (s_p − r·s_y)² + (1 − r²)·s_y²，
  即 R²(测试均值) = r² − 位置偏移项 − 尺度失配项。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# 逆变换：分析尺度 → 原始测量尺度
INVERSE = {"log": np.exp, "log1p": np.expm1, "none": lambda v: v}
SCALE_NAME = {"log": "log", "log1p": "log1p", "none": "raw"}

METRICS = ["spearman", "r2_test_mean", "r2_train_mean", "r2_train_glucose_mean", "mae", "mae_train_mean",
           "mae_train_glucose_mean", "mae_ratio_train_mean", "mae_orig", "mae_orig_train_mean",
           "mae_orig_train_glucose_mean", "calib_slope", "calib_intercept", "within_slope"]
SHIFT = ["n_cells", "n_donors", "measured_mean", "pred_mean", "train_mean", "shift", "measured_minus_train",
         "pred_minus_train", "sd_measured", "sd_pred", "pearson_r", "mse", "location_term", "scale_term",
         "residual_term", "r2_test_mean", "location_penalty", "scale_penalty", "r2_linear"]
SHIFT_CI = ["shift", "measured_minus_train", "pred_minus_train", "location_penalty", "scale_penalty",
            "r2_test_mean", "r2_linear"]


def _r2(y, p, b):
    den = np.sum((y - b) ** 2)
    return 1 - np.sum((y - p) ** 2) / den if den > 0 else np.nan


def _corr(a, b):
    sa, sb = a.std(), b.std()
    if len(a) < 3 or sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _centre(v, g):
    """比较组内去均值；g 为 0..G-1 的整数编码。"""
    s = np.bincount(g, v)
    n = np.bincount(g)
    return v - (s / np.maximum(n, 1))[g], n[g]


def point_metrics(y, p, base, base_g, g, inv, min_within: int = 2) -> dict:
    """y、p：分析尺度；base、base_g：每个细胞对应的训练均值与同葡萄糖训练均值；g：比较组编码。"""
    out = {"spearman": _corr(rankdata(y), rankdata(p)),
           "r2_test_mean": _r2(y, p, y.mean()), "r2_train_mean": _r2(y, p, base),
           "r2_train_glucose_mean": _r2(y, p, base_g),
           "mae": np.mean(np.abs(y - p)), "mae_train_mean": np.mean(np.abs(y - base)),
           "mae_train_glucose_mean": np.mean(np.abs(y - base_g))}
    out["mae_ratio_train_mean"] = out["mae"] / out["mae_train_mean"] if out["mae_train_mean"] > 0 else np.nan
    yo = inv(y)
    out["mae_orig"] = np.mean(np.abs(yo - inv(p)))
    out["mae_orig_train_mean"] = np.mean(np.abs(yo - inv(base)))
    out["mae_orig_train_glucose_mean"] = np.mean(np.abs(yo - inv(base_g)))
    vp = p.var()
    slope = np.mean((p - p.mean()) * (y - y.mean())) / vp if vp > 1e-12 else np.nan
    out["calib_slope"] = slope
    out["calib_intercept"] = y.mean() - slope * p.mean()
    yc, n = _centre(y, g)
    pc, _ = _centre(p, g)
    k = n >= min_within
    den = np.sum(pc[k] ** 2)
    out["within_slope"] = np.sum(yc[k] * pc[k]) / den if den > 1e-12 else np.nan
    return out


def shift_terms(y, p, base, donor) -> dict:
    """位置偏移与尺度失配分解（总体或单个葡萄糖浓度内）。"""
    sy, sp = y.std(), p.std()
    r = _corr(y, p)
    mse = np.mean((y - p) ** 2)
    loc = (p.mean() - y.mean()) ** 2
    scale = (sp - r * sy) ** 2 if np.isfinite(r) else np.nan
    var = sy ** 2
    return {"n_cells": len(y), "n_donors": len(np.unique(donor)), "measured_mean": y.mean(), "pred_mean": p.mean(), "train_mean": np.mean(base),
            "shift": p.mean() - y.mean(), "measured_minus_train": y.mean() - np.mean(base),
            "pred_minus_train": p.mean() - np.mean(base), "sd_measured": sy, "sd_pred": sp, "pearson_r": r,
            "mse": mse, "location_term": loc, "scale_term": scale, "residual_term": (1 - r ** 2) * var,
            "r2_test_mean": 1 - mse / var if var > 0 else np.nan,
            "location_penalty": loc / var if var > 0 else np.nan,
            "scale_penalty": scale / var if var > 0 else np.nan, "r2_linear": r ** 2}


def shift_by_glucose(y, p, base, glucose, donor, levels, min_cells: int, min_donors: int) -> dict:
    """返回 {"all": 分解, 1: ..., 5: ..., 10: ...}；某浓度细胞或供体不足时为 None（不报告该浓度）。"""
    out = {"all": shift_terms(y, p, base, donor)}
    for gl in levels:
        k = glucose == gl
        ok = k.sum() >= min_cells and len(np.unique(donor[k])) >= min_donors
        out[gl] = shift_terms(y[k], p[k], base[k], donor[k]) if ok else None
    return out
