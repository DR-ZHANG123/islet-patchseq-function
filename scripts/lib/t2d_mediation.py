"""Stage 09g 用函数：允许特征得分与胞吐的关系在 T2D 与 ND 细胞中不同（暴露 × 中介交互）时的中介分解。

线性模型（供体等权，协变量 = 糖尿病前期、葡萄糖、来源中心）：
    M = α0 + a·T + γ'C,        Y = θ0 + θ_t·T + θ_s·M + θ_int·T·M + β'C。
自然间接效应 NIE = a·(θ_s + θ_int)，自然直接效应 NDE = θ_t + θ_int·E[M | T = 0]（按协变量分布标准化），
总效应 TE = NDE + NIE，携带比例 = NIE / TE（VanderWeele 线性形式）。不含交互项时 NIE / TE 与 Stage 09f 的
（总 − 直接）/ 总一致。区间来自状态 × 中心分层的供体重抽样。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from t2d_conditioned import boot_df
from t2d_effects import donor_weights
from t2d_models import design
from t2d_robust import summarise_boot


def _fit(X: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return beta


def decompose(df: pd.DataFrame, col: str) -> dict:
    X, names = design(df)
    j = names.index("t2d")
    w = donor_weights(df.donor.values)
    m = df[col].values
    t = df.t2d.values
    bm = _fit(X, w, m)
    a = bm[j]
    X0 = X.copy()
    X0[:, j] = 0.0
    if "pre" in names:
        X0[:, names.index("pre")] = 0.0
    m0 = float(np.average(X0 @ bm, weights=w))            # E[M | T = 0]，按协变量分布标准化
    by = _fit(np.column_stack([X, m, t * m]), w, df.y.values)
    th_t, th_s, th_int = by[j], by[X.shape[1]], by[X.shape[1] + 1]
    nie = a * (th_s + th_int)
    nde = th_t + th_int * m0
    te = nde + nie
    tot = _fit(X, w, df.y.values)[j]
    return {"total_without_mediator": tot, "a_score_t2d": a, "theta_score": th_s, "theta_interaction": th_int,
            "nde": nde, "nie": nie, "te": te, "share_with_interaction": nie / te if te != 0 else np.nan}


def boot_decompose(df: pd.DataFrame, col: str, n_boot: int, rng) -> list[dict]:
    point = decompose(df, col)
    return summarise_boot(point, [decompose(boot_df(df, rng), col) for _ in range(n_boot)])
