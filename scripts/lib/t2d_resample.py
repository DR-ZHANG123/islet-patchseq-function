"""Stage 09：按供体的推断。

置信区间：在 状态 × 来源中心 层内按供体有放回重抽样（重复抽到的供体各自成簇），取 2.5/97.5 百分位。
P 值：在每个来源中心内置换 ND 与 T2D 供体的状态标签（糖尿病前期供体保持不变），
P = (1 + #{|T*| ≥ |T|}) / (1 + 有效置换数)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from t2d_effects import effects

# 只描述、不检验的量（例如 ND 组自身的比例）
DESCRIPTIVE = ("_nd", "_t2d", "_expected")


def _positions(df):
    return {d: np.where(df.donor.values == d)[0] for d in df.donor.unique()}


def _bootstrap_frame(df, pos, strata, rng):
    idx, cid = [], []
    k = 0
    for donors in strata:
        for d in rng.choice(donors, len(donors), replace=True):
            idx.append(pos[d])
            cid.append(np.full(len(pos[d]), k))
            k += 1
    b = df.iloc[np.concatenate(idx)].copy()
    b["donor"] = np.concatenate(cid)
    return b


def _permuted_frame(df, dtab, rng):
    lab = dtab.status.copy()
    for _, s in dtab[dtab.status.isin(["ND", "T2D"])].groupby("site"):
        lab.loc[s.index] = rng.permutation(s.status.values)
    st = df.donor.map(lab)
    p = df.copy()
    p["status"] = st.values
    p["t2d"] = (st == "T2D").astype(float).values
    return p


def _safe(df, spec):
    try:
        return effects(df, spec)
    except (ValueError, np.linalg.LinAlgError):
        return {}


def _chunk(df, spec, kind, n, seed):
    rng = np.random.default_rng(seed)
    dtab = df.groupby("donor").agg(status=("status", "first"), site=("site", "first"))
    pos = _positions(df)
    strata = [g.index.values for _, g in dtab.groupby(["status", "site"])]
    out = []
    for _ in range(n):
        frame = _bootstrap_frame(df, pos, strata, rng) if kind == "boot" else _permuted_frame(df, dtab, rng)
        out.append(_safe(frame, spec))
    return out


def resample(df: pd.DataFrame, spec: dict, n_boot: int, n_perm: int, seed: int, n_jobs: int, chunk: int = 50):
    """返回 (点估计 dict, 重抽样 DataFrame, 置换 DataFrame)。各块的随机种子由 SeedSequence 派生，结果与并行度无关。"""
    point = effects(df, spec)
    tasks = [("boot", min(chunk, n_boot - i)) for i in range(0, n_boot, chunk)] + \
            [("perm", min(chunk, n_perm - i)) for i in range(0, n_perm, chunk)]
    seeds = np.random.SeedSequence(seed).spawn(len(tasks))
    res = Parallel(n_jobs=n_jobs)(delayed(_chunk)(df, spec, k, n, int(s.generate_state(1)[0]))
                                  for (k, n), s in zip(tasks, seeds))
    boots = [r for (k, _), rr in zip(tasks, res) if k == "boot" for r in rr]
    perms = [r for (k, _), rr in zip(tasks, res) if k == "perm" for r in rr]
    return point, pd.DataFrame(boots, columns=list(point)), pd.DataFrame(perms, columns=list(point))


def summarize(point: dict, boots: pd.DataFrame, perms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, v in point.items():
        b = boots[k].dropna().values if k in boots else np.array([])
        p = perms[k].dropna().values if k in perms else np.array([])
        is_log = k.endswith("_logratio") or k.endswith("_logor")
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan))
        lo90 = np.percentile(b, [5, 95]) if len(b) else (np.nan, np.nan)
        test = not k.endswith(DESCRIPTIVE) and len(p) > 0 and np.isfinite(v)
        pval = (1 + np.sum(np.abs(p) >= abs(v) - 1e-12)) / (1 + len(p)) if test else np.nan
        tf = np.exp if is_log else (lambda x: x)
        name = k.replace("_logratio", "_ratio").replace("_logor", "_odds_ratio")
        rows.append({"effect": name, "scale": "ratio" if is_log else ("sd" if k == "mean_sd" else "proportion"),
                     "estimate": tf(v), "lo": tf(lo), "hi": tf(hi), "lo90": tf(lo90[0]), "hi90": tf(lo90[1]),
                     "p_perm": pval, "n_boot": len(b), "n_perm": len(p)})
    return pd.DataFrame(rows)
