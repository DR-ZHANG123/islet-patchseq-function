"""稳健性分析的评价：三个层面的指标、按供体重抽样的置信区间与配对差值、置换零分布校准。

层面定义与 Stage 03c 相同：细胞层面 Spearman；供体层面为葡萄糖内标准化后的供体均值 Spearman（≥ min_group 个细胞）；
供体内层面为比较组（供体 × 葡萄糖）内去均值后的 Pearson。配对差值在同一组重抽样供体上计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

from metrics import donor_means, three_level

LEVELS = ("cell", "donor", "within")


def attach_meta(pred: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    p = pred.join(cells.set_index("cell_id")[["donor", "glucose"]], on="cell_id")
    p["group"] = p.donor + "|" + p.glucose.astype(str)
    return p


def per_repeat_metrics(pred: pd.DataFrame, keys: list, cfg: dict) -> pd.DataFrame:
    rows = []
    for k, s in pred.groupby(keys):
        rows.append(dict(zip(keys, k)) | three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                                     cfg["cv"]["min_within"], s.donor.values, s.glucose.values))
    return pd.DataFrame(rows)


def _corr(a, b, rank):
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return pearsonr(rankdata(a), rankdata(b))[0] if rank else pearsonr(a, b)[0]


def _prepare(s: pd.DataFrame, min_group: int) -> dict:
    s = s.copy()
    s["yc"] = s.y - s.groupby("group").y.transform("mean")
    s["pc"] = s.pred - s.groupby("group").pred.transform("mean")
    dm = donor_means(s.y.values, s.pred.values, s.donor.values, s.glucose.values, min_group)
    by = {d: (g.y.values, g.pred.values, g.yc.values, g.pc.values) for d, g in s.groupby("donor")}
    return {"by": by, "dm": dm}


def _stats(prep: dict, donors) -> np.ndarray:
    ys, ps, ycs, pcs, dy, dp = [], [], [], [], [], []
    for d in donors:
        if d not in prep["by"]:
            continue
        y, p, yc, pc = prep["by"][d]
        ys.append(y); ps.append(p); ycs.append(yc); pcs.append(pc)
        if d in prep["dm"].index:
            dy.append(prep["dm"].at[d, "y"]); dp.append(prep["dm"].at[d, "p"])
    y, p, yc, pc = map(np.concatenate, (ys, ps, ycs, pcs))
    return np.array([_corr(y, p, True), _corr(np.array(dy), np.array(dp), True), _corr(yc, pc, False)])


def boot_compare(avg: pd.DataFrame, pairs: list, n_boot: int, seed: int, min_group: int, label: dict) -> tuple:
    """avg：每个细胞一行（列 model, y, pred, donor, glucose, group）。返回 (指标表, 配对差值表)。
    pairs：[(模型A, 模型B)]，差值 = A − B；P 为双侧重抽样 P 值。"""
    rng = np.random.default_rng(seed)
    models = sorted(avg.model.unique())
    preps = {m: _prepare(avg[avg.model == m], min_group) for m in models}
    donors = np.array(sorted(avg.donor.unique()))
    boots = {m: [] for m in models}
    for _ in range(n_boot):
        pick = rng.choice(donors, len(donors), replace=True)
        for m in models:
            boots[m].append(_stats(preps[m], pick))
    boots = {m: np.array(v) for m, v in boots.items()}
    point = {m: _stats(preps[m], donors) for m in models}
    met, dif = [], []
    for m in models:
        lo, hi = np.nanpercentile(boots[m], [2.5, 97.5], axis=0)
        up95 = np.nanpercentile(boots[m], 95, axis=0)
        for j, lev in enumerate(LEVELS):
            met.append(label | {"model": m, "level": lev, "estimate": point[m][j], "lo": lo[j], "hi": hi[j],
                                "upper95_one_sided": up95[j], "boot_sd": np.nanstd(boots[m][:, j]),
                                "n_donors": len(donors)})
    for a, b in pairs:
        if a not in boots or b not in boots:
            continue
        d = boots[a] - boots[b]
        pt = point[a] - point[b]
        lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
        for j, lev in enumerate(LEVELS):
            dj = d[:, j][~np.isnan(d[:, j])]
            pv = min(1.0, 2 * min((dj <= 0).mean(), (dj >= 0).mean())) if len(dj) else np.nan
            dif.append(label | {"model_a": a, "model_b": b, "level": lev, "difference": pt[j], "lo": lo[j],
                                "hi": hi[j], "p_boot": pv})
    return pd.DataFrame(met), pd.DataFrame(dif)


def average_repeats(pred: pd.DataFrame, keys: list) -> pd.DataFrame:
    """每个细胞在各重复中的留出预测取平均（与 Stage 03c 的点估计相同）。"""
    return pred.groupby(keys + ["cell_id", "donor", "glucose", "group"], as_index=False).agg(
        y=("y", "mean"), pred=("pred", "mean"))


def calibrate(obs: float, null: np.ndarray) -> dict:
    """置换零分布校准：零均值（期望偏倚）、校准后效应 = 观测 − 零均值、单侧置换 P = (1 + #{零 ≥ 观测}) / (B + 1)。"""
    null = null[~np.isnan(null)]
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_q025": float(np.quantile(null, 0.025)), "null_q975": float(np.quantile(null, 0.975)),
            "calibrated": obs - float(null.mean()), "p_perm": float((1 + (null >= obs).sum()) / (len(null) + 1)),
            "n_perm": int(len(null))}
