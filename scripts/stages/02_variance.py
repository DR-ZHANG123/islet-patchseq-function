"""Stage 02：功能终点的供体间/供体内方差结构与糖代谢状态效应（全部供体，描述性分析）。

- 组内相关（ICC）：先在葡萄糖浓度内对终点去均值，再用单因素方差分析的矩估计
  ICC(1) = (MSB − MSW) / (MSB + (n₀ − 1)·MSW)（n₀ 为不等组大小的校正组大小，负值截为 0）；
  95% 置信区间来自按供体的重抽样。
- 糖代谢状态效应：y（标准化）~ 状态 + 葡萄糖 + 来源中心 + (1 | 供体)，参照组为 ND；不含 T1D 供体。
- 终点之间在供体内的相关：比较组内秩去均值后的 Pearson 相关。
输出 results/02_variance/{icc.tsv, status_effects.tsv, within_endpoint_correlation.tsv, cohort_summary.tsv}。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "02_variance"


def fit_mixed(formula: str, df: pd.DataFrame):
    """线性混合模型（供体随机截距）；数值优化失败时依次换用其他优化方法。"""
    last = None
    for kw in ({}, {"method": ["lbfgs"]}, {"method": ["powell"]}):
        try:
            return smf.mixedlm(formula, df, groups=df["donor"]).fit(reml=True, **kw)
        except (ValueError, np.linalg.LinAlgError) as exc:
            last = exc
    raise last


def icc(df: pd.DataFrame) -> float:
    """单因素方差分析 ICC(1)，终点先在葡萄糖浓度内去均值。"""
    y = df.y - df.groupby("glucose").y.transform("mean")
    g = df.donor.values
    k = pd.Series(y.values).groupby(g).agg(["mean", "size"])
    n, a = len(y), len(k)
    grand = y.mean()
    ssb = (k["size"] * (k["mean"] - grand) ** 2).sum()
    ssw = ((y.values - pd.Series(g).map(k["mean"]).values) ** 2).sum()
    msb, msw = ssb / (a - 1), ssw / (n - a)
    n0 = (n - (k["size"] ** 2).sum() / n) / (a - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))


def icc_with_ci(df, n_boot, rng):
    est = icc(df)
    donors = df.donor.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(donors, len(donors), replace=True)
        b = pd.concat([df[df.donor == d].assign(donor=f"{d}#{j}") for j, d in enumerate(pick)])
        try:
            boots.append(icc(b))
        except (ValueError, np.linalg.LinAlgError):
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    icc_rows, st_rows, cor_rows, coh = [], [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "all")
        c = td.cells
        coh.append(c.groupby(["site", "status", "glucose"]).agg(cells=("cell_id", "size"),
                                                                donors=("donor", "nunique")).reset_index()
                   .assign(cell_type=ct))
        for k, ep in enumerate(td.endpoints):
            df = pd.DataFrame({"y": td.y[:, k], "donor": td.donor, "glucose": td.glucose,
                               "status": c.status.values, "site": c.site.values}).dropna()
            df["y"] = (df.y - df.y.mean()) / df.y.std()
            est, lo, hi = icc_with_ci(df, cfg["variance"]["n_boot"], rng)
            icc_rows.append({"cell_type": ct, "endpoint": ep, "n_cells": len(df), "n_donors": df.donor.nunique(),
                             "icc": est, "icc_lo": lo, "icc_hi": hi})
            s = df[df.status.isin(["ND", "preT2D", "T2D"])].copy()
            s["status"] = pd.Categorical(s.status, ["ND", "preT2D", "T2D"])
            m = fit_mixed("y ~ status + C(glucose) + C(site)", s)
            ci = m.conf_int()
            for lev in ("preT2D", "T2D"):
                term = f"status[T.{lev}]"
                st_rows.append({"cell_type": ct, "endpoint": ep, "contrast": f"{lev} vs ND", "estimate_sd": m.params[term],
                                "lo": ci.loc[term, 0], "hi": ci.loc[term, 1], "p": m.pvalues[term],
                                "n_donors": s.donor.nunique()})
        ranks = pd.DataFrame(td.y, columns=td.endpoints).groupby(td.group).rank(pct=True)
        ranks = ranks - ranks.groupby(td.group).transform("mean")
        cm = ranks.corr()
        for a in td.endpoints:
            for b in td.endpoints:
                cor_rows.append({"cell_type": ct, "a": a, "b": b, "within_r": cm.loc[a, b]})
    files = {"icc.tsv": pd.DataFrame(icc_rows), "status_effects.tsv": pd.DataFrame(st_rows),
             "within_endpoint_correlation.tsv": pd.DataFrame(cor_rows), "cohort_summary.tsv": pd.concat(coh)}
    for fn, df in files.items():
        df.to_csv(out / fn, sep="\t", index=False)
    print(files["icc.tsv"].round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], [out / f for f in files], seed=cfg["seed"])


if __name__ == "__main__":
    main()
