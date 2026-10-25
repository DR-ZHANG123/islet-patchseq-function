"""Stage 03c：汇总开发供体交叉验证的预测（Stage 03、03b），计算三个层面的指标、按供体重抽样的置信区间，
以及最终模型与各对照之间的配对差值。

- 每个细胞在 5 次重复中各有一个留出预测，取平均后计算点估计；另报告逐重复指标的均值与标准差。
- 置信区间：按供体有放回重抽样（n_boot 次）；配对差值在同一组重抽样供体上计算，P 值为双侧重抽样 P 值。
- 按细胞随机分折（Stage 03 --split cells）的结果一并汇总，用于比较两种评价方式。
输出 results/03c_summary/{metrics_cv.tsv, metrics_by_repeat.tsv, paired_differences.tsv}。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from metrics import donor_means, three_level  # noqa: E402

STAGE = "03c_summary"
REFERENCE = "context_model"


def load_preds(ct: str, split: str) -> pd.DataFrame:
    parts = [ROOT / "data/interim/03_benchmark" / f"pred_{ct}_{split}.parquet"]
    if split == "donors":
        parts.append(ROOT / "data/interim/03b_benchmark_nets" / f"pred_{ct}_donors.parquet")
    p = pd.concat([pd.read_parquet(f) for f in parts if f.exists()], ignore_index=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    return p.join(cells[["donor", "glucose"]], on="cell_id")


def _spear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return pearsonr(rankdata(a), rankdata(b))[0]


def _pear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return pearsonr(a, b)[0]


def prepare(s: pd.DataFrame, min_group: int) -> dict:
    """为快速重抽样预先按供体整理：细胞层面数组、比较组内去均值数组、葡萄糖校正的供体均值。"""
    s = s.assign(group=s.donor + "|" + s.glucose.astype(str))
    s["yc"] = s.y - s.groupby("group").y.transform("mean")
    s["pc"] = s.pred - s.groupby("group").pred.transform("mean")
    dm = donor_means(s.y.values, s.pred.values, s.donor.values, s.glucose.values, min_group)
    by = {d: (g.y.values, g.pred.values, g.yc.values, g.pc.values) for d, g in s.groupby("donor")}
    return {"by": by, "dm": dm}


def stats_from(prep: dict, donors) -> np.ndarray:
    ys, ps, ycs, pcs, dy, dp = [], [], [], [], [], []
    for d in donors:
        y, p, yc, pc = prep["by"][d]
        ys.append(y); ps.append(p); ycs.append(yc); pcs.append(pc)
        if d in prep["dm"].index:
            dy.append(prep["dm"].at[d, "y"]); dp.append(prep["dm"].at[d, "p"])
    y, p, yc, pc = map(np.concatenate, (ys, ps, ycs, pcs))
    return np.array([_spear(y, p), _spear(dy, dp) if len(dy) > 3 else np.nan, _pear(yc, pc)])


def endpoint_job(ct, split, ep, s, cfg):
    rng = np.random.default_rng(cfg["seed"])
    avg = s.groupby(["cell_id", "model", "donor", "glucose"], as_index=False).agg(y=("y", "mean"), pred=("pred", "mean"))
    models = sorted(avg.model.unique())
    preps = {m: prepare(avg[avg.model == m], cfg["cv"]["min_group"]) for m in models}
    donors = np.array(sorted(avg.donor.unique()))
    boots = {m: [] for m in models}
    for _ in range(cfg["summary"]["n_boot"]):
        pick = rng.choice(donors, len(donors), replace=True)
        for m in models:
            boots[m].append(stats_from(preps[m], pick))
    rows, diffs = [], []
    for m in models:
        point = stats_from(preps[m], donors)
        b = np.array(boots[m])
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        rows.append({"cell_type": ct, "split": split, "endpoint": ep, "model": m,
                     "cell_spearman": point[0], "cell_lo": lo[0], "cell_hi": hi[0],
                     "donor_spearman": point[1], "donor_lo": lo[1], "donor_hi": hi[1],
                     "within_pearson": point[2], "within_lo": lo[2], "within_hi": hi[2],
                     "n_cells": int(len(avg[avg.model == m])), "n_donors": len(donors)})
    if REFERENCE in models:
        ref = np.array(boots[REFERENCE])
        pref = stats_from(preps[REFERENCE], donors)
        for m in models:
            if m == REFERENCE:
                continue
            d = ref - np.array(boots[m])
            pt = pref - stats_from(preps[m], donors)
            lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
            pv = [min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean())) for j in range(3)]
            for j, lev in enumerate(("cell", "donor", "within")):
                diffs.append({"cell_type": ct, "split": split, "endpoint": ep, "comparator": m, "level": lev,
                              "difference": pt[j], "lo": lo[j], "hi": hi[j], "p_boot": pv[j]})
    return rows, diffs


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    jobs, by_rep = [], []
    for ct in cfg["dataset"]["cell_types"]:
        for split in ("donors", "cells"):
            f = ROOT / "data/interim/03_benchmark" / f"pred_{ct}_{split}.parquet"
            if not f.exists():
                continue
            p = load_preds(ct, split)
            p["group"] = p.donor + "|" + p.glucose.astype(str)
            for (ep, m, rep), s in p.groupby(["endpoint", "model", "repeat"]):
                by_rep.append({"cell_type": ct, "split": split, "endpoint": ep, "model": m, "repeat": rep,
                               **three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                             cfg["cv"]["min_within"], s.donor.values, s.glucose.values)})
            jobs += [(ct, split, ep, s) for ep, s in p.groupby("endpoint")]
    res = Parallel(n_jobs=cfg["summary"]["jobs"], verbose=2)(delayed(endpoint_job)(ct, sp, ep, s, cfg)
                                                             for ct, sp, ep, s in jobs)
    met = pd.DataFrame([r for rr, _ in res for r in rr])
    dif = pd.DataFrame([d for _, dd in res for d in dd])
    rep = pd.DataFrame(by_rep)
    files = {"metrics_cv.tsv": met, "paired_differences.tsv": dif, "metrics_by_repeat.tsv": rep}
    for fn, df in files.items():
        df.to_csv(out / fn, sep="\t", index=False)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], [out / f for f in files], seed=cfg["seed"])
    print(met[met.split == "donors"].pivot_table(index=["cell_type", "endpoint"], columns="model",
                                                 values="within_pearson").round(3).to_string())


if __name__ == "__main__":
    main()
