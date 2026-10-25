"""Stage 06c：供体层面的外部检验（HPAP 供体）。

开发供体的供体汇总表达谱（同类型细胞计数求和，log1p CPM）训练供体层面模型，分别输入：
(i)  HPAP 供体 Patch-seq 细胞的汇总表达谱（同平台、新供体）；
(ii) 同一批供体在 PanKbase 整合 10x 图谱中的 β/α 细胞汇总表达谱（液滴平台、不同细胞）；
与 HPAP Patch-seq 实测的供体平均表型（葡萄糖内标准化后按供体取均值）比较。
跨队列时基因在各自队列内跨供体标准化；10x 的标准化参照为图谱全部未处理供体。只用两平台共有的基因。
输出 results/06c_external_donor_level/{donor_level_summary.tsv, donor_level_predictions.tsv}。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import donor_level as dl  # noqa: E402
from atlas import load_pankbase_pseudobulk  # noqa: E402

STAGE = "06c_external_donor_level"


def bootstrap_rho(y, p, n_boot, rng):
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.std(y[i]) > 0 and np.std(p[i]) > 0:
            vals.append(spearmanr(y[i], p[i]).statistic)
    return np.percentile(vals, [2.5, 97.5])


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dcfg = cfg["donor_level"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    summ, preds = [], []
    for ct in cfg["dataset"]["cell_types"]:
        dev, ext = load_type(cfg, ct, "dev"), load_type(cfg, ct, "external")
        pb_dev = dl.pseudobulk(dev.counts.toarray(), dev.donor)
        pb_ext = dl.pseudobulk(ext.counts.toarray(), ext.donor)
        pb_dev.columns = pb_ext.columns = dev.symbols
        keep = ~pb_dev.columns.duplicated()
        pb_dev, pb_ext = pb_dev.loc[:, keep], pb_ext.loc[:, keep]
        pb_10x, _ = load_pankbase_pseudobulk(cfg, ct)
        shared = [g for g in pb_dev.columns if g in pb_10x.columns]
        for k, ep in enumerate(dev.endpoints):
            t_dev = dl.donor_targets(dev.y[:, k], dev.donor, dev.glucose, np.ones(len(dev.donor), bool),
                                     dcfg["min_cells"])
            t_ext = dl.donor_targets(ext.y[:, k], ext.donor, ext.glucose, np.ones(len(ext.donor), bool),
                                     dcfg["min_cells"])
            genes = dl.select_genes(pb_dev.loc[t_dev.index, shared], dcfg["n_genes"], dcfg["min_logcpm"])
            path, a = dl.fit_donor_model(dl.cohort_z(pb_dev.loc[t_dev.index], genes), t_dev.y.values,
                                         np.sqrt(t_dev.n.values), cfg["model"]["alphas"], dcfg["inner_folds"],
                                         cfg["seed"])
            on10 = [d for d in t_ext.index if d in pb_10x.index]
            z10 = pd.DataFrame(dl.cohort_z(pb_10x, genes), index=pb_10x.index)
            inputs = {"Patch-seq profile": (t_ext.index, dl.cohort_z(pb_ext.loc[t_ext.index], genes)),
                      "Droplet profile": (on10, z10.loc[on10].values)}
            for name, (idx, x) in inputs.items():
                p = path.predict(x, a)
                y = t_ext.loc[idx].y.values
                lo, hi = bootstrap_rho(y, p, cfg["external"]["n_boot"], rng)
                summ.append({"cell_type": ct, "endpoint": ep, "input": name, "n_donors": len(idx),
                             "spearman": spearmanr(y, p).statistic, "lo": lo, "hi": hi})
                preds.append(pd.DataFrame({"cell_type": ct, "endpoint": ep, "input": name, "donor": list(idx),
                                           "measured": y, "n_cells": t_ext.loc[idx].n.values, "predicted": p}))
    s = pd.DataFrame(summ)
    p1, p2 = out / "donor_level_summary.tsv", out / "donor_level_predictions.tsv"
    s.to_csv(p1, sep="\t", index=False)
    pd.concat(preds).to_csv(p2, sep="\t", index=False)
    print(s.round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], [p1, p2], seed=cfg["seed"])


if __name__ == "__main__":
    main()
