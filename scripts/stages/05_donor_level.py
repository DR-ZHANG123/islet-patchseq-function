"""Stage 05：供体层面模型的留出供体交叉验证（开发供体）。

比较两种得到“供体平均表型”的方式：
(a) 供体伪 bulk 模型：同供体同类型细胞的伪 bulk 表达 → 供体平均表型（岭回归，供体层面内层交叉验证）；
(b) 单细胞模型取平均：Stage 03 中单细胞岭回归与供体参照模型的逐细胞预测，按与终点相同的方式标准化后取供体均值。
终点在葡萄糖浓度内标准化后按供体取均值（≥ min_cells 个细胞）。
输出 results/05_donor_level/donor_cv_<type>.tsv（逐重复 Spearman）与 donor_predictions_<type>.tsv。
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

STAGE = "05_donor_level"


def donor_cv(td, cfg, k, pb, alphas):
    dcfg = cfg["donor_level"]
    y = td.y[:, k]
    tgt = dl.donor_targets(y, td.donor, td.glucose, np.ones(len(y), bool), dcfg["min_cells"])
    donors = tgt.index.values
    rows = []
    for rep in range(dcfg["repeats"]):
        rng = np.random.default_rng(cfg["seed"] + 1000 * rep)
        fold = rng.permutation(len(donors)) % dcfg["folds"]
        pred = np.zeros(len(donors))
        for f in range(dcfg["folds"]):
            trd, ted = donors[fold != f], donors[fold == f]
            # 终点标准化只用训练供体的细胞
            m_tr = np.isin(td.donor, trd)
            t_tr = dl.donor_targets(y, td.donor, td.glucose, m_tr, dcfg["min_cells"]).reindex(trd).dropna()
            genes = dl.select_genes(pb.loc[t_tr.index], dcfg["n_genes"], dcfg["min_logcpm"])
            xtr = dl.cohort_z(pb.loc[t_tr.index], genes)
            xte = dl.cohort_z(pb.loc[ted], genes, ref=pb.loc[t_tr.index])
            path, a = dl.fit_donor_model(xtr, t_tr.y.values, np.sqrt(t_tr.n.values), alphas, dcfg["inner_folds"],
                                         cfg["seed"] + f)
            pred[fold == f] = path.predict(xte, a)
        rows.append(pd.DataFrame({"donor": donors, "repeat": rep, "y": tgt.y.values, "n": tgt.n.values,
                                  "pred": pred}))
    return pd.concat(rows)


def cellwise_to_donor(td, cfg, ct, ep, models):
    """Stage 03 逐细胞预测 → 与终点相同的葡萄糖内标准化 → 供体均值。"""
    p = pd.read_parquet(ROOT / "data" / "interim" / "03_benchmark" / f"pred_{ct}_donors.parquet",
                        filters=[("endpoint", "==", ep)])
    p = p[p.model.isin(models)].join(td.cells.set_index("cell_id")[["donor", "glucose"]], on="cell_id")
    out = []
    for (m, rep), s in p.groupby(["model", "repeat"]):
        for col in ("y", "pred"):
            s[col + "_z"] = s.groupby("glucose")[col].transform(lambda v: (v - v.mean()) / (v.std() + 1e-8))
        agg = s.groupby("donor").agg(y=("y_z", "mean"), pred=("pred_z", "mean"), n=("y", "size"))
        agg = agg[agg.n >= cfg["donor_level"]["min_cells"]]
        out.append({"model": m, "repeat": rep, "spearman": spearmanr(agg.y, agg.pred).statistic,
                    "n_donors": len(agg)})
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out_dir = ROOT / "results" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = cfg["model"]["alphas"]
    outputs = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        pb = dl.pseudobulk(td.counts.toarray(), td.donor)
        pb.columns = td.symbols
        pb = pb.T.groupby(level=0).sum().T if pb.columns.duplicated().any() else pb
        summ, preds = [], []
        for k, ep in enumerate(td.endpoints):
            res = donor_cv(td, cfg, k, pb, alphas)
            res.insert(0, "endpoint", ep)
            preds.append(res)
            for rep, s in res.groupby("repeat"):
                summ.append({"endpoint": ep, "model": "donor_pseudobulk", "repeat": rep,
                             "spearman": spearmanr(s.y, s.pred).statistic, "n_donors": len(s)})
            cw = cellwise_to_donor(td, cfg, ct, ep, ["ridge", "cell_branch", "context_model", "gradient_boosting"])
            cw.insert(0, "endpoint", ep)
            summ += cw.to_dict("records")
        summ = pd.DataFrame(summ)
        summ.insert(0, "cell_type", ct)
        p1 = out_dir / f"donor_cv_{ct}.tsv"
        p2 = out_dir / f"donor_predictions_{ct}.tsv"
        summ.to_csv(p1, sep="\t", index=False)
        pd.concat(preds).to_csv(p2, sep="\t", index=False)
        outputs += [p1, p2]
        print(ct)
        print(summ.groupby(["endpoint", "model"]).spearman.mean().unstack().round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
