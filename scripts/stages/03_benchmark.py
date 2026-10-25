"""Stage 03：开发供体上的留出供体重复交叉验证（线性双分支模型及其组件、经典对照）。

每个重复把开发供体随机分成 10 折；每折内只用训练细胞选高变基因、拟合标准化，模型超参数在训练供体内部
按供体分组的内层交叉验证选择。测试供体的供体参照只用其自身细胞的表达（不用功能标签）。
--split cells 时改为按细胞随机分折（同一供体的细胞可同时出现在训练与测试中），用于比较两种评价方式。
输出：data/interim/03_benchmark/pred_<type>_<split>.parquet（逐细胞预测），results/03_benchmark/metrics_*.tsv。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from features import context_mean, fit_scaler, glucose_onehot  # noqa: E402
import baselines as bl  # noqa: E402
from context_model import ContextModel  # noqa: E402
from metrics import three_level  # noqa: E402

STAGE = "03_benchmark"


def run_fold(td, cfg, rep, fold, folds, seed):
    tr_all = np.where(folds != fold)[0]
    te_all = np.where(folds == fold)[0]
    sc = fit_scaler(td.xlog[tr_all], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(td.xlog)
    mu = context_mean(z, td.donor)
    g1 = glucose_onehot(td.glucose)
    f = np.c_[z, g1]
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    nb = cfg["model"]["n_boost_features"]
    rows, params = [], []
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        if len(te) == 0:
            continue
        mdl = ContextModel(mcfg, True, nb, seed + k).fit(z, mu, g1, y, td.donor, td.group, tr, cells_of)
        preds = mdl.predict(z, mu, g1, td.donor, te, cells_of)
        inner = bl.inner_splits(td.donor[tr], cfg["cv"]["inner_folds"], seed + k)
        preds["knn_predictive_set"] = bl.knn_predictive_set(td.xlog, y, tr, te, 200, 5, cfg["features"]["min_mean"])
        preds["elastic_net"] = bl.elastic_net(f, y, tr, te, inner)
        preds["pls"] = bl.pls(f, y, tr, te, inner)
        hv = np.r_[np.arange(nb), z.shape[1] + np.arange(3)]
        preds["gradient_boosting"] = bl.gradient_boosting(f[:, hv], y, tr, te, seed + k)
        for m, p in preds.items():
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": m,
                                      "repeat": rep, "fold": fold, "y": y[te], "pred": p}))
        params.append({"endpoint": ep, "repeat": rep, "fold": fold, "alpha_cell": mdl.alpha_c,
                       "within_lambda": mdl.lam, "alpha_donor": mdl.alpha_d, "gate": mdl.gate,
                       "gate_linear": mdl.gate_lin, "alpha_ridge": mdl.alpha_p})
    return pd.concat(rows), pd.DataFrame(params)


def summarize(pred: pd.DataFrame, td, cfg) -> pd.DataFrame:
    meta = td.cells.set_index("cell_id")
    pred = pred.join(meta[["donor", "glucose"]], on="cell_id")
    pred["group"] = pred.donor + "|" + pred.glucose.astype(str)
    out = []
    for (ep, m, rep), s in pred.groupby(["endpoint", "model", "repeat"]):
        r = three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"], cfg["cv"]["min_within"])
        out.append({"endpoint": ep, "model": m, "repeat": rep, **r})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["donors", "cells"], default="donors")
    ap.add_argument("--jobs", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out_dir = ROOT / "results" / STAGE
    int_dir = ROOT / "data" / "interim" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        jobs = []
        for rep in range(cfg["cv"]["repeats"]):
            seed = cfg["seed"] + 1000 * rep
            if args.split == "donors":
                folds = donor_folds(td.donor, cfg["cv"]["outer_folds"], seed)
            else:
                folds = np.random.default_rng(seed).integers(0, cfg["cv"]["outer_folds"], len(td.donor))
            jobs += [(rep, f, folds, seed + 37 * f) for f in range(cfg["cv"]["outer_folds"])]
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_fold)(td, cfg, r, f, fo, s) for r, f, fo, s in jobs)
        pred = pd.concat([r[0] for r in res], ignore_index=True)
        params = pd.concat([r[1] for r in res], ignore_index=True)
        pred_path = int_dir / f"pred_{ct}_{args.split}.parquet"
        pred.to_parquet(pred_path)
        met = summarize(pred, td, cfg)
        met.insert(0, "cell_type", ct)
        met_path = out_dir / f"metrics_{ct}_{args.split}.tsv"
        met.to_csv(met_path, sep="\t", index=False)
        params.insert(0, "cell_type", ct)
        par_path = out_dir / f"hyperparameters_{ct}_{args.split}.tsv"
        params.to_csv(par_path, sep="\t", index=False)
        outputs += [pred_path, met_path, par_path]
        show = met.groupby(["endpoint", "model"])[["cell_spearman", "donor_spearman", "within_pearson"]].mean()
        print(ct, args.split)
        print(show.unstack("model")["within_pearson"].round(3).to_string())
    write_provenance(STAGE if args.split == "donors" else STAGE + "_cellsplit", [ROOT / "results/01_dataset/cells.tsv"],
                     outputs, seed=cfg["seed"], extra={"split": args.split})


if __name__ == "__main__":
    main()
