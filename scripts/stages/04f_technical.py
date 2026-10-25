"""Stage 04f：技术协变量对照（开发供体，与 Stage 03 前 5 个重复相同的留出供体折）。

(a) 只用质控特征的模型（文库大小、检出基因数、线粒体/核糖体/MALAT1 比例、他类激素环境 RNA 评分等）；
(b) 只用批次特征的模型（GEO 系列、来源中心）；(a)+(b) 合并；三者均含葡萄糖，岭回归与梯度提升树各一；
(c) 表达特征先对质控特征回归取残差（只在训练细胞上估计）后再拟合双分支模型；
(d) 控制质控特征后，双分支模型预测与实测值的供体内偏相关；各质控特征与终点的供体内相关。
与 Stage 03 的双分支模型做按供体重抽样的配对比较。输出 results/04f_technical/*.tsv。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402
import robust_qc as rq  # noqa: E402

STAGE = "04f_technical"
BASE = ["context_model", "cell_branch", "ridge", "gradient_boosting"]
PAIRS = [("context_model", f"{s}_{m}") for s in ("qc_only", "batch_only", "qc_batch") for m in ("ridge", "boost")] + \
        [(f"{m}_qc_partialled", m) for m in ("context_model", "cell_branch", "ridge")]


def job(td, cfg, sets, qmat, rep, fold):
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, rep)
    fseed = seed + 37 * fold
    tc = cfg["robustness"]["technical"]
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    rows = []
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        for name, f in sets.items():
            preds = {"ridge": rq.ridge_small(f, y, tr, te, td.donor, tc["alphas"], cfg["cv"]["inner_folds"], fseed + k),
                     "boost": rq.boosting_small(f, y, tr, te, fseed + k)}
            for m, p in preds.items():
                rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep,
                                          "model": f"{name}_{m}", "repeat": rep, "fold": fold, "y": y[te], "pred": p}))
    p, _ = rc.run_fold(td, cfg, fold, folds, fseed, td.endpoints, qc=qmat, tag={"repeat": rep})
    robust_cache.clear()
    p = p[p.model.isin(["context_model", "cell_branch", "ridge"])].copy()
    p["model"] = p.model + "_qc_partialled"
    rows.append(p)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None, help="并行进程数（缺省取 robustness.jobs）")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    tc = cfg["robustness"]["technical"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    reps, mets, difs, parts, qcw, outs = [], [], [], [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        q = rq.qc_features(td, cfg, ct)
        q.insert(0, "cell_id", td.cells.cell_id.values)
        qpath = tmp / f"qc_features_{ct}.tsv"
        q.to_csv(qpath, sep="\t", index=False)
        outs.append(qpath)
        qv = q.drop(columns="cell_id")
        sets = rq.feature_sets(td, qv, cfg)
        jobs = [(r, f) for r in range(tc["variant_repeats"]) for f in range(cfg["cv"]["outer_folds"])]
        res = Parallel(n_jobs=args.jobs or cfg["robustness"]["jobs"], verbose=5)(
            delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_r{r}_f{f}.pkl", job, td, cfg, sets, qv.values, r, f)
            for r, f in jobs)
        new = pd.concat(res, ignore_index=True)
        s03 = pd.read_parquet(ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet")
        s03 = s03[s03.model.isin(BASE) & (s03.repeat < tc["variant_repeats"])]
        pred = re_.attach_meta(pd.concat([new, s03[new.columns]], ignore_index=True), cells)
        pp = tmp / f"pred_{ct}.parquet"
        pred.to_parquet(pp)
        outs.append(pp)
        r = re_.per_repeat_metrics(pred, ["endpoint", "model", "repeat"], cfg)
        reps.append(r.assign(cell_type=ct))
        avg = re_.average_repeats(pred, ["endpoint", "model"])
        qmap = q.set_index("cell_id")
        rng = np.random.default_rng(cfg["seed"])
        for ep, s in avg.groupby("endpoint"):
            m, d = re_.boot_compare(s, PAIRS, cfg["summary"]["n_boot"], cfg["seed"], cfg["cv"]["min_group"],
                                    {"cell_type": ct, "endpoint": ep})
            mets.append(m); difs.append(d)
            for mdl in ("context_model", "cell_branch"):
                t = s[s.model == mdl]
                qm = qmap.loc[t.cell_id].values
                parts.append({"cell_type": ct, "endpoint": ep, "model": mdl, "n_cells": len(t)}
                             | rq.partial_within(t.y.values, t.pred.values, qm, t.group.values, t.donor.values,
                                                 cfg["summary"]["n_boot"], rng))
        for k, ep in enumerate(td.endpoints):
            qcw.append({"cell_type": ct, "endpoint": ep} | rq.qc_endpoint_within(td.y[:, k], qv, td.group))
    tabs = {"metrics_by_repeat.tsv": pd.concat(reps), "metrics_boot.tsv": pd.concat(mets),
            "paired_differences.tsv": pd.concat(difs), "partial_within_qc.tsv": pd.DataFrame(parts),
            "qc_endpoint_within.tsv": pd.DataFrame(qcw)}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outs, seed=cfg["seed"])
    met = tabs["metrics_boot.tsv"]
    print(met.pivot_table(index=["cell_type", "endpoint"], columns=["level", "model"], values="estimate")
          .round(2).to_string())
    print(tabs["partial_within_qc.tsv"].round(3).to_string())


if __name__ == "__main__":
    main()
