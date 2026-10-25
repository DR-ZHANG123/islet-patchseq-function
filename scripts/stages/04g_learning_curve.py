"""Stage 04g：按训练供体数的学习曲线（开发供体，留出供体交叉验证）。

第 d 次抽样使用 Stage 03 第 d 个重复的 10 折供体划分（重复 d 的种子 = seed + 1000·d，折种子 + 37·折号）。
每个外层折把训练供体随机排序一次，比例 f 取前 max(min_train_donors, round(f·n)) 位供体（嵌套子集），
在这些供体的细胞上按 Stage 03 的流程拟合（训练细胞内选高变基因与标准化、内层按供体调参、岭回归 + 梯度提升树），
在同一批留出供体上评价；比例 1.0 即 Stage 03 的完整训练集（与 Stage 04b standard 的同一重复逐位一致）。
输出 results/04g_learning_curve/{metrics_by_draw.tsv, curve.tsv, trend.tsv, train_sizes.tsv, full_check.tsv}。
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
import robust_curve  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402
import robust_light as rl  # noqa: E402

STAGE = "04g_learning_curve"


def donor_subset(td, cfg, ct, draw, fold, folds, frac):
    """返回训练细胞掩码与训练供体数；随机排序只取决于 (seed, 细胞类型, 抽样, 折)，各比例共用（嵌套）。"""
    lc = cfg["learning_curve"]
    train_d = np.array(sorted(np.unique(td.donor[folds != fold])))
    order = np.random.default_rng([cfg["seed"], 9100, cfg["dataset"]["cell_types"].index(ct), draw, fold]
                                  ).permutation(train_d)
    n = min(len(train_d), max(lc["min_train_donors"], int(round(frac * len(train_d)))))
    keep = order[:n]
    return np.isin(td.donor, keep), n


def job(td, cfg, ct, draw, fold):
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, draw)
    lc = cfg["learning_curve"]
    out, sizes = [], []
    for frac in lc["fractions"]:
        mask, n = donor_subset(td, cfg, ct, draw, fold, folds, frac)
        inp = rl.fold_inputs(td, cfg, fold, folds, train_mask=mask)
        p = rl.run_fold_light(td, cfg, fold, seed + 37 * fold, lc["endpoints"], inp,
                              tag={"fraction": frac, "draw": draw})
        out.append(p[p.model.isin(lc["models"])])
        sizes.append({"cell_type": ct, "draw": draw, "fold": fold, "fraction": frac, "n_train_donors": n,
                      "n_train_cells": len(inp[0]), "n_test_cells": len(inp[1])})
        robust_cache.clear()
    o = pd.concat(out, ignore_index=True)
    o[["y", "pred"]] = o[["y", "pred"]].astype(np.float32)
    return o, pd.DataFrame(sizes)


def full_check(pred, ct):
    """比例 1.0 的逐重复指标与 Stage 04b standard 同一重复的比较（应一致）。"""
    obs = pd.read_csv(ROOT / "results/04b_robustness/metrics_by_repeat.tsv", sep="\t")
    obs = obs[(obs.cell_type == ct) & (obs.variant == "standard")]
    rows = []
    for (d, ep, m), s in pred[pred.fraction == 1.0].groupby(["draw", "endpoint", "model"]):
        o = obs[(obs.repeat == d) & (obs.endpoint == ep) & (obs.model == m)]
        if len(o):
            r = s.within_pearson.iloc[0]
            rows.append({"cell_type": ct, "draw": d, "endpoint": ep, "model": m, "within_here": r,
                         "within_04b": o.within_pearson.iloc[0], "abs_diff": abs(r - o.within_pearson.iloc[0])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    lc = cfg["learning_curve"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    mets, curves, trends, pers, sizes, checks, outputs = [], [], [], [], [], [], []
    for ct in lc["cell_types"]:
        td = load_type(cfg, ct, "dev")
        res = Parallel(n_jobs=args.jobs or lc["jobs"], verbose=5)(
            delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_d{d}_f{f}.pkl", job, td, cfg, ct, d, f)
            for d in range(lc["draws"]) for f in range(cfg["cv"]["outer_folds"]))
        pred = re_.attach_meta(pd.concat([r[0] for r in res], ignore_index=True), cells)
        sz = pd.concat([r[1] for r in res], ignore_index=True)
        sizes.append(sz)
        pp = tmp / f"pred_{ct}.parquet"
        pred.to_parquet(pp)
        outputs.append(pp)
        met = re_.per_repeat_metrics(pred, ["fraction", "draw", "endpoint", "model"], cfg)
        met.insert(0, "cell_type", ct)
        mets.append(met)
        checks.append(full_check(met, ct))
        n_train = sz.groupby("fraction").n_train_donors.mean().to_dict()
        groups = list(pred.groupby(["endpoint", "model"]))
        parts = Parallel(n_jobs=min(len(groups), args.jobs or lc["jobs"]))(
            delayed(robust_curve.boot_curve)(s, n_train, lc["n_boot"], cfg["seed"], cfg["cv"]["min_group"],
                                             {"cell_type": ct, "endpoint": ep, "model": m})
            for (ep, m), s in groups)
        curves += [p[0] for p in parts]
        trends += [p[1] for p in parts]
        pers += [p[2] for p in parts]
    sz = pd.concat(sizes, ignore_index=True)
    tabs = {"metrics_by_draw.tsv": pd.concat(mets, ignore_index=True),
            "curve.tsv": pd.concat(curves, ignore_index=True), "trend.tsv": pd.concat(trends, ignore_index=True),
            "train_sizes.tsv": sz.groupby(["cell_type", "fraction"]).agg(
                n_train_donors_mean=("n_train_donors", "mean"), n_train_donors_min=("n_train_donors", "min"),
                n_train_donors_max=("n_train_donors", "max"), n_train_cells_mean=("n_train_cells", "mean"),
                n_folds=("fold", "size")).reset_index(),
            "full_check.tsv": pd.concat(checks, ignore_index=True)}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outputs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/04b_robustness/metrics_by_repeat.tsv"],
                     outputs, seed=cfg["seed"])
    print("比例 1.0 与 Stage 04b 的最大差：", tabs["full_check.tsv"].abs_diff.max())
    t = tabs["trend.tsv"]
    print(t[t.level == "within"].round(3).to_string())
    c = tabs["curve.tsv"]
    print(c[c.level == "within"].pivot_table(index=["cell_type", "endpoint", "model"], columns="fraction",
                                             values="estimate").round(3).to_string())


if __name__ == "__main__":
    main()
