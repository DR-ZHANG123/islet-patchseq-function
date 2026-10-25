"""Stage 04：供体参照的必要性检验（开发供体，留出供体交叉验证）。

对每个外层折拟合双分支模型后，只改变测试供体的供体参照再预测：
- full：用该供体全部同类型细胞；
- k 个细胞：每个测试供体随机抽 k 个细胞计算参照（k ∈ context_sizes，细胞不足时用全部），重复 draws 次取平均；
- mismatched：测试供体之间随机互换参照（错配供体），检验模型是否依赖“正确的”供体背景。
输出 results/04_context_tests/context_tests_<type>.tsv。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from features import context_mean, fit_scaler, glucose_onehot  # noqa: E402
from context_model import ContextModel  # noqa: E402
from metrics import three_level  # noqa: E402

STAGE = "04_context_tests"


def subset_context(td, te, k, rng):
    """测试供体只保留 k 个随机细胞作为参照（细胞不足时用全部）。"""
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    for d in np.unique(td.donor[te]):
        idx = cells_of[d]
        if len(idx) > k:
            cells_of[d] = rng.choice(idx, k, replace=False)
    return cells_of


def mismatched_context(td, te, rng):
    """测试供体之间随机互换参照细胞集合（错配供体）。"""
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    ds = np.unique(td.donor[te])
    if len(ds) >= 2:
        perm = np.roll(rng.permutation(ds), 1)
        orig = {d: cells_of[d] for d in ds}
        for d, src in zip(ds, perm):
            cells_of[d] = orig[src]
    return cells_of


def mu_from(z, donor, cells_of):
    mu = np.zeros_like(z)
    for d, idx in cells_of.items():
        mu[donor == d] = z[idx].mean(0)
    return mu


def run_fold(td, cfg, rep, fold, folds, seed):
    rng = np.random.default_rng(seed)
    tc = cfg["context_tests"]
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    sc = fit_scaler(td.xlog[tr_all], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(td.xlog)
    mu = context_mean(z, td.donor)
    g1 = glucose_onehot(td.glucose)
    full = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    rows = []
    for k, ep in enumerate(td.endpoints):
        if ep not in tc["endpoints"]:
            continue
        y = td.y[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        mdl = ContextModel(mcfg, True, cfg["model"]["n_boost_features"], seed + k).fit(
            z, mu, g1, y, td.donor, td.group, tr, full)
        variants = {"full": [full]}
        for kk in tc["context_sizes"]:
            variants[f"k{kk}"] = [subset_context(td, te_all, kk, rng) for _ in range(tc["draws"])]
        variants["mismatched"] = [mismatched_context(td, te_all, rng) for _ in range(tc["draws"])]
        for name, cos in variants.items():
            preds = [mdl.predict(z, mu_from(z, td.donor, co), g1, td.donor, te, co) for co in cos]
            for comp in ("donor_branch", "context_model"):
                p = np.mean([q[comp] for q in preds], 0)
                rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "context": name,
                                          "component": comp, "repeat": rep, "y": y[te], "pred": p}))
    return pd.concat(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out_dir = ROOT / "results" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        jobs = []
        for rep in range(cfg["context_tests"]["repeats"]):
            seed = cfg["seed"] + 1000 * rep
            folds = donor_folds(td.donor, cfg["cv"]["outer_folds"], seed)
            jobs += [(rep, f, folds, seed + 37 * f) for f in range(cfg["cv"]["outer_folds"])]
        res = Parallel(n_jobs=60, verbose=5)(delayed(run_fold)(td, cfg, r, f, fo, s) for r, f, fo, s in jobs)
        pred = pd.concat(res, ignore_index=True)
        meta = td.cells.set_index("cell_id")
        pred = pred.join(meta[["donor", "glucose"]], on="cell_id")
        pred["group"] = pred.donor + "|" + pred.glucose.astype(str)
        met = []
        for (ep, cx, comp, rep), s in pred.groupby(["endpoint", "context", "component", "repeat"]):
            met.append({"cell_type": ct, "endpoint": ep, "context": cx, "component": comp, "repeat": rep,
                        **three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                      cfg["cv"]["min_within"], s.donor.values, s.glucose.values)})
        met = pd.DataFrame(met)
        path = out_dir / f"context_tests_{ct}.tsv"
        met.to_csv(path, sep="\t", index=False)
        outputs.append(path)
        print(ct, met[met.component == "context_model"].groupby(["endpoint", "context"]).donor_spearman.mean()
              .unstack().round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
