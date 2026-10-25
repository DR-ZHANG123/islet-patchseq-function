"""Stage 06：外部供体检验（HPAP 供体，开发阶段未参与任何选择）。

A. 单细胞层面：全部开发供体上训练（超参数仍由开发供体内部的内层交叉验证选择），预测 HPAP 供体的 Patch-seq 细胞；
   报告细胞、供体均值、供体内三个层面，并按供体重抽样给出 95% 置信区间。
供体层面的外部检验见 Stage 06c。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
from features import context_mean, fit_scaler, glucose_onehot  # noqa: E402
from context_model import ContextModel  # noqa: E402
import baselines as bl  # noqa: E402
from metrics import three_level  # noqa: E402

STAGE = "06_external_hpap"


def stack(dev, ext):
    """把开发与外部细胞拼接为一个数组集合（基因相同）。"""
    n_dev = len(dev.donor)
    cat = lambda a, b: np.concatenate([a, b])  # noqa: E731
    return (np.vstack([dev.xlog, ext.xlog]), np.vstack([dev.y, ext.y]), cat(dev.donor, ext.donor),
            cat(dev.group, ext.group), cat(dev.glucose, ext.glucose), np.arange(n_dev),
            np.arange(n_dev, n_dev + len(ext.donor)))


def endpoint_job(cfg, xlog, ymat, donor, group, glucose, dev_idx, ext_idx, k, seed):
    sc = fit_scaler(xlog[dev_idx], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(xlog)
    mu = context_mean(z, donor)
    g1 = glucose_onehot(glucose)
    cells_of = {d: np.where(donor == d)[0] for d in np.unique(donor)}
    y = ymat[:, k]
    tr = dev_idx[~np.isnan(y[dev_idx])]
    te = ext_idx[~np.isnan(y[ext_idx])]
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    nb = cfg["model"]["n_boost_features"]
    mdl = ContextModel(mcfg, True, nb, seed).fit(z, mu, g1, y, donor, group, tr, cells_of)
    preds = mdl.predict(z, mu, g1, donor, te, cells_of)
    f = np.c_[z, g1]
    inner = bl.inner_splits(donor[tr], cfg["cv"]["inner_folds"], seed)
    preds["knn_predictive_set"] = bl.knn_predictive_set(xlog, y, tr, te, 200, 5, cfg["features"]["min_mean"])
    preds["elastic_net"] = bl.elastic_net(f, y, tr, te, inner)
    preds["pls"] = bl.pls(f, y, tr, te, inner)
    hv = np.r_[np.arange(nb), z.shape[1] + np.arange(3)]
    preds["gradient_boosting"] = bl.gradient_boosting(f[:, hv], y, tr, te, seed)
    return te, preds, {"alpha_cell": mdl.alpha_c, "within_lambda": mdl.lam, "alpha_donor": mdl.alpha_d,
                       "gate": mdl.gate}


def nets_job(cfg, xlog, ymat, donor, group, glucose, dev_idx, ext_idx, device):
    import nets
    sc = fit_scaler(xlog[dev_idx], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(xlog).astype(np.float32)
    g1 = glucose_onehot(glucose).astype(np.float32)
    out = {}
    for name, (mode, lam) in {"mlp_cell": ("cell", 0.0), "mlp_concat": ("concat", 0.0), "mlp_two": ("two", 0.0),
                              "mlp_two_within": ("two", 1.0)}.items():
        out[name] = np.mean([nets.train_predict(z, g1, ymat, donor, group, dev_idx, ext_idx, mode, lam,
                                                cfg["seed"] + s, device) for s in range(cfg["nets"]["seeds"])], 0)
    return out


def bootstrap_ci(y, p, grp, donors, glucose, n_boot, seed):
    """按供体有放回重抽样；重复抽到的供体视为不同供体（比较组与供体标签加后缀）。"""
    rng = np.random.default_rng(seed)
    ud = np.unique(donors)
    pos = {d: np.where(donors == d)[0] for d in ud}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ud, len(ud), replace=True)
        idx = np.concatenate([pos[d] for d in pick])
        tag = np.concatenate([np.full(len(pos[d]), j) for j, d in enumerate(pick)]).astype(str)
        r = three_level(y[idx], p[idx], np.char.add(grp[idx].astype(str), tag), 3, 5,
                        np.char.add(donors[idx].astype(str), tag), glucose[idx])
        vals.append([r["cell_spearman"], r["donor_spearman"], r["within_pearson"]])
    return np.nanpercentile(np.array(vals), [2.5, 97.5], axis=0)


def cell_level(cfg, ct, device):
    dev, ext = load_type(cfg, ct, "dev"), load_type(cfg, ct, "external")
    xlog, ymat, donor, group, glucose, di, ei = stack(dev, ext)
    jobs = Parallel(n_jobs=len(dev.endpoints))(delayed(endpoint_job)(cfg, xlog, ymat, donor, group, glucose, di, ei,
                                                                      k, cfg["seed"] + k)
                                               for k in range(len(dev.endpoints)))
    netp = nets_job(cfg, xlog, ymat, donor, group, glucose, di, ei, device)
    rows, met, params = [], [], []
    for k, (te, preds, par) in enumerate(jobs):
        ep = dev.endpoints[k]
        params.append({"cell_type": ct, "endpoint": ep, **par})
        ok = ~np.isnan(ymat[ei, k])
        for name, p in netp.items():
            preds[name] = p[ok, k]
        for name, p in preds.items():
            y = ymat[te, k]
            r = three_level(y, p, group[te], cfg["cv"]["min_group"], cfg["cv"]["min_within"], donor[te], glucose[te])
            lo, hi = (bootstrap_ci(y, p, group[te], donor[te], glucose[te], cfg["external"]["n_boot"], cfg["seed"])
                      if name in cfg["external"]["ci_models"] else (np.full(3, np.nan), np.full(3, np.nan)))
            met.append({"cell_type": ct, "endpoint": ep, "model": name, **r, "cell_lo": lo[0], "cell_hi": hi[0],
                        "donor_lo": lo[1], "donor_hi": hi[1], "within_lo": lo[2], "within_hi": hi[2]})
            rows.append(pd.DataFrame({"cell_id": np.r_[dev.cells.cell_id, ext.cells.cell_id][te], "endpoint": ep,
                                      "model": name, "y": y, "pred": p}))
    return pd.concat(rows), pd.DataFrame(met), pd.DataFrame(params)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out_dir = ROOT / "results" / STAGE
    int_dir = ROOT / "data" / "interim" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ct in cfg["dataset"]["cell_types"]:
        pred, met, par = cell_level(cfg, ct, cfg["external"]["device"])
        p0 = int_dir / f"pred_{ct}.parquet"
        pred.to_parquet(p0)
        p1, p2 = out_dir / f"metrics_{ct}.tsv", out_dir / f"hyperparameters_{ct}.tsv"
        met.to_csv(p1, sep="\t", index=False)
        par.to_csv(p2, sep="\t", index=False)
        outputs += [p0, p1, p2]
        print(ct, met.pivot_table(index="endpoint", columns="model", values="within_pearson").round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
