"""Stage 03e 外部供体（HPAP）检验：开发阶段全部设定固定之后，每个新增方法只在外部供体上评价一次。

与 Stage 06 相同：全部开发供体上训练（超参数仍由开发供体内部的内层交叉验证选择），预测 HPAP 供体的 Patch-seq 细胞；
三个层面指标与按供体重抽样的 95% 置信区间直接用 06_external_hpap.py 的 stack / bootstrap_ci。
另在 HPAP 供体上计算与 Stage 06 参照模型（context_model、ridge）的配对差值（同一组重抽样供体）。
scvi_external 部分在 islet_patchseq_scvi 环境中运行：全部开发细胞上训练 scVI，嵌入开发与 HPAP 细胞。
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from data import load_type
from features import fit_scaler, glucose_onehot
from metrics import three_level
from repro import ROOT, write_provenance

STAGE = "03e_sota"
INT = ROOT / "data" / "interim" / STAGE
OUT = ROOT / "results" / STAGE
NET_VARIANTS = {"mlp_cell": ("cell", 0.0), "mlp_concat": ("concat", 0.0), "mlp_two": ("two", 0.0),
                "mlp_two_within": ("two", 1.0)}


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/stages" / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stacked(cfg, ct):
    dev, ext = load_type(cfg, ct, "dev"), load_type(cfg, ct, "external")
    s06 = _load("s06", "06_external_hpap.py")
    xlog, ymat, donor, group, glucose, di, ei = s06.stack(dev, ext)
    cells = pd.concat([dev.cells, ext.cells], ignore_index=True)
    return dev, ext, cells, xlog, ymat, donor, group, glucose, di, ei


def scvi_external(cfg):
    import sota_scvi
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["sota"]["gpus"][0])
    (INT / "scvi").mkdir(parents=True, exist_ok=True)
    for ct in cfg["dataset"]["cell_types"]:
        dev, ext, cells, xlog, *_rest, di, ei = _stacked(cfg, ct)
        counts = sp.vstack([dev.counts, ext.counts]).tocsr()
        genes = fit_scaler(xlog[di], cfg["features"]["n_hvg"], cfg["features"]["min_mean"]).genes
        lat, info = sota_scvi.fit_latent(counts, genes, cells[cfg["sota"]["scvi"]["batch_key"]].values, di,
                                         cfg["sota"]["scvi"], cfg["seed"], 0)
        np.savez_compressed(INT / "scvi" / f"{ct}_external.npz", lat=lat, cell_id=cells.cell_id.values.astype(str),
                            **info)
        print(ct, "scVI external", info, flush=True)


def _inner(donor, cfg):
    import baselines as bl
    return lambda tr, s: bl.inner_splits(donor[tr], cfg["cv"]["inner_folds"], s)


def _gbm_job(cfg, f, y, tr, te, inner, seed):
    import sota_heads
    return sota_heads.tuned_gbm(f, y, tr, te, inner, cfg["sota"]["gbm_grid"], seed)


def _net_job(cfg, z, g1, ymat, donor, group, di, ei, name, device, inner):
    import sota_nets
    mode, lam = NET_VARIANTS[name]
    return sota_nets.tuned_predict(z, g1, ymat, donor, group, di, ei, mode, lam, inner, cfg["sota"]["mlp_grid"],
                                   cfg["nets"]["seeds"], cfg["seed"], device)


def _rep_job(cfg, rep, ymat, endpoints, donor, group, glucose, di, ei, prefix):
    import baselines as bl
    import sota_heads
    return sota_heads.rep_heads(rep, ymat, endpoints, donor, group, glucose, di, ei, cfg, cfg["seed"], prefix,
                                lambda tr, s: bl.inner_splits(donor[tr], cfg["cv"]["inner_folds"], s))


def predict_external(cfg, ct, models):
    """返回逐细胞外部预测（新增方法）与超参数。"""
    import os
    import sota_srrr
    dev, ext, cells, xlog, ymat, donor, group, glucose, di, ei = _stacked(cfg, ct)
    eps = dev.endpoints
    z = fit_scaler(xlog[di], cfg["features"]["n_hvg"], cfg["features"]["min_mean"]).transform(xlog)
    g1 = glucose_onehot(glucose)
    inner = _inner(donor, cfg)
    ids = cells.cell_id.values
    preds, params = {}, []                       # preds[(ep, model)] = (idx, p)
    # sRRR（9 终点联合）
    p, spar = sota_srrr.select_and_fit(z, ymat, glucose, di, ei, inner(di, cfg["seed"]), cfg["sota"]["srrr"])
    params.append({"endpoint": "all", "model": "srrr", **spar})
    for k, ep in enumerate(eps):
        ok = ~np.isnan(ymat[ei, k])
        preds[(ep, "srrr")] = (ei[ok], p[ok, k])
    # 调参梯度提升树（逐终点；种子 = seed + k，同 Stage 06）
    nb = cfg["model"]["n_boost_features"]
    f = np.c_[z[:, :nb], g1]
    tasks = []
    for k in range(len(eps)):
        y = ymat[:, k]
        tr, te = di[~np.isnan(y[di])], ei[~np.isnan(y[ei])]
        tasks.append((k, tr, te))
    res = Parallel(n_jobs=len(tasks))(delayed(_gbm_job)(cfg, f, ymat[:, k], tr, te, inner(tr, cfg["seed"] + k),
                                                         cfg["seed"] + k) for k, tr, te in tasks)
    for (k, tr, te), (pg, gpar) in zip(tasks, res):
        preds[(eps[k], "gradient_boosting_tuned")] = (te, pg)
        params.append({"endpoint": eps[k], "model": "gradient_boosting_tuned", **gpar})
    # 调参神经网络（四个变体并行于不同 GPU）
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["sota"]["gpus"]))
    n = len(cfg["sota"]["gpus"])
    res = Parallel(n_jobs=len(NET_VARIANTS))(
        delayed(_net_job)(cfg, z.astype(np.float32), g1.astype(np.float32), ymat, donor, group, di, ei, v,
                          f"cuda:{i % n}", inner(di, cfg["seed"])) for i, v in enumerate(NET_VARIANTS))
    for v, (pn, hp) in zip(NET_VARIANTS, res):
        params.append({"endpoint": "all", "model": f"{v}_tuned", **hp})
        for k, ep in enumerate(eps):
            ok = ~np.isnan(ymat[ei, k])
            preds[(ep, f"{v}_tuned")] = (ei[ok], pn[ok, k])
    # 表示头：scVI（开发细胞上训练）与基础模型嵌入
    reps = {}
    sz = np.load(INT / "scvi" / f"{ct}_external.npz")
    assert (sz["cell_id"] == ids.astype(str)).all()
    reps["scvi"] = sz["lat"]
    s03e = _load("s03e", "03e_sota.py")
    for name in models:
        reps[name] = s03e.load_embedding(cfg, name, type("T", (), {"cells": cells})())
    res = Parallel(n_jobs=len(reps))(delayed(_rep_job)(cfg, r, ymat, eps, donor, group, glucose, di, ei, pre)
                                     for pre, r in reps.items())
    for out, par in res:
        params += [p_ | {"model": p_["representation"]} for p_ in par]
        for ep, (te, pr) in out.items():
            for m, p_ in pr.items():
                preds[(ep, m)] = (te, p_)
    rows = [pd.DataFrame({"cell_id": ids[idx], "endpoint": ep, "model": m, "y": ymat[idx, eps.index(ep)], "pred": p_})
            for (ep, m), (idx, p_) in preds.items()]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(params), cells


def _metric_job(cfg, ct, ep, m, s):
    s06 = _load("s06", "06_external_hpap.py")
    y, p, grp, don, glu = s.y.values, s.pred.values, s.group.values, s.donor.values, s.glucose.values
    r = three_level(y, p, grp, cfg["cv"]["min_group"], cfg["cv"]["min_within"], don, glu)
    lo, hi = s06.bootstrap_ci(y, p, grp, don, glu, cfg["external"]["n_boot"], cfg["seed"])
    return {"cell_type": ct, "endpoint": ep, "model": m, **r, "cell_lo": lo[0], "cell_hi": hi[0],
            "donor_lo": lo[1], "donor_hi": hi[1], "within_lo": lo[2], "within_hi": hi[2]}


def run(cfg, part, models):
    if part == "scvi_external":
        scvi_external(cfg)
        return
    import sota_summary
    if not (OUT / "metrics_cv.tsv").exists():
        raise RuntimeError("开发供体交叉验证汇总（--part summary）尚未完成，外部检验不能运行")
    done = json.loads((OUT / "PROVENANCE.json").read_text()).get("embeddings", [])
    models = done if models is None else models
    outputs, met, diffs = [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        pred, par, cells = predict_external(cfg, ct, models)
        meta = cells.set_index("cell_id")[["donor", "glucose"]]
        pred = pred.join(meta, on="cell_id")
        pred["group"] = pred.donor + "|" + pred.glucose.astype(str)
        f0 = INT / f"pred_external_{ct}.parquet"
        pred.drop(columns=["donor", "glucose", "group"]).to_parquet(f0)
        f1 = OUT / f"hyperparameters_external_{ct}.tsv"
        par.insert(0, "cell_type", ct)
        par.to_csv(f1, sep="\t", index=False)
        met += Parallel(n_jobs=cfg["sota"]["jobs"], verbose=2)(
            delayed(_metric_job)(cfg, ct, ep, m, s) for (ep, m), s in pred.groupby(["endpoint", "model"]))
        # 与 Stage 06 参照模型在 HPAP 供体上的配对差值
        ref = pd.read_parquet(ROOT / "data/interim/06_external_hpap" / f"pred_{ct}.parquet")
        ref = ref[ref.model.isin(cfg["sota"]["references"])].join(meta, on="cell_id")
        allp = pd.concat([ref, pred.drop(columns="group")], ignore_index=True)
        res = Parallel(n_jobs=len(allp.endpoint.unique()))(
            delayed(sota_summary.boot_endpoint)(ct, ep, s, cfg["sota"]["references"], cfg["external"]["n_boot"],
                                                cfg["seed"], cfg["cv"]["min_group"], "HPAP")
            for ep, s in allp.groupby("endpoint"))
        diffs += [d for _, dd in res for d in dd]
        outputs += [f0, f1]
    tables = {"metrics_external.tsv": pd.DataFrame(met), "paired_differences_external.tsv": pd.DataFrame(diffs)}
    for fn, df in tables.items():
        df.to_csv(OUT / fn, sep="\t", index=False)
        outputs.append(OUT / fn)
    prov = json.loads((OUT / "PROVENANCE.json").read_text())
    prev = {k: v for k, v in prov.get("outputs", {}).items()}
    p = write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"],
                         extra={k: v for k, v in prov.items() if k in ("parts_summarized", "embeddings")}
                         | {"external_models": ["srrr", "gradient_boosting_tuned"] + [f"{v}_tuned" for v in NET_VARIANTS]
                            + ["scvi"] + list(models)})
    rec = json.loads(p.read_text())
    rec["outputs"] = prev | rec["outputs"]
    rec["inputs"] = prov.get("inputs", {}) | rec["inputs"]
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(tables["metrics_external.tsv"].pivot_table(index=["cell_type", "endpoint"], columns="model",
                                                     values="within_pearson").round(3).T.to_string())
