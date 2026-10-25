"""Stage 03e：新增对照方法，使用与 Stage 03 完全相同的留出供体重复交叉验证（10 折 × 5 次，同一折种子与供体分配），
因此每个细胞的留出预测可与 Stage 03/03b 的方法逐细胞配对比较。

方法（细节见 scripts/lib/sota_*.py）：
  srrr                     稀疏降秩回归（Kobak 等 2021），9 个终点联合，秩/l1 比例/λ 由内层供体交叉验证选择
  gradient_boosting_tuned  梯度提升树，深度/学习率/迭代次数由内层供体交叉验证选择（输入同 Stage 03）
  mlp_*_tuned              Stage 03b 的四个网络变体，宽度/权重衰减/dropout 由内层供体交叉验证选择
  scvi_*                   折内训练的 scVI 潜变量（批次 = GEO series）上的岭回归、梯度提升树与 ± 供体参照的双分支模型
  <基础模型>_*             冻结嵌入（Stage 03d）上的同一组预测头
分步运行（--part）：
  scvi_dev       [islet_patchseq_scvi 环境] 每折训练 scVI，潜变量写入 data/interim/03e_sota/scvi/
  hvg            sRRR 与调参梯度提升树
  nets           调参神经网络（GPU）
  scvi_heads     scVI 潜变量上的预测头
  emb            基础模型嵌入上的预测头（--models 指定，缺省为 config 中已生成嵌入且尚未运行者）
  summary        三个层面的指标、按供体重抽样置信区间、与参照模型的配对差值；写 PROVENANCE.json
  scvi_external  [islet_patchseq_scvi 环境] 全部开发细胞上训练 scVI，嵌入 HPAP 细胞
  external       HPAP 外部供体：全部方法在开发供体上按已固定的设定训练，只评价一次
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
from repro import ROOT, load_config, set_global_seed  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from features import fit_scaler, glucose_onehot  # noqa: E402

STAGE = "03e_sota"
INT = ROOT / "data" / "interim" / STAGE
OUT = ROOT / "results" / STAGE
NET_VARIANTS = {"mlp_cell": ("cell", 0.0), "mlp_concat": ("concat", 0.0), "mlp_two": ("two", 0.0),
                "mlp_two_within": ("two", 1.0)}


def fold_jobs(td, cfg):
    """与 Stage 03 相同：重复 rep 的种子 = seed + 1000·rep，供体分折 donor_folds，折种子 = 重复种子 + 37·fold。"""
    for rep in range(cfg["cv"]["repeats"]):
        seed = cfg["seed"] + 1000 * rep
        folds = donor_folds(td.donor, cfg["cv"]["outer_folds"], seed)
        for f in range(cfg["cv"]["outer_folds"]):
            yield rep, f, folds, seed + 37 * f


def inner_fn(donor, cfg):
    import baselines as bl
    return lambda tr, s: bl.inner_splits(donor[tr], cfg["cv"]["inner_folds"], s)


def frame(td, te, ep, model, rep, fold, y, p):
    return pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": model,
                         "repeat": rep, "fold": fold, "y": y, "pred": p})


def multi_rows(td, te, pred, model, rep, fold):
    rows = []
    for k, ep in enumerate(td.endpoints):
        ok = ~np.isnan(td.y[te, k])
        rows.append(frame(td, te[ok], ep, model, rep, fold, td.y[te, k][ok], pred[ok, k]))
    return rows


# ---------------- 各部分的单折任务 ----------------
def job_hvg(td, cfg, rep, fold, folds, seed):
    import sota_heads
    import sota_srrr
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    z = fit_scaler(td.xlog[tr_all], cfg["features"]["n_hvg"], cfg["features"]["min_mean"]).transform(td.xlog)
    g1 = glucose_onehot(td.glucose)
    inner = inner_fn(td.donor, cfg)
    p, spar = sota_srrr.select_and_fit(z, td.y, td.glucose, tr_all, te_all, inner(tr_all, seed), cfg["sota"]["srrr"])
    rows = multi_rows(td, te_all, p, "srrr", rep, fold)
    params = [{"endpoint": "all", "repeat": rep, "fold": fold, **spar}]
    nb = cfg["model"]["n_boost_features"]
    f = np.c_[z[:, :nb], g1]                         # 与 Stage 03 gradient_boosting 相同的输入（前 nb 个高变基因 + 葡萄糖）
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        if len(te) == 0:
            continue
        pg, gpar = sota_heads.tuned_gbm(f, y, tr, te, inner(tr, seed + k), cfg["sota"]["gbm_grid"], seed + k)
        rows.append(frame(td, te, ep, "gradient_boosting_tuned", rep, fold, y[te], pg))
        params.append({"endpoint": ep, "repeat": rep, "fold": fold, **gpar})
    return pd.concat(rows), pd.DataFrame(params)


def job_nets(td, cfg, rep, fold, folds, seed, name, device):
    import sota_nets
    tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
    z = fit_scaler(td.xlog[tr], cfg["features"]["n_hvg"], cfg["features"]["min_mean"]).transform(td.xlog)
    g1 = glucose_onehot(td.glucose).astype(np.float32)
    mode, lam = NET_VARIANTS[name]
    p, hp = sota_nets.tuned_predict(z.astype(np.float32), g1, td.y, td.donor, td.group, tr, te, mode, lam,
                                    inner_fn(td.donor, cfg)(tr, seed), cfg["sota"]["mlp_grid"], cfg["nets"]["seeds"],
                                    seed, device)
    rows = multi_rows(td, te, p, f"{name}_tuned", rep, fold)
    return pd.concat(rows), pd.DataFrame([{"endpoint": "all", "model": f"{name}_tuned", "repeat": rep, "fold": fold,
                                           **hp}])


def job_rep(td, cfg, rep, fold, folds, seed, rep_arr, prefix):
    import sota_heads
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    out, params = sota_heads.rep_heads(rep_arr, td.y, td.endpoints, td.donor, td.group, td.glucose, tr_all, te_all,
                                       cfg, seed, prefix, inner_fn(td.donor, cfg))
    rows = [frame(td, te, ep, m, rep, fold, td.y[te, td.endpoints.index(ep)], p)
            for ep, (te, preds) in out.items() for m, p in preds.items()]
    for r in params:
        r.update(repeat=rep, fold=fold)
    return pd.concat(rows), pd.DataFrame(params)


def job_scvi(td, cfg, rep, fold, folds, seed, device):
    import sota_scvi
    path = INT / "scvi" / f"{td.cells.cell_type.iloc[0]}_r{rep}_f{fold}.npz"
    if path.exists():
        return path
    tr = np.where(folds != fold)[0]
    genes = fit_scaler(td.xlog[tr], cfg["features"]["n_hvg"], cfg["features"]["min_mean"]).genes
    lat, info = sota_scvi.fit_latent(td.counts, genes, td.cells[cfg["sota"]["scvi"]["batch_key"]].values, tr,
                                     cfg["sota"]["scvi"], seed, device)
    np.savez_compressed(path, lat=lat, cell_id=td.cells.cell_id.values.astype(str), **info)
    return path


# ---------------- 运行与保存 ----------------
def save_part(ct, part, res):
    pred = pd.concat([r[0] for r in res], ignore_index=True)
    par = pd.concat([r[1] for r in res], ignore_index=True)
    par.insert(0, "cell_type", ct)
    pred.to_parquet(INT / f"pred_{ct}_{part}.parquet")
    par.to_csv(OUT / f"hyperparameters_{ct}_{part}.tsv", sep="\t", index=False)
    print(ct, part, pred.groupby("model").size().to_dict(), flush=True)


def gpu_env(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["sota"]["gpus"]))
    return len(cfg["sota"]["gpus"])


def load_embedding(cfg, name, td):
    z = np.load(ROOT / cfg["foundation"]["out_dir"] / f"{name}.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["cell_id"])), index=z["cell_id"].astype(str))
    ids = td.cells.cell_id.astype(str)
    miss = ~ids.isin(pos.index)
    if miss.any():
        raise ValueError(f"{name}: {int(miss.sum())} 个细胞没有嵌入")
    return np.asarray(z["emb"], dtype=np.float64)[pos[ids].values]


def run_part(cfg, part, models):
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        jobs = list(fold_jobs(td, cfg))
        if part == "hvg":
            res = Parallel(n_jobs=cfg["sota"]["jobs"], verbose=5)(delayed(job_hvg)(td, cfg, *j) for j in jobs)
            save_part(ct, "hvg", res)
        elif part == "nets":
            n = gpu_env(cfg)
            tasks = [(j, v) for v in NET_VARIANTS for j in jobs]
            res = Parallel(n_jobs=n * cfg["sota"]["workers_per_gpu"], verbose=5)(
                delayed(job_nets)(td, cfg, *j, v, f"cuda:{i % n}") for i, (j, v) in enumerate(tasks))
            save_part(ct, "nets", res)
        elif part == "scvi_dev":
            n = gpu_env(cfg)
            (INT / "scvi").mkdir(parents=True, exist_ok=True)
            Parallel(n_jobs=2 * n, verbose=5)(delayed(job_scvi)(td, cfg, *j, i % n) for i, j in enumerate(jobs))
        elif part == "scvi_heads":
            lat = {(r, f): np.load(INT / "scvi" / f"{ct}_r{r}_f{f}.npz")["lat"] for r, f, _, _ in jobs}
            res = Parallel(n_jobs=cfg["sota"]["jobs"], verbose=5)(
                delayed(job_rep)(td, cfg, r, f, fo, s, lat[(r, f)], "scvi") for r, f, fo, s in jobs)
            save_part(ct, "scvi", res)
        elif part == "emb":
            for name in models:
                rep_arr = load_embedding(cfg, name, td)
                res = Parallel(n_jobs=cfg["sota"]["jobs"], verbose=5)(
                    delayed(job_rep)(td, cfg, *j, rep_arr, name) for j in jobs)
                save_part(ct, f"emb_{name}", res)


def pending_embeddings(cfg):
    """config 中已生成嵌入文件、且两个细胞类型的预测都尚未写出的基础模型。"""
    d = ROOT / cfg["foundation"]["out_dir"]
    return [m for m in cfg["sota"]["embeddings"] if (d / f"{m}.npz").exists()
            and not all((INT / f"pred_{ct}_emb_{m}.parquet").exists() for ct in cfg["dataset"]["cell_types"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["scvi_dev", "hvg", "nets", "scvi_heads", "emb", "summary",
                                                       "scvi_external", "external"])
    ap.add_argument("--models", nargs="*", default=None, help="emb/external：基础模型名（缺省见说明）")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    INT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "summary":
        import sota_summary
        sota_summary.run(cfg)
    elif args.part in ("scvi_external", "external"):
        import sota_external
        sota_external.run(cfg, args.part, args.models)
    else:
        models = args.models if args.models is not None else pending_embeddings(cfg)
        run_part(cfg, args.part, models)


if __name__ == "__main__":
    main()
