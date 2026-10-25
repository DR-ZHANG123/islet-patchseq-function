"""Stage 03g：Camunas-Soler 等 2020 原始“预测基因集”（PS）+ kNN 方法的折内忠实重建（实现见 scripts/lib/ps_faithful.py）。

- 每个外层训练折内：基因过滤 → 按细胞重抽样的基因–参数 Spearman z 值 → PS 基因（|z|>2 的参数分属 ≥ 2 个功能组，
  中位表达 > 0）→ 一个基因集用于全部 9 个终点的 kNN（k=5，Spearman 相似度，等权）。
- 划分：与 Stage 03 完全相同的留出供体折（10 折 × 5 次）与按细胞随机折（--split cells 的同种子划分）。
- 两个变体：ps_faithful_nd（原文：训练只用非糖尿病供体的细胞）与 ps_faithful_all（全部训练细胞）。
- 汇总：与 Stage 03c/03e 相同的点估计、按供体重抽样置信区间与配对差值；对照为 Stage 03 同一划分下的
  cell_branch（集成细胞分支）、context_model、ridge 与 knn_predictive_set（结局特异近邻对照）。
用法：python scripts/stages/03g_ps_faithful.py --part cv|summary
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
from data import donor_folds, load_type  # noqa: E402

STAGE = "03g_ps_faithful"
INT = ROOT / "data" / "interim" / STAGE
OUT = ROOT / "results" / STAGE
COMPARATORS = ["cell_branch", "context_model", "ridge", "knn_predictive_set"]


def fold_list(td, cfg, split):
    """与 Stage 03 相同：重复种子 = seed + 1000·rep；折种子 = 重复种子 + 37·fold。"""
    out = []
    for rep in cfg["ps_faithful"]["repeats"]:
        seed = cfg["seed"] + 1000 * rep
        if split == "donors":
            folds = donor_folds(td.donor, cfg["cv"]["outer_folds"], seed)
        else:
            folds = np.random.default_rng(seed).integers(0, cfg["cv"]["outer_folds"], len(td.donor))
        out += [(rep, f, folds, seed + 37 * f) for f in range(cfg["cv"]["outer_folds"])]
    return out


def raw_endpoints(td) -> dict:
    """原始单位的终点（cells.tsv），另加相关分析用的 Exocytosis norm Ca2+ = 总胞吐 / Ca2+ 内流。"""
    raw = {ep: td.cells[ep].values.astype(float) for ep in td.endpoints}
    ca = raw["calcium_entry"]
    raw["exocytosis_per_calcium"] = np.where(ca > 0, raw["exocytosis"] / np.where(ca > 0, ca, 1), np.nan)
    return raw


def run_fold(td, cfg, split, variant, status, rep, fold, folds, seed, device):
    import torch
    torch.set_num_threads(1)
    import ps_faithful as ps
    pcfg = cfg["ps_faithful"]
    panel = {v[0]: v[1] for v in cfg["dataset"]["endpoint_panel"].values()}
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    tr = tr_all if status is None else tr_all[np.isin(td.cells.status.values[tr_all], status)]
    counts = td.counts.toarray().astype(np.float64)
    keep = ps.prefilter_genes(counts[tr], pcfg["min_cells"], pcfg["min_counts"])
    x = ps.log2cpm(counts[:, keep])
    expr = np.where(x[tr].mean(0) > pcfg["min_log2cpm_mean"])[0]
    raw = raw_endpoints(td)
    used = sorted({p for g in pcfg["groups"].values() for p in g})
    yc = ps.corr_targets({p: raw[p] for p in used}, tr, tuple(pcfg["quantiles"]))
    bs = ps.BootSpearman(x[np.ix_(tr, expr)], device)
    z = {p: bs.zscores(yc[p], pcfg["n_boot"], seed * 31 + j) for j, p in enumerate(used)}
    del bs
    torch.cuda.empty_cache()
    sel = ps.select_ps(z, pcfg["groups"], pcfg["z_th"], pcfg["n_blocks"])
    med_ok = np.median(x[np.ix_(tr, expr)], axis=0) > pcfg["min_log2cpm_median"]
    genes = expr[sel & med_ok]
    order = ps.knn_neighbours(x[np.ix_(tr, genes)], x[np.ix_(te_all, genes)], pcfg["k"])
    rows = []
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        tgt, sign = ps.knn_targets(raw[ep], tr, pcfg["target_clip_q"])
        p = ps.to_model_scale(ps.knn_predict(order, tgt, pcfg["k"]), sign, panel[ep])
        ok = ~np.isnan(y[te_all])
        rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te_all][ok], "endpoint": ep, "model": variant,
                                  "repeat": rep, "fold": fold, "y": y[te_all][ok], "pred": p[ok]}))
    info = {"model": variant, "split": split, "repeat": rep, "fold": fold, "n_train_cells": len(tr),
            "n_genes_prefilter": len(keep), "n_genes_tested": len(expr),
            "n_genes_two_groups": int(sel.sum()), "n_ps_genes": len(genes)}
    info |= {f"n_z2_{p}": int((np.abs(np.nan_to_num(z[p])) > pcfg["z_th"]).sum()) for p in used}
    gl = pd.DataFrame({"model": variant, "split": split, "repeat": rep, "fold": fold,
                       "symbol": td.symbols[keep][genes]})
    return pd.concat(rows), info, gl


def part_cv(cfg):
    pcfg = cfg["ps_faithful"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, pcfg["gpus"]))
    n_dev = len(pcfg["gpus"])
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        for split in pcfg["splits"]:
            path = INT / f"pred_{ct}_{split}.parquet"
            if path.exists():
                print(f"[done] {ct} {split}")
                continue
            tasks = [(v, st, *j) for v, st in pcfg["variants"].items() for j in fold_list(td, cfg, split)]
            res = Parallel(n_jobs=n_dev * pcfg["workers_per_gpu"], verbose=5)(
                delayed(run_fold)(td, cfg, split, v, st, r, f, fo, s, f"cuda:{i % n_dev}")
                for i, (v, st, r, f, fo, s) in enumerate(tasks))
            pd.DataFrame([r[1] for r in res]).assign(cell_type=ct).to_csv(OUT / f"geneset_sizes_{ct}_{split}.tsv", sep="\t", index=False)
            pd.concat([r[2] for r in res]).to_parquet(INT / f"genes_{ct}_{split}.parquet")
            pd.concat([r[0] for r in res], ignore_index=True).to_parquet(path)
            print(ct, split, pd.DataFrame([r[1] for r in res]).groupby("model").n_ps_genes.describe().round(1))


def part_summary(cfg):
    import sota_summary as ss                       # 与 03c/03e 相同的统计实现
    from sota_summary import per_repeat
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    refs = cfg["ps_faithful"]["references"]
    outputs, inputs, jobs, reps = [], [ROOT / "results/01_dataset/cells.tsv"], [], []
    for ct in cfg["dataset"]["cell_types"]:
        for split in cfg["ps_faithful"]["splits"]:
            f_new = INT / f"pred_{ct}_{split}.parquet"
            f_ref = ROOT / "data/interim/03_benchmark" / f"pred_{ct}_{split}.parquet"
            new = pd.read_parquet(f_new)
            ref = pd.read_parquet(f_ref)
            ref = ref[ref.model.isin(COMPARATORS) & ref["repeat"].isin(cfg["ps_faithful"]["repeats"])]
            inputs += [f_new, f_ref]
            n_ref = ref[ref.model == "cell_branch"].groupby("endpoint").size()
            n_new = new.groupby(["model", "endpoint"]).size().unstack("model")
            if n_new.reindex(n_ref.index).ne(n_ref, axis=0).any().any():
                raise RuntimeError(f"{ct} {split}: 忠实重建 PS 的留出预测覆盖与 Stage 03 不一致")
            allp = pd.concat([ref, new], ignore_index=True)
            r = per_repeat(allp, cells, cfg, ct)
            reps.append(r.assign(split=split))
            allp = allp.join(cells[["donor", "glucose", "status"]], on="cell_id")
            for ep, s in allp.groupby("endpoint"):
                jobs.append((ct, ep, s.drop(columns="status"), f"{split}|all_test_cells"))
                jobs.append((ct, ep, s[s.status == "ND"].drop(columns="status"), f"{split}|ND_test_cells"))
    res = Parallel(n_jobs=min(36, len(jobs)), verbose=2)(
        delayed(ss.boot_endpoint)(ct, ep, s, refs + ["cell_branch"], cfg["summary"]["n_boot"], cfg["seed"],
                                  cfg["cv"]["min_group"], lab) for ct, ep, s, lab in jobs)
    tabs = {"metrics_cv.tsv": pd.DataFrame([r for rr, _ in res for r in rr]),
            "paired_differences.tsv": pd.DataFrame([d for _, dd in res for d in dd]),
            "metrics_by_repeat.tsv": pd.concat(reps, ignore_index=True)}
    for t in tabs.values():
        if "set" in t:
            t[["split", "test_cells"]] = t.pop("set").str.split("|", expand=True)
    for fn, df in tabs.items():
        df.to_csv(OUT / fn, sep="\t", index=False)
        outputs.append(OUT / fn)
    outputs += sorted(OUT.glob("geneset_sizes_*.tsv"))
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"],
                     extra={"source_code": "github.com/jcamunas/patchseq functions/predictions.py, "
                                           "functions/helper_functions.py, notebooks/Figure3"})
    m = tabs["metrics_cv.tsv"]
    print(m[m.test_cells == "all_test_cells"].pivot_table(index=["cell_type", "split", "endpoint"], columns="model",
                                                          values="cell_spearman").round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cv", "summary"])
    a = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    INT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    part_cv(cfg) if a.part == "cv" else part_summary(cfg)


if __name__ == "__main__":
    main()
