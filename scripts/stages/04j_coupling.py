"""Stage 04j：转录组编码 Ca²⁺ 触发还是对它的分泌反应？

1. 分解（开发供体，X 完整的细胞）：胞吐 = Ca²⁺ 可解释部分 b·X + 偶联效率 r（lib/coupling.py）；
   b 只用每个外层折的训练细胞估计。比值形式 log(胞吐 / Ca²⁺ 内流) 为敏感性分析。
2. 留出供体交叉验证（Stage 03 前 5 个重复的折划分；岭回归–梯度提升集成 cell_branch 与 ridge，robust_light），
   三个层面的指标；按留出供体重抽样的置信区间与终点间配对差值（同一组重抽样供体）。
3. 供体层面置换零分布（同 Stage 04c / 04h）：派生终点的供体平均偏移在全部供体间或同一来源中心内互换，
   每次置换重跑完整交叉验证；校准后效应与单侧置换 P。
4. 供体内基因关联与 Hallmark/GO 预排序富集（全部 Patch-seq 供体，方法同 Stage 08）：偶联效率。
5. T2D 比较（全部 Patch-seq 供体）：偶联效率（斜率由 ND 细胞估计）与比值的 T2D 系数，推断同 Stage 09。
输出 results/04j_coupling/*.tsv 与 PROVENANCE.json。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import coupling as cp  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402
import robust_light as rl  # noqa: E402
from t2d_effects import MAIN_SITES, STATUSES  # noqa: E402

STAGE = "04j_coupling"


def eval_endpoints(cc):
    return [cc["ratio_entry"]] + cc["targets"] + cp.derived_names(cc)


def observed_job(td, cc, cfg, complete, rep, fold):
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, rep)
    ymat, bs = cp.fill_derived(td, cc, complete, np.where(folds != fold)[0])
    inp = rl.fold_inputs(td, cfg, fold, folds)
    p = rl.run_fold_light(td, cfg, fold, seed + 37 * fold, eval_endpoints(cc), inp, ymat=ymat, tag={"repeat": rep})
    robust_cache.clear()
    p = p[p.model.isin(cc["models"])]
    p[["y", "pred"]] = p[["y", "pred"]].astype(np.float32)
    sl = pd.DataFrame([{"repeat": rep, "fold": fold, "target": t} | dict(zip(cc["covariates"], b))
                       for t, b in bs.items()])
    return p, sl


def null_ymat(td, cc, cfg, ct, ymat, perm, variant, strata):
    ym = ymat.copy()
    vi = list(cc["null_variants"]).index(variant)
    for ep in null_endpoints(cc, ct):
        k = td.endpoints.index(ep)
        rng = np.random.default_rng([cfg["seed"], 4100, cfg["dataset"]["cell_types"].index(ct), k, perm, vi])
        if cc["null_variants"][variant] is None:
            ym[:, k] = rc.permute_donor_offsets(ymat[:, k], td.donor, td.glucose, rng)
        else:
            ym[:, k] = rl.permute_offsets_strata(ymat[:, k], td.donor, td.glucose, rng, strata, "within")
    return ym


def null_endpoints(cc, ct):
    return cc["null_endpoints"][ct]


def null_job(td, cc, cfg, ct, complete, strata, part, fold, perms):
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, part)
    ymat, _ = cp.fill_derived(td, cc, complete, np.where(folds != fold)[0])
    inp = rl.fold_inputs(td, cfg, fold, folds)
    out = []
    for perm in perms:
        for v in cc["null_variants"]:
            ym = null_ymat(td, cc, cfg, ct, ymat, perm, v, strata)
            p = rl.run_fold_light(td, cfg, fold, seed + 37 * fold, null_endpoints(cc, ct), inp, ymat=ym,
                                  tag={"perm": perm, "null": v})
            out.append(p[p.model.isin(cc["models"])])
            robust_cache.clear(svd=False)
    robust_cache.clear()
    o = pd.concat(out, ignore_index=True)
    o[["y", "pred"]] = o[["y", "pred"]].astype(np.float32)
    return o


def boot_tables(pred, cc, cfg, ct):
    """每个模型：终点作为“模型”列进入 boot_compare，得到各终点的指标与终点间配对差值（同一组重抽样供体）。"""
    avg = re_.average_repeats(pred, ["endpoint", "model"])
    pairs = []
    for t in cc["targets"]:
        s = cp.SHORT[t]
        pairs += [(f"coupling_{s}", f"ca_explained_{s}"), (f"coupling_{s}", t), (f"coupling_{s}", cc["ratio_entry"]),
                  (f"ca_explained_{s}", t), (f"ratio_{s}", t), (f"ratio_{s}", f"positive_{s}")]
    mets, difs = [], []
    for m in cc["models"]:
        a = avg[avg.model == m].assign(model=lambda d: d.endpoint)
        met, dif = re_.boot_compare(a, pairs, cc["n_boot"], cfg["seed"], cfg["cv"]["min_group"],
                                    {"cell_type": ct, "predictor": m})
        mets.append(met.rename(columns={"model": "endpoint"}))
        difs.append(dif.rename(columns={"model_a": "endpoint_a", "model_b": "endpoint_b"}))
    return pd.concat(mets, ignore_index=True), pd.concat(difs, ignore_index=True)


def calibration(nullp, obs_rep, cfg, ct):
    rows = []
    for (v, ep, m), s in nullp.groupby(["null", "endpoint", "model"]):
        nv = np.array([re_.per_repeat_metrics(g.assign(k=0), ["k"], cfg).iloc[0].donor_spearman
                       for _, g in s.groupby("perm")])
        o = obs_rep[(obs_rep.endpoint == ep) & (obs_rep.model == m)].donor_spearman.mean()
        rows.append({"cell_type": ct, "null": v, "endpoint": ep, "model": m, "level": "donor_spearman"}
                    | re_.calibrate(float(o), nv))
    return pd.DataFrame(rows)


def gene_associations(cfg, cc, ct, rng, libs):
    """全部 Patch-seq 供体：偶联效率（斜率用全部 X 完整细胞估计）与基因的供体内关联（Stage 08 方法）。"""
    import gseapy as gp
    spec = importlib.util.spec_from_file_location("s08", ROOT / "scripts/stages/08_programmes.py")
    s08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s08)
    td, complete = cp.augment(load_type(cfg, ct, "all"), cc)
    ymat, _ = cp.fill_derived(td, cc, complete, np.where(complete)[0])
    expressed = (td.xlog > 0).mean(0) >= cc["gene_min_detect"]
    sym, xw = td.symbols[expressed], td.xlog[:, expressed]
    ga, enr = [], []
    for ep in ["coupling_total", "coupling_early", "ratio_total"]:
        y = ymat[:, td.endpoints.index(ep)]
        ok = ~np.isnan(y)
        r, z, p = s08.within_assoc(xw[ok], y[ok], td.group[ok], cc["gene_n_perm"], rng)
        df = pd.DataFrame({"cell_type": ct, "endpoint": ep, "gene": sym, "within_r": r, "within_z": z,
                           "within_p": p, "n_cells": int(ok.sum())})
        df["within_fdr"] = multipletests(df.within_p, method="fdr_bh")[1]
        ga.append(df)
        rnk = df[["gene", "within_z"]].sort_values("within_z", ascending=False)
        for lib, sets in libs.items():
            res = gp.prerank(rnk, gene_sets=sets, min_size=15, max_size=500, permutation_num=cc["gsea_perm"],
                             seed=cfg["seed"], threads=8, outdir=None, verbose=False).res2d
            enr.append(res.assign(cell_type=ct, endpoint=ep, library=lib)[
                ["cell_type", "endpoint", "library", "Term", "NES", "NOM p-val", "FDR q-val", "Lead_genes"]])
    return pd.concat(ga, ignore_index=True), pd.concat(enr, ignore_index=True)


def t2d_tables(cfg, cc, cells):
    rows = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for i, ct in enumerate(cc["cell_types"]):
            df = cp.t2d_frame(cells, ct, cc, MAIN_SITES, STATUSES)
            cols = ["calcium_entry"] + [f"{k}_{cp.SHORT[t]}" for t in cc["targets"] for k in ("coupling", "ratio")]
            for j, col in enumerate(cols):
                rows.append(cp.t2d_test(df, col, cc["t2d_n_boot"], cc["t2d_n_perm"], cfg["seed"] + 500 + 50 * i + j,
                                        cc["t2d_jobs"]).assign(cell_type=ct))
                print(f"[t2d] {ct} {col}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--parts", default="cv,null,genes,t2d")
    args = ap.parse_args()
    parts = args.parts.split(",")
    cfg = load_config()
    set_global_seed(cfg["seed"])
    cc = cfg["coupling"]
    jobs = args.jobs or cc["jobs"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    tabs = {}
    if "cv" in parts or "null" in parts:
        dec, slopes, mets, difs, reps, cals = [], [], [], [], [], []
        for ct in cc["cell_types"]:
            td, complete = cp.augment(load_type(cfg, ct, "dev"), cc)
            dec.append(cp.decomposition_summary(td, cc, complete, ct))
            res = Parallel(n_jobs=jobs, verbose=5)(
                delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_r{r}_f{f}.pkl", observed_job, td, cc, cfg, complete, r, f)
                for r in cc["repeats"] for f in range(cfg["cv"]["outer_folds"]))
            pred = re_.attach_meta(pd.concat([r[0] for r in res], ignore_index=True), cells)
            slopes.append(pd.concat([r[1] for r in res], ignore_index=True).assign(cell_type=ct))
            rep = re_.per_repeat_metrics(pred, ["repeat", "endpoint", "model"], cfg)
            reps.append(rep.assign(cell_type=ct))
            met, dif = boot_tables(pred, cc, cfg, ct)
            mets.append(met)
            difs.append(dif)
            if "null" in parts:
                strata = td.cells.groupby("donor")[["site"]].first()
                nj = [(p, f, pl[i:i + cc["perm_chunk"]]) for p in range(cc["partitions"])
                      for pl in [list(range(p, cc["n_perm"][ct], cc["partitions"]))]
                      for i in range(0, len(pl), cc["perm_chunk"]) for f in range(cfg["cv"]["outer_folds"])]
                nres = Parallel(n_jobs=args.jobs or cc["null_jobs"], verbose=5)(
                    delayed(rc.checkpointed)(tmp / "null" / f"{ct}_p{p}_f{f}_c{pl[0]}.pkl", null_job, td, cc, cfg, ct,
                                             complete, strata, p, f, pl) for p, f, pl in nj)
                nullp = re_.attach_meta(pd.concat(nres, ignore_index=True), cells)
                cals.append(calibration(nullp, rep, cfg, ct))
        tabs |= {"decomposition.tsv": pd.concat(dec), "slopes_by_fold.tsv": pd.concat(slopes),
                 "metrics_by_repeat.tsv": pd.concat(reps), "metrics.tsv": pd.concat(mets),
                 "differences.tsv": pd.concat(difs)}
        if cals:
            tabs["donor_null_calibration.tsv"] = pd.concat(cals)
    if "genes" in parts:
        import gseapy as gp
        rng = np.random.default_rng(cfg["seed"] + 4200)
        libs = {n: gp.read_gmt(str(ROOT / cfg["paths"]["raw"] / "genesets" / f"{n}.gmt")) for n in cc["libraries"]}
        g = [gene_associations(cfg, cc, ct, rng, libs) for ct in cc["cell_types"]]
        tabs["gene_associations.tsv"] = pd.concat([x[0] for x in g])
        tabs["enrichment.tsv"] = pd.concat([x[1] for x in g])
    if "t2d" in parts:
        tabs["t2d_coupling.tsv"] = t2d_tables(cfg, cc, cells)
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
    all_out = sorted(out.glob("*.tsv"))
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], all_out, seed=cfg["seed"],
                     extra={"parts_run_last": parts})
    pd.set_option("display.width", 250)
    if "metrics.tsv" in tabs:
        print(tabs["decomposition.tsv"].round(3).to_string())
        m = tabs["metrics.tsv"]
        print(m[m.predictor == "cell_branch"].pivot_table(index=["cell_type", "endpoint"], columns="level",
                                                           values=["estimate", "lo", "hi"]).round(3).to_string())
        d = tabs["differences.tsv"]
        print(d[d.predictor == "cell_branch"].round(3).to_string())
    if "donor_null_calibration.tsv" in tabs:
        print(tabs["donor_null_calibration.tsv"].round(3).to_string())
    if "t2d_coupling.tsv" in tabs:
        print(tabs["t2d_coupling.tsv"].round(3).to_string())


if __name__ == "__main__":
    main()
