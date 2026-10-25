"""Stage 03h：同一基因集对照与 GEO 原始计数敏感性分析（与 Stage 03 相同的留出供体划分）。

分步运行（--part）：
  geneset     [islet_patchseq] 集成模型与岭回归只用基础模型的 8,244 个共有基因（重新归一化），全部开发细胞
  raw_prep    [islet_patchseq] 有 GEO 计数的开发细胞：读取 GEO 原始计数、基因对齐、逐细胞一致性
  raw_cv      [islet_patchseq] 同一批细胞、同一组基因：还原计数 vs GEO 原始计数上的集成模型
  raw_embed   [islet_patchseq_fm] Geneformer V2-104M 冻结嵌入：GEO 原始计数 vs 还原计数（同一组基因）
  raw_heads   [islet_patchseq] 上述两个嵌入及 Stage 03d 原嵌入在同一子集上的预测头（与 Stage 03e 相同）
  summary     [islet_patchseq] 三个层面的指标、按供体重抽样置信区间与配对差值；写 PROVENANCE.json
供体划分：donor_folds 作用于全部开发细胞（与 Stage 03 相同），子集细胞沿用所属供体的折号。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
import geneset_raw as gr  # noqa: E402

STAGE = "03h_geneset_rawcounts"
INT = ROOT / "data" / "interim" / STAGE
OUT = ROOT / "results" / STAGE
GF = "gf104"


def folds_for(td_full, cfg, rows=None):
    """与 Stage 03 相同的供体折（全部开发细胞上分配），rows 给定时取子集。"""
    for rep in cfg["geneset_raw"]["repeats"]:
        seed = cfg["seed"] + 1000 * rep
        folds = donor_folds(td_full.donor, cfg["cv"]["outer_folds"], seed)
        folds = folds if rows is None else folds[rows]
        for f in range(cfg["cv"]["outer_folds"]):
            yield rep, f, folds, seed + 37 * f


def run_jobs(cfg, tasks):
    res = Parallel(n_jobs=cfg["geneset_raw"]["jobs"], verbose=5)(delayed(gr.ensemble_fold)(*t) for t in tasks)
    return pd.concat(res, ignore_index=True)


def fm_genes(cfg) -> pd.DataFrame:
    return pd.read_csv(ROOT / cfg["geneset_raw"]["fm_gene_file"], sep="\t")


def subset_rows(td, cfg, final: bool = True) -> np.ndarray:
    """有 GEO 计数的开发细胞；final=True 时再去掉 GEO 计数总数低于 raw_min_total 的细胞（raw_prep 写出的名单）。"""
    rows = np.where(td.cells.geo.isin(cfg["download"]["geo_counts"].keys()).values)[0]
    if final:
        keep = set(pd.read_csv(INT / "subset_cells.tsv", sep="\t").cell_id)
        rows = rows[np.isin(td.cells.cell_id.values[rows], list(keep))]
    return rows


# ---------------------------------------------------------------- 各部分
def part_geneset(cfg):
    sym = set(fm_genes(cfg).symbol)
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        cols = np.where(np.isin(td.symbols, list(sym)))[0]
        tdr = gr.restrict(td, cols=cols)
        print(ct, "基础模型基因", len(cols), flush=True)
        pred = run_jobs(cfg, [(tdr, cfg, *j, "_fm_genes") for j in folds_for(td, cfg)])
        pred.to_parquet(INT / f"pred_{ct}_geneset.parquet")


def part_raw_prep(cfg):
    raw_dir, hg = ROOT / cfg["paths"]["raw"] / "geo", ROOT / cfg["foundation"]["hgnc_file"]
    conc, ali_all, dropped = [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        rows = subset_rows(td, cfg, final=False)
        c = td.cells.iloc[rows]
        raw, ali = gr.geo_counts(c, td.symbols, raw_dir, cfg["download"]["geo_counts"], hg)
        ok = np.asarray(raw.sum(1)).ravel() >= cfg["geneset_raw"]["raw_min_total"]
        dropped.append(c.loc[~ok, ["cell_id", "geo"]].assign(cell_type=ct, raw_total=np.asarray(raw.sum(1)).ravel()[~ok]))
        rows, c, raw = rows[ok], c[ok], raw[np.where(ok)[0]]
        keep = np.where(ali.found.values)[0]
        rec = td.counts[rows][:, keep]
        sp.save_npz(INT / f"raw_{ct}.npz", raw[:, keep].tocsr())
        sp.save_npz(INT / f"recovered_{ct}.npz", rec.tocsr())
        np.save(INT / f"genes_{ct}.npy", keep)
        conc.append(gr.concordance(rec.tocsr(), raw[:, keep].tocsr(), c.cell_id.values, c.geo.values).assign(cell_type=ct))
        ali_all.append(ali.assign(cell_type=ct))
        print(ct, "子集细胞", len(rows), "供体", c.donor.nunique(), "共同基因", len(keep), "/", len(ali), flush=True)
    # 基础模型嵌入用：两类细胞合并，8,244 个基因
    g = fm_genes(cfg)
    cells = pd.concat([load_type(cfg, ct, "dev").cells for ct in cfg["dataset"]["cell_types"]], ignore_index=True)
    sub = pd.concat(conc)[["cell_id", "cell_type", "geo"]]
    sub.to_csv(INT / "subset_cells.tsv", sep="\t", index=False)
    pd.concat(dropped).to_csv(OUT / "raw_excluded_cells.tsv", sep="\t", index=False)
    cells = cells[cells.cell_id.isin(sub.cell_id)].reset_index(drop=True)
    raw, ali = gr.geo_counts(cells, g.symbol.values, raw_dir, cfg["download"]["geo_counts"], hg)
    sp.save_npz(INT / "fm_raw.npz", raw)
    cells[["cell_id", "donor", "cell_type", "site", "glucose", "matrix", "geo"]].to_csv(INT / "fm_cells.tsv", sep="\t", index=False)
    ali.assign(cell_type="fm_genes").to_csv(INT / "fm_gene_alignment.tsv", sep="\t", index=False)
    pd.concat(conc).to_csv(OUT / "raw_concordance.tsv", sep="\t", index=False)
    pd.concat(ali_all + [ali.assign(cell_type="fm_genes")]).groupby("cell_type").agg(
        n_genes=("symbol", "size"), n_found_all_series=("found", "sum")).to_csv(OUT / "raw_gene_alignment.tsv", sep="\t")


def load_raw_pair(cfg, ct):
    td = load_type(cfg, ct, "dev")
    rows, keep = subset_rows(td, cfg), np.load(INT / f"genes_{ct}.npy")
    rec = gr.restrict(td, rows=rows, cols=keep)
    raw = gr.restrict(td, rows=rows, cols=keep, counts=sp.load_npz(INT / f"raw_{ct}.npz"))
    return td, rows, rec, raw


def part_raw_cv(cfg):
    for ct in cfg["dataset"]["cell_types"]:
        td, rows, rec, raw = load_raw_pair(cfg, ct)
        tasks = [(rec, cfg, *j, "_recovered") for j in folds_for(td, cfg, rows)]
        tasks += [(raw, cfg, *j, "_raw") for j in folds_for(td, cfg, rows)]
        run_jobs(cfg, tasks).to_parquet(INT / f"pred_{ct}_raw_ensemble.parquet")


def part_raw_embed(cfg):
    """[islet_patchseq_fm] 同一组基因（三个 series 都能对齐的基础模型基因）上分别嵌入 GEO 原始计数与还原计数。"""
    from fm_data import CellSet, load_cellset
    from fm_models import embed_geneformer
    gcfg = cfg["foundation"]["geneformer"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["geneset_raw"]["embed_gpu"])
    cells = pd.read_csv(INT / "fm_cells.tsv", sep="\t")
    ali = pd.read_csv(INT / "fm_gene_alignment.tsv", sep="\t")
    found = ali.found.values
    full = load_cellset(cfg)
    pos = pd.Series(np.arange(len(full.cells)), index=full.cells.cell_id.values)[cells.cell_id].values
    if list(full.genes.symbol) != list(ali.symbol):
        raise RuntimeError("基础模型基因顺序与 Stage 03d 不一致")
    genes = full.genes[found].reset_index(drop=True)
    inputs = {"raw": sp.load_npz(INT / "fm_raw.npz")[:, found], "recovered": full.counts[pos][:, found]}
    var = cfg["geneset_raw"]["raw_embedding"]
    mdir = ROOT / cfg["foundation"]["model_dir"] / "geneformer"
    for name, x in inputs.items():
        out = INT / f"emb_{name}.npz"
        if out.exists():
            continue
        set_global_seed(cfg["seed"])
        cs = CellSet(cells[["cell_id", "donor", "cell_type", "site", "glucose", "matrix"]], x.tocsr().astype(np.float32), genes)
        emb, info = embed_geneformer(cs, mdir, gcfg["variants"][var], gcfg, INT / "_work" / name, cfg["seed"])
        np.savez(out, emb=emb, cell_id=cells.cell_id.values.astype(str))
        (INT / f"emb_{name}.json").write_text(json.dumps({"input": name, "n_genes": int(found.sum()), **info}, indent=2))
        print(name, emb.shape, info, flush=True)


def _heads_job(td, cfg, rep, fold, folds, seed, arr, prefix):
    import sota_heads
    import baselines as bl
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    out, _ = sota_heads.rep_heads(arr, td.y, td.endpoints, td.donor, td.group, td.glucose, tr_all, te_all, cfg, seed,
                                  prefix, lambda tr, s: bl.inner_splits(td.donor[tr], cfg["cv"]["inner_folds"], s))
    return pd.concat([pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": m, "repeat": rep,
                                    "fold": fold, "y": td.y[te, td.endpoints.index(ep)], "pred": p})
                      for ep, (te, preds) in out.items() for m, p in preds.items()])


def part_raw_heads(cfg):
    orig = np.load(ROOT / cfg["foundation"]["out_dir"] / f"{cfg['geneset_raw']['raw_embedding']}.npz", allow_pickle=True)
    embs = {"recovered": np.load(INT / "emb_recovered.npz"), "raw": np.load(INT / "emb_raw.npz"), "original": orig}
    for ct in cfg["dataset"]["cell_types"]:
        td, rows, rec, _ = load_raw_pair(cfg, ct)
        tasks = []
        for name, z in embs.items():
            pos = pd.Series(np.arange(len(z["cell_id"])), index=z["cell_id"].astype(str))
            arr = np.asarray(z["emb"], np.float64)[pos[rec.cells.cell_id.astype(str)].values]
            tasks += [(rec, cfg, *j, arr, f"{GF}_{name}") for j in folds_for(td, cfg, rows)]
        res = Parallel(n_jobs=cfg["geneset_raw"]["jobs"], verbose=5)(delayed(_heads_job)(*t) for t in tasks)
        pd.concat(res, ignore_index=True).to_parquet(INT / f"pred_{ct}_raw_heads.parquet")


def part_summary(cfg):
    import sota_summary as ss
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    reps = cfg["geneset_raw"]["repeats"]
    jobs, per_rep, inputs = [], [], [ROOT / "results/01_dataset/cells.tsv"]
    for ct in cfg["dataset"]["cell_types"]:
        f03 = ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet"
        fem = sorted((ROOT / "data/interim/03e_sota").glob(f"pred_{ct}_emb_*.parquet"))
        fgs, fre, fhe = (INT / f"pred_{ct}_{p}.parquet" for p in ("geneset", "raw_ensemble", "raw_heads"))
        inputs += [f03, fgs, fre, fhe] + fem
        p03 = pd.read_parquet(f03)
        p03 = p03[p03.model.isin(["cell_branch", "ridge", "context_model", "linear_cell_branch"]) & p03["repeat"].isin(reps)]
        emb = pd.concat([pd.read_parquet(f) for f in fem], ignore_index=True)
        emb = emb[emb.model.str.endswith(("_ridge", "_cell_branch", "_context", "_gbm")) & emb["repeat"].isin(reps)]
        sets = {"geneset": pd.concat([p03, pd.read_parquet(fgs), emb], ignore_index=True)}
        sub = pd.concat([pd.read_parquet(fre), pd.read_parquet(fhe)], ignore_index=True)
        ids = set(sub.cell_id)
        sets["raw_subset"] = pd.concat([sub, p03[p03.cell_id.isin(ids)].assign(model=lambda d: d.model + "_all_cells_fit")],
                                       ignore_index=True)
        for lab, s in sets.items():
            per_rep.append(ss.per_repeat(s, cells, cfg, ct).assign(set=lab))
            s = s.join(cells[["donor", "glucose"]], on="cell_id")
            refs = ["cell_branch", "cell_branch_fm_genes"] if lab == "geneset" else ["cell_branch_recovered", f"{GF}_recovered_cell_branch"]
            jobs += [(ct, ep, g, refs, lab) for ep, g in s.groupby("endpoint")]
    res = Parallel(n_jobs=min(36, len(jobs)), verbose=2)(
        delayed(ss.boot_endpoint)(ct, ep, s, refs, cfg["summary"]["n_boot"], cfg["seed"], cfg["cv"]["min_group"], lab)
        for ct, ep, s, refs, lab in jobs)
    tabs = {"metrics_cv.tsv": pd.DataFrame([r for rr, _ in res for r in rr]),
            "paired_differences.tsv": pd.DataFrame([d for _, dd in res for d in dd]),
            "metrics_by_repeat.tsv": pd.concat(per_rep, ignore_index=True)}
    outputs = []
    for fn, df in tabs.items():
        df.to_csv(OUT / fn, sep="\t", index=False)
        outputs.append(OUT / fn)
    outputs += [OUT / "raw_concordance.tsv", OUT / "raw_gene_alignment.tsv", OUT / "raw_excluded_cells.tsv", INT / "emb_raw.npz", INT / "emb_recovered.npz"]
    inputs += sorted((ROOT / cfg["paths"]["raw"] / "geo").glob("*"))
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"])
    m = tabs["metrics_cv.tsv"]
    print(m.pivot_table(index=["set", "cell_type", "endpoint"], columns="model", values="within_pearson").round(3).T.to_string())


PARTS = {"geneset": part_geneset, "raw_prep": part_raw_prep, "raw_cv": part_raw_cv, "raw_embed": part_raw_embed,
         "raw_heads": part_raw_heads, "summary": part_summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(PARTS))
    a = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    INT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS[a.part](cfg)


if __name__ == "__main__":
    main()
