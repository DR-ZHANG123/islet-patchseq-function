"""Stage 08：与电生理表型相关的基因程序（全部 Patch-seq 供体）。

1. 供体内关联：比较组（供体 × 葡萄糖）内对表达与终点分别取秩并去均值，计算每个基因与终点的相关；
   比较组内置换终点（n_perm 次）估计每个基因的零分布标准差，z = r / SD_null，P 值取正态尾概率，BH 校正得 FDR。
2. 供体间关联：供体伪 bulk 与供体平均表型（葡萄糖内标准化）的 Spearman 相关；供体标签置换估计零分布标准差，P 值同上。
3. 两个层面基因关联的一致性（基因间 Spearman 相关），以及各自前列基因的重叠。
4. 预排序富集分析（GO 生物过程、MSigDB Hallmark），分别对两个层面的基因统计量进行。
输出 results/08_programmes/{gene_associations_<type>.tsv, level_concordance.tsv, enrichment_<type>.tsv}。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import donor_level as dl  # noqa: E402

STAGE = "08_programmes"


def centered_ranks(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        r = rankdata(a[m], axis=0) / m.sum()
        out[m] = r - r.mean(0)
    return out


def within_assoc(x, y, groups, n_perm, rng):
    keep = pd.Series(groups).map(pd.Series(groups).value_counts()).values >= 4
    x, y, g = x[keep], y[keep], groups[keep]
    xr, yr = centered_ranks(x, g), centered_ranks(y[:, None], g)[:, 0]
    nx = np.sqrt((xr ** 2).sum(0)) + 1e-12
    r = (xr * yr[:, None]).sum(0) / (nx * np.sqrt((yr ** 2).sum()))
    null = np.zeros((n_perm, x.shape[1]))
    idx_by_g = [np.where(g == k)[0] for k in np.unique(g)]
    for b in range(n_perm):
        yp = yr.copy()
        for ix in idx_by_g:
            yp[ix] = yr[rng.permutation(ix)]
        null[b] = (xr * yp[:, None]).sum(0) / (nx * np.sqrt((yp ** 2).sum()))
    # 置换零分布只用于估计每个基因统计量的零方差；P 值取标准正态尾概率，避免经验 P 值的分辨率下限压住 FDR
    z = r / (null.std(0) + 1e-12)
    return r, z, 2 * norm.sf(np.abs(z))


def donor_assoc(pb, t, n_perm, rng):
    x = pb.loc[t.index].values
    rho = np.array([spearmanr(x[:, j], t.y.values).statistic for j in range(x.shape[1])])
    rx = rankdata(x, axis=0)
    rx = (rx - rx.mean(0)) / (rx.std(0) + 1e-12)
    null = np.zeros((n_perm, x.shape[1]))
    for b in range(n_perm):
        yp = rankdata(rng.permutation(t.y.values))
        null[b] = (rx * ((yp - yp.mean()) / yp.std())[:, None]).mean(0)
    z = rho / (null.std(0) + 1e-12)
    return rho, 2 * norm.sf(np.abs(z))


def main() -> None:
    import gseapy as gp
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    pc = cfg["programmes"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    conc, outputs, enr_all = [], [], []
    libs = {name: gp.read_gmt(str(ROOT / cfg["paths"]["raw"] / "genesets" / f"{name}.gmt")) for name in pc["libraries"]}
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "all")
        expressed = (td.xlog > 0).mean(0) >= pc["min_detect"]
        sym = td.symbols[expressed]
        xw = td.xlog[:, expressed]
        pb = dl.pseudobulk(td.counts.toarray(), td.donor)
        pb.columns = td.symbols
        pb = pb.loc[:, expressed]
        rows = []
        for ep in pc["endpoints"][ct]:
            k = td.endpoints.index(ep)
            ok = ~np.isnan(td.y[:, k])
            r, z, p = within_assoc(xw[ok], td.y[ok, k], td.group[ok], pc["n_perm"], rng)
            t = dl.donor_targets(td.y[:, k], td.donor, td.glucose, np.ones(len(td.donor), bool),
                                 cfg["donor_level"]["min_cells"])
            rho, pd_ = donor_assoc(pb, t, pc["n_perm"], rng)
            df = pd.DataFrame({"cell_type": ct, "endpoint": ep, "gene": sym, "within_r": r, "within_z": z,
                               "within_p": p, "donor_rho": rho, "donor_p": pd_})
            df["within_fdr"] = multipletests(df.within_p, method="fdr_bh")[1]
            df["donor_fdr"] = multipletests(df.donor_p, method="fdr_bh")[1]
            rows.append(df)
            ok2 = df.donor_rho.notna()
            top_w = set(df.nlargest(pc["top_n"], "within_z").gene) | set(df.nsmallest(pc["top_n"], "within_z").gene)
            top_d = set(df.loc[ok2].nlargest(pc["top_n"], "donor_rho").gene) | set(
                df.loc[ok2].nsmallest(pc["top_n"], "donor_rho").gene)
            conc.append({"cell_type": ct, "endpoint": ep, "n_genes": len(df), "n_donors": len(t),
                         "gene_level_spearman": spearmanr(df.within_z[ok2], df.donor_rho[ok2]).statistic,
                         "n_within_fdr05": int((df.within_fdr < 0.05).sum()),
                         "n_donor_fdr05": int((df.donor_fdr < 0.05).sum()),
                         "top_overlap": len(top_w & top_d)})
            for level, stat in (("within", "within_z"), ("donor", "donor_rho")):
                rnk = df.loc[ok2, ["gene", stat]].dropna().sort_values(stat, ascending=False)
                for lib, sets in libs.items():
                    res = gp.prerank(rnk, gene_sets=sets, min_size=15, max_size=500, permutation_num=pc["gsea_perm"],
                                     seed=cfg["seed"], threads=8, outdir=None, verbose=False).res2d
                    res = res.assign(cell_type=ct, endpoint=ep, level=level, library=lib)
                    enr_all.append(res[["cell_type", "endpoint", "level", "library", "Term", "NES", "NOM p-val",
                                        "FDR q-val", "Lead_genes"]])
        ga = pd.concat(rows)
        p1 = out / f"gene_associations_{ct}.tsv"
        ga.to_csv(p1, sep="\t", index=False)
        outputs.append(p1)
    p2, p3 = out / "level_concordance.tsv", out / "enrichment.tsv"
    pd.DataFrame(conc).to_csv(p2, sep="\t", index=False)
    pd.concat(enr_all).to_csv(p3, sep="\t", index=False)
    print(pd.DataFrame(conc).round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs + [p2, p3], seed=cfg["seed"])


if __name__ == "__main__":
    main()
