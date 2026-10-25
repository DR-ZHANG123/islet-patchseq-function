"""Stage 08b：Fig. 6 的基因–终点关联与 hallmark 富集在控制质控协变量后是否保持。

对 Fig. 6c 的每个 细胞类型 × 终点：供体内偏相关（表达秩与终点秩先对质控协变量取残差：log 检出基因数、
log 计数、线粒体比例、本类激素比例、他类激素 log1p CPM；离子电流类终点另加 log 细胞大小），
置换次数、基因过滤与 Stage 08 相同；随后用调整后的 z 排序重做 GSEApy 预排序富集（MSigDB Hallmark，
min_size 15、max_size 500、置换次数与种子同 Stage 08）。与 Stage 08 原结果逐一比较符号与显著性。
输出 results/08b_programme_qc/{gene_associations_qc_<type>.tsv, featured_genes.tsv, hallmark_qc.tsv, hallmark_compare.tsv}。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/run.yaml").read_text())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_cfg["programme_qc"]["numpy_threads"]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
from programme_qc import bh, compare_genes, compare_terms, covariate_matrix, partial_within_assoc  # noqa: E402

STAGE = "08b_programme_qc"


def run_type(cfg, ct, rng, gp, sets):
    pc, qc = cfg["programmes"], cfg["programme_qc"]
    td = load_type(cfg, ct, "all")
    expressed = (td.xlog > 0).mean(0) >= pc["min_detect"]
    sym, xw = td.symbols[expressed], td.xlog[:, expressed]
    orig = pd.read_csv(ROOT / "results/08_programmes" / f"gene_associations_{ct}.tsv", sep="\t")
    enr = pd.read_csv(ROOT / "results/08_programmes/enrichment.tsv", sep="\t")
    rows, hall, genes_cmp, terms_cmp = [], [], [], []
    for ep in qc["endpoints"][ct]:
        k = td.endpoints.index(ep)
        qm, cols = covariate_matrix(td, cfg, ct, ep)
        ok = ~np.isnan(td.y[:, k])
        r, z, p, n = partial_within_assoc(xw[ok], td.y[ok, k], qm[ok], td.group[ok], pc["n_perm"], rng)
        df = pd.DataFrame({"cell_type": ct, "endpoint": ep, "gene": sym, "within_r": r, "within_z": z, "within_p": p})
        df["within_fdr"] = bh(df.within_p.values)
        df["covariates"] = ",".join(cols)
        df["n_cells"] = n
        rows.append(df)
        o = orig[orig.endpoint == ep]
        both = o[["gene", "within_z"]].merge(df[["gene", "within_z"]], on="gene", suffixes=("_o", "_a"))
        print(f"[{ct} {ep}] n={n}  gene-level Spearman original vs adjusted z = "
              f"{both.within_z_o.corr(both.within_z_a, method='spearman'):.3f}", flush=True)
        for gene_ep, glist in qc["featured_genes"].get(ct, {}).items():
            if gene_ep == ep:
                genes_cmp.append(compare_genes(o, df, glist).assign(cell_type=ct, endpoint=ep))
        # 与 Stage 08 相同的排序基因集合：供体层面相关可计算的基因
        keep = set(o.loc[o.donor_rho.notna(), "gene"])
        rnk = df[df.gene.isin(keep)][["gene", "within_z"]].dropna().sort_values("within_z", ascending=False)
        res = gp.prerank(rnk, gene_sets=sets, min_size=15, max_size=500, permutation_num=pc["gsea_perm"],
                         seed=cfg["seed"], threads=qc["gsea_threads"], outdir=None, verbose=False).res2d
        res = res.assign(cell_type=ct, endpoint=ep, level="within_qc_adjusted", library=qc["library"])
        hall.append(res[["cell_type", "endpoint", "level", "library", "Term", "NES", "NOM p-val", "FDR q-val",
                         "Lead_genes"]])
        e0 = enr[(enr.cell_type == ct) & (enr.endpoint == ep) & (enr.level == "within") & (enr.library == qc["library"])]
        terms_cmp.append(compare_terms(e0, res, qc["featured_terms"]).assign(cell_type=ct, endpoint=ep))
    return pd.concat(rows), hall, genes_cmp, terms_cmp


def main() -> None:
    import gseapy as gp
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 8)
    qc = cfg["programme_qc"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    sets = gp.read_gmt(str(ROOT / cfg["paths"]["raw"] / "genesets" / f"{qc['library']}.gmt"))
    outputs, hall, genes_cmp, terms_cmp = [], [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        ga, h, g, t = run_type(cfg, ct, rng, gp, sets)
        p = out / f"gene_associations_qc_{ct}.tsv"
        ga.to_csv(p, sep="\t", index=False)
        outputs.append(p)
        hall += h
        genes_cmp += g
        terms_cmp += t
    tabs = {"featured_genes.tsv": pd.concat(genes_cmp), "hallmark_qc.tsv": pd.concat(hall),
            "hallmark_compare.tsv": pd.concat(terms_cmp)}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outputs.append(out / fn)
    pd.set_option("display.width", 250)
    print(tabs["featured_genes.tsv"].round(3).to_string())
    print(tabs["hallmark_compare.tsv"].round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv",
                             ROOT / "results/08_programmes/gene_associations_Beta.tsv",
                             ROOT / "results/08_programmes/gene_associations_Alpha.tsv",
                             ROOT / "results/08_programmes/enrichment.tsv"], outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
