"""Stage 01：构建配对数据集（单细胞计数 + 实测电生理 + 供体信息）并完成配对样本审计。

输出
- data/processed/counts.npz：全部 α/β 细胞 × 基因并集的整数计数（还原自 HumanIslets TMM-logCPM）
- results/01_dataset/genes.tsv：基因表，含其在六个矩阵（类型 × 葡萄糖）中是否保留的标记
- results/01_dataset/cells.tsv：细胞表（供体、类型、葡萄糖、功能终点、GEO 来源、还原质控）
- results/01_dataset/donors.tsv：供体表（统一糖代谢分组、年龄、性别、BMI、HbA1c、来源中心）
- results/01_dataset/audit_*.tsv：按类型 × 葡萄糖 × 分组 × 终点统计可用细胞与独立供体数
- results/01_dataset/geo_concordance.tsv：还原计数与 GEO 原始计数的一致性（逐细胞 Spearman）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from hi_expr import read_matrix  # noqa: E402
import donors as dn  # noqa: E402

STAGE = "01_dataset"


def load_matrices(raw: Path, types: list[str], glucose: list[int]):
    mats = {}
    for t in types:
        for g in glucose:
            m = read_matrix(raw / "humanislets" / f"sc_{t}_{g}.h5")
            mats[f"{t}_{g}"] = m
            print(f"  {t}_{g}: {m.counts.shape}, max integer deviation {np.nanmax(m.max_dev):.2e}")
    return mats


def union_counts(mats: dict):
    genes = pd.DataFrame(columns=["entrez", "symbol"])
    for m in mats.values():
        genes = pd.concat([genes, pd.DataFrame({"entrez": m.entrez, "symbol": m.symbols})])
    genes = genes.drop_duplicates("entrez").reset_index(drop=True)
    pos = {e: i for i, e in enumerate(genes.entrez)}
    blocks, cells = [], []
    for key, m in mats.items():
        cols = np.array([pos[e] for e in m.entrez])
        coo = m.counts.tocoo()
        blocks.append(sp.csr_matrix((coo.data, (coo.row, cols[coo.col])), shape=(m.counts.shape[0], len(genes))))
        genes[key] = genes.entrez.isin(set(m.entrez))
        cells.append(pd.DataFrame({"cell_id": m.cells, "matrix": key, "max_dev": m.max_dev,
                                   "min_count": m.min_count, "n_counts": np.asarray(m.counts.sum(1)).ravel(),
                                   "n_genes": np.diff(m.counts.indptr)}))
    return sp.vstack(blocks).tocsr(), genes, pd.concat(cells, ignore_index=True)


def geo_membership(raw: Path, cfg: dict) -> dict[str, set]:
    out = {}
    for acc, rel in cfg["download"]["geo_counts"].items():
        p = raw / "geo" / Path(rel).name
        sep = "," if p.name.endswith(".csv.gz") else "\t"
        out[acc] = set(pd.read_csv(p, sep=sep, nrows=0, encoding="utf-8-sig").columns[1:])
    return out


def geo_concordance(raw: Path, cfg: dict, counts, genes, cells, n_per: int, rng) -> pd.DataFrame:
    """抽样核对：还原计数与 GEO 原始计数在共有基因上的逐细胞 Spearman 相关。"""
    rows = []
    sym = genes.symbol.values
    for acc, rel in cfg["download"]["geo_counts"].items():
        sub = cells[cells.geo == acc]
        if sub.empty:
            continue
        pick = sub.sample(min(n_per, len(sub)), random_state=int(rng.integers(1e9)))
        p = raw / "geo" / Path(rel).name
        sep = "," if p.name.endswith(".csv.gz") else "\t"
        first = pd.read_csv(p, sep=sep, nrows=0, encoding="utf-8-sig").columns[0]
        g = pd.read_csv(p, sep=sep, usecols=[first] + list(pick.cell_id), index_col=0, encoding="utf-8-sig")
        g = g[~g.index.duplicated()]
        common = [s for s in pd.unique(sym) if s in g.index]
        idx = pd.Series(np.arange(len(sym)), index=sym)
        idx = idx[~idx.index.duplicated()]
        for _, c in pick.iterrows():
            a = counts[c.row].toarray().ravel()[idx[common].values]
            b = g.loc[common, c.cell_id].values
            keep = (a > 0) | (b > 0)
            rows.append({"geo": acc, "cell_id": c.cell_id, "n_genes": int(keep.sum()),
                         "spearman": spearmanr(a[keep], b[keep]).statistic})
    return pd.DataFrame(rows)


def audit(cells: pd.DataFrame, panel: dict) -> pd.DataFrame:
    rows = []
    for ep in panel.values():
        ok = cells[cells[ep].notna()]
        for (t, gl, st), s in ok.groupby(["cell_type", "glucose", "status"]):
            rows.append({"endpoint": ep, "cell_type": t, "glucose": gl, "status": st, "cells": len(s),
                         "donors": s.donor.nunique(), "donors_ge5": int((s.groupby("donor").size() >= 5).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    raw = ROOT / cfg["paths"]["raw"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / cfg["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["dataset"]

    print("reading HumanIslets matrices ...")
    mats = load_matrices(raw, dcfg["cell_types"], dcfg["glucose_levels"])
    counts, genes, expr_cells = union_counts(mats)
    expr_cells["row"] = np.arange(len(expr_cells))

    eph = pd.read_csv(raw / "humanislets" / "ephys_cell.csv")
    panel = {k: v[0] for k, v in dcfg["endpoint_panel"].items()}
    eph = eph.rename(columns={**panel, "glucose_mM_cell": "glucose_ephys"})
    cells = expr_cells.merge(eph, on="cell_id", how="inner")
    cells["glucose"] = cells.matrix.str.split("_").str[1].astype(int)
    assert (cells.glucose == cells.glucose_ephys).all(), "glucose mismatch between ephys and expression"
    assert (cells.matrix.str.split("_").str[0] == cells.cell_type).all(), "cell type mismatch"
    cells = cells.rename(columns={"record_id": "donor"})

    member = geo_membership(raw, cfg)
    cells["geo"] = [next((a for a, s in member.items() if c in s), "PANC-DB") for c in cells.cell_id]

    # 供体信息
    soft = {a: dn.parse_geo_soft(raw / "geo_soft" / f"{a}_family.soft.gz") for a in cfg["download"]["geo_soft"]}
    eph_map = eph.rename(columns={"record_id": "record_id"})[["cell_id", "record_id"]]
    sources = [dn.from_humanislets(raw / "humanislets" / "donor.csv"),
               dn.from_pankbase(raw / "pankbase" / "pankbase_matched_donors.json"),
               dn.from_jci(raw / "jci2025" / "jci-135-195267-s109.xlsx")]
    sources += [dn.from_geo(s, eph_map) for s in soft.values()]
    donors = dn.harmonize(sources, sorted(cells.donor.unique()))
    if donors.clinical.isna().any():
        raise ValueError(f"donors without diagnosis: {donors.donor[donors.clinical.isna()].tolist()}")
    cells = cells.merge(donors[["donor", "status", "site"]], on="donor", how="left")
    cells["qc_pass"] = (cells.max_dev < 1e-2) & (cells.min_count == 1) & (cells.cell_score >= dcfg["min_cell_score"])

    conc = geo_concordance(raw, cfg, counts, genes, cells, 40, rng)
    aud = audit(cells[cells.qc_pass], panel)

    sp.save_npz(proc / "counts.npz", counts)
    files = {"genes.tsv": genes, "cells.tsv": cells, "donors.tsv": donors, "audit_endpoints.tsv": aud,
             "geo_concordance.tsv": conc}
    for fn, df in files.items():
        df.to_csv(out / fn, sep="\t", index=False)
    summ = (cells[cells.qc_pass].groupby(["cell_type", "status"])
            .agg(cells=("cell_id", "size"), donors=("donor", "nunique")).reset_index())
    summ.to_csv(out / "audit_summary.tsv", sep="\t", index=False)
    print(summ.to_string(index=False))
    print("GEO concordance (median Spearman):", conc.groupby("geo").spearman.median().round(3).to_dict())
    print("QC fail:", int((~cells.qc_pass).sum()), "of", len(cells))
    inputs = sorted((raw / "humanislets").glob("*")) + sorted((raw / "geo").glob("*"))
    write_provenance(STAGE, inputs, [proc / "counts.npz"] + [out / f for f in files] + [out / "audit_summary.tsv"],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
