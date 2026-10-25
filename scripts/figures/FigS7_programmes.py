"""Fig. 5：供体内电生理差异对应内分泌身份与细胞应激程序。只读结果表。

a–b  β、α 细胞峰值 Na⁺ 电流：每个基因的供体内关联（z）与供体层面关联（Spearman ρ）。
c    供体内关联的 MSigDB hallmark 富集（标准化富集分数；* FDR < 0.05）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, panel_label, place_labels, save_fig, save_source  # noqa: E402
from Fig2_variance import SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"Beta": ["SCN3A", "NPY", "SHISAL2B", "LSAMP", "GAP43", "FXYD2", "ATF4", "CD9", "PPP1R1A", "TMED6"],
          "Alpha": ["GCG", "SCN3A", "SLC30A8", "FEV", "PLK2", "PPP1R15A", "SQSTM1", "RHOB"]}
COLS = [("Beta", "sodium_current"), ("Beta", "late_ca_current"), ("Beta", "calcium_entry"), ("Beta", "size"),
        ("Beta", "exocytosis"), ("Alpha", "sodium_current"), ("Alpha", "early_ca_current"),
        ("Alpha", "late_ca_current"), ("Alpha", "size"), ("Alpha", "exocytosis")]
TERMS = ["Pancreas Beta Cells", "Oxidative Phosphorylation", "Unfolded Protein Response",
         "TNF-alpha Signaling via NF-kB", "Hypoxia", "mTORC1 Signaling", "p53 Pathway", "Apoptosis"]
TERM_LABEL = {"Pancreas Beta Cells": "Pancreas beta cells", "Oxidative Phosphorylation": "Oxidative phosphorylation",
              "Unfolded Protein Response": "Unfolded protein response",
              "TNF-alpha Signaling via NF-kB": "TNF-α signalling via NF-κB", "Hypoxia": "Hypoxia",
              "mTORC1 Signaling": "mTORC1 signalling", "p53 Pathway": "p53 pathway", "Apoptosis": "Apoptosis"}


def panel_genes(ax, ct):
    g = pd.read_csv(ROOT / "results/08_programmes" / f"gene_associations_{ct}.tsv", sep="\t")
    g = g[g.endpoint == "sodium_current"].dropna(subset=["donor_rho"])
    lab = g[g.gene.isin(LABELS[ct])]
    ax.scatter(g.within_z, g.donor_rho, s=2, color="#BBBBBB", lw=0, rasterized=True, gid="background")
    ax.scatter(lab.within_z, lab.donor_rho, s=10, color=TYPE_COLORS[ct], lw=0.3, edgecolor="white", zorder=3)
    ax.axhline(0, color="#999999", lw=0.5); ax.axvline(0, color="#999999", lw=0.5)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.08 * (x1 - x0), x1 + 0.12 * (x1 - x0)); ax.set_ylim(y0, y1 + 0.06 * (y1 - y0))
    ax.set_xlabel("Within-donor association (z)")
    ax.set_ylabel("Donor-level association (ρ)")
    ax.set_title(f"{'β' if ct == 'Beta' else 'α'} cells, peak Na$^{{+}}$ current", fontsize=7.5, color=TYPE_COLORS[ct])
    place_labels(ax, lab.within_z.values, lab.donor_rho.values, lab.gene.values, fontsize=6, italic=True,
                 min_r=8, max_r=60)
    return g[["gene", "within_z", "within_fdr", "donor_rho", "donor_fdr"]].assign(cell_type=ct)


def panel_hallmark(ax):
    e = pd.read_csv(ROOT / "results/08_programmes/enrichment.tsv", sep="\t")
    e = e[(e.library == "MSigDB_Hallmark_2020") & (e.level == "within")].copy()
    e["q"] = pd.to_numeric(e["FDR q-val"], errors="coerce")
    nes = np.full((len(TERMS), len(COLS)), np.nan)
    q = np.full_like(nes, np.nan)
    for j, (ct, ep) in enumerate(COLS):
        s = e[(e.cell_type == ct) & (e.endpoint == ep)].set_index("Term")
        for i, t in enumerate(TERMS):
            if t in s.index:
                nes[i, j], q[i, j] = s.loc[t, "NES"], s.loc[t, "q"]
    im = ax.imshow(nes, cmap="RdBu_r", vmin=-2.6, vmax=2.6, aspect="auto")
    for i in range(nes.shape[0]):
        for j in range(nes.shape[1]):
            if q[i, j] < 0.05:
                t = ax.text(j, i, "*", ha="center", va="center", fontsize=8,
                            color="white" if abs(nes[i, j]) > 1.8 else "#222222")
                t._overlap_exempt = True
    ax.set_yticks(range(len(TERMS))); ax.set_yticklabels([TERM_LABEL[t] for t in TERMS], fontsize=6)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([f"{'β' if ct == 'Beta' else 'α'} {SHORT[ep]}" for ct, ep in COLS], rotation=45, ha="right",
                       fontsize=6)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Normalized enrichment score", fontsize=6); cb.ax.tick_params(labelsize=6)
    rows = [{"term": TERMS[i], "cell_type": COLS[j][0], "endpoint": COLS[j][1], "NES": nes[i, j], "fdr": q[i, j]}
            for i in range(len(TERMS)) for j in range(len(COLS))]
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(170 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.55, wspace=0.32)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ga, gb = panel_genes(a, "Beta"), panel_genes(b, "Alpha")
    c = fig.add_subplot(gs[1, :])
    hm = panel_hallmark(c)
    save_source(pd.concat([ga, gb]), ROOT, "FigS7", "ab_gene_associations_sodium")
    save_source(hm, ROOT, "FigS7", "c_hallmark_within")
    for ax, l in zip((a, b, c), "abc"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "FigS7")


if __name__ == "__main__":
    main()
