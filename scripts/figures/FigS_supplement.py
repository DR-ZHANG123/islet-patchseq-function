"""补充图 S6、S8、S10。只读结果表，输出面板与作图输入。（其余补充图见 FigS*_*.py：S1 嵌入、S2 方法差、S3 测序质量、
S4 供体参照、S5 外部校准与跨平台、S7 基因程序、S9 可靠性、S11 胰岛分泌、S12 ND 训练预测、S13 各状态供体内精度）

S6  供体层面模型（供体汇总表达谱）与单细胞预测取平均的供体层面精度（开发供体交叉验证）。
S8  供体层面基因关联的 hallmark 富集，以及两个层面基因关联的一致性。
S10 糖代谢状态对实测供体平均表型的影响（线性混合模型）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, STATUS_COLORS, TYPE_COLORS, apply_style, model_label, panel_label, save_fig, save_source  # noqa: E402
from Fig2_variance import ORDER, SHORT  # noqa: E402
from FigS7_programmes import COLS, TERM_LABEL, TERMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def fig_s6():
    rows = []
    for ct in ("Beta", "Alpha"):
        d = pd.read_csv(ROOT / "results/05_donor_level" / f"donor_cv_{ct}.tsv", sep="\t")
        rows.append(d.groupby(["endpoint", "model"]).spearman.agg(["mean", "std"]).reset_index().assign(cell_type=ct))
    d = pd.concat(rows)
    save_source(d, ROOT, "FigS6", "donor_level_cv")
    mods = [("donor_pseudobulk", "Summed-profile donor model", "#B2182B"), ("context_model", model_label("context_model", short=True), "#E7877B"),
            ("gradient_boosting", model_label("gradient_boosting"), "#1A9850"), ("ridge", "Ridge", "#2C6FB0")]
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 82 * MM), sharey=True)
    for ax, ct in zip(axs, ("Beta", "Alpha")):
        x = np.arange(len(ORDER))
        for k, (mod, lab, col) in enumerate(mods):
            s = d[(d.cell_type == ct) & (d.model == mod)].set_index("endpoint").loc[ORDER]
            ax.errorbar(x + (k - 1.5) * 0.18, s["mean"], yerr=s["std"], fmt="o", ms=2.8, color=col, elinewidth=0.6,
                        label=lab)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[e] for e in ORDER], rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="#999999", lw=0.5, ls=":")
        ax.set_title("β cells" if ct == "Beta" else "α cells", fontsize=7.5, color=TYPE_COLORS[ct])
        ax.set_ylim(-0.6, 1.0)
    axs[0].set_ylabel("Donor-level ρ (mean ± SD over repetitions)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=4, fontsize=6.5, frameon=False)
    for a_, l in zip(axs, "ab"):
        panel_label(a_, l)
    save_fig(fig, ROOT, "FigS6")


def fig_s8():
    e = pd.read_csv(ROOT / "results/08_programmes/enrichment.tsv", sep="\t")
    e = e[(e.library == "MSigDB_Hallmark_2020") & (e.level == "donor")].copy()
    e["q"] = pd.to_numeric(e["FDR q-val"], errors="coerce")
    nes = np.full((len(TERMS), len(COLS)), np.nan); q = np.full_like(nes, np.nan)
    for j, (ct, ep) in enumerate(COLS):
        s = e[(e.cell_type == ct) & (e.endpoint == ep)].set_index("Term")
        for i, t in enumerate(TERMS):
            if t in s.index:
                nes[i, j], q[i, j] = s.loc[t, "NES"], s.loc[t, "q"]
    c = pd.read_csv(ROOT / "results/08_programmes/level_concordance.tsv", sep="\t")
    save_source(c, ROOT, "FigS8", "b_level_concordance")
    save_source(pd.DataFrame([{"term": TERMS[i], "cell_type": COLS[j][0], "endpoint": COLS[j][1], "NES": nes[i, j],
                               "fdr": q[i, j]} for i in range(len(TERMS)) for j in range(len(COLS))]),
                ROOT, "FigS8", "a_hallmark_donor")
    fig = plt.figure(figsize=(170 * MM, 85 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
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
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Normalized enrichment score", fontsize=6); cb.ax.tick_params(labelsize=6)
    bx = fig.add_subplot(gs[0, 1])
    lab = [f"{'β' if ct == 'Beta' else 'α'} {SHORT[ep]}" for ct, ep in zip(c.cell_type, c.endpoint)]
    bx.barh(np.arange(len(c)), c.gene_level_spearman, color=[TYPE_COLORS[t] for t in c.cell_type])
    bx.set_yticks(np.arange(len(c))); bx.set_yticklabels(lab, fontsize=6)
    bx.invert_yaxis()
    bx.axvline(0, color="#999999", lw=0.5)
    bx.set_xlabel("Within–between agreement\nof gene associations (ρ)")
    for a_, l in zip((ax, bx), "ab"):
        panel_label(a_, l)
    save_fig(fig, ROOT, "FigS8")


def fig_s10():
    s = pd.read_csv(ROOT / "results/02_variance/status_effects.tsv", sep="\t")
    save_source(s, ROOT, "FigS10", "status_effects")
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 80 * MM), sharex=True)
    for ax, ct in zip(axs, ("Beta", "Alpha")):
        y = np.arange(len(ORDER))
        for k, (con, st) in enumerate((("preT2D vs ND", "preT2D"), ("T2D vs ND", "T2D"))):
            r = s[(s.cell_type == ct) & (s.contrast == con)].set_index("endpoint").loc[ORDER]
            ax.errorbar(r.estimate_sd, y + (k - 0.5) * 0.3, xerr=[r.estimate_sd - r.lo, r.hi - r.estimate_sd], fmt="o",
                        ms=3, color=STATUS_COLORS[st], elinewidth=0.7,
                        label="Prediabetes" if st == "preT2D" else "Type 2 diabetes")
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[e] for e in ORDER], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("Difference from no diabetes (SD units)")
        ax.set_title("β cells" if ct == "Beta" else "α cells", fontsize=7.5, color=TYPE_COLORS[ct])
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    for a_, l in zip(axs, "ab"):
        panel_label(a_, l)
    save_fig(fig, ROOT, "FigS10")


def main() -> None:
    apply_style()
    fig_s6(); fig_s8(); fig_s10()


if __name__ == "__main__":
    main()
