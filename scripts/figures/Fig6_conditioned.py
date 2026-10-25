"""Fig. 6：以细胞自身转录组为条件后 β 细胞的 T2D 胞吐缺陷不变，α 细胞的缺陷部分被吸收。
只读 results/09e_t2d_conditioned、results/09f_t2d_robust 与 results/09g_t2d_mechanism 的结果表，
并把每个面板的输入写到 figures/source_data/。

a, b  β（a）与 α（b）细胞：T2D − ND 的实测差异、以 50 个表达主成分为条件后的差异与两者之差（转录组携带的部分），
      ND 标准差单位；供体等权线性模型（调整糖尿病前期、葡萄糖浓度与来源中心）；按状态 × 中心分层重抽样供体
      95% 置信区间；右侧为实测与条件化差异的置换 P 值。
c     正值胞吐反应 T2D 差异中由各种转录组表示携带的比例（β 总胞吐、β 晚期胞吐、α 总胞吐）：50 个主成分、
      交叉拟合岭回归、主成分的供体均值、留出供体与外部 T2D 转录特征、58 个胞吐/颗粒基因、ETV1。

"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, panel_label, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/09e_t2d_conditioned"
MAIN = 50
ROWS = {"Beta": [("exocytosis", "positive responses", "Total exo. (pos.)"),
                 ("late_exocytosis", "positive responses", "Late exo. (pos.)"),
                 ("early_exocytosis", "positive responses", "Early exo. (pos.)"),
                 ("calcium_entry", "all cells", "Ca$^{2+}$ entry"), ("late_ca_current", "all cells", "Late Ca$^{2+}$"),
                 ("sodium_current", "all cells", "Na$^{+}$ peak"), ("size", "all cells", "Size")],
        "Alpha": [("exocytosis", "positive responses", "Total exo. (pos.)"),
                  ("early_exocytosis", "positive responses", "Early exo. (pos.)"),
                  ("late_exocytosis", "positive responses", "Late exo. (pos.)"),
                  ("calcium_entry", "all cells", "Ca$^{2+}$ entry"), ("late_ca_current", "all cells", "Late Ca$^{2+}$"),
                  ("sodium_current", "all cells", "Na$^{+}$ peak"), ("size", "all cells", "Size")]}
PARTS = [("total", "Measured", "o", "#C0392B", "#C0392B"),
         (f"direct_pc{MAIN}", "Conditioned on transcriptome", "s", "white", "#2C6FB0"),
         (f"indirect_pc{MAIN}", "Carried by transcriptome", "D", "#9A9A9A", "#9A9A9A")]


def _p(p: float) -> str:
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def forest(ax, e: pd.DataFrame, ct: str, legend: bool) -> pd.DataFrame:
    rows = []
    for i, (ep, part, lab) in enumerate(ROWS[ct]):
        s = e[(e.cell_type == ct) & (e.endpoint == ep) & (e.part == part)].set_index("quantity")
        for k, (q, name, mk, fc, ec) in enumerate(PARTS):
            r = s.loc[q]
            y = i + (k - 1) * 0.24
            ax.plot([r.lo_sd, r.hi_sd], [y, y], color=ec, lw=0.8)
            ax.plot(r.estimate_sd, y, mk, ms=3.4, mfc=fc, mec=ec, mew=0.8)
            rows.append({"cell_type": ct, "endpoint": ep, "cells": part, "quantity": name, "effect_sd": r.estimate_sd,
                         "lo_sd": r.lo_sd, "hi_sd": r.hi_sd, "p_perm": r.p_perm, "n_nd_donors": r.n_nd_donors,
                         "n_t2d_donors": r.n_t2d_donors})
        for x, q in ((1.02, "total"), (1.17, f"direct_pc{MAIN}")):
            ax.text(x, i, _p(s.loc[q, "p_perm"]), transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=6)
    for x, h in ((1.02, "P"), (1.17, "P$_{cond.}$")):
        ax.text(x, -0.85, h, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6)
    ax.axvline(0, color="#999999", lw=0.6, zorder=0)
    ax.set_yticks(range(len(ROWS[ct])))
    ax.set_yticklabels([lab for _, _, lab in ROWS[ct]], fontsize=6.5)
    ax.set_ylim(len(ROWS[ct]) - 0.5, -0.5)
    ax.set_xlim(-0.9, 0.7)
    ax.set_xlabel(f"{'β' if ct == 'Beta' else 'α'} cells, T2D − ND (ND SD units)")
    ax.set_title("β cells" if ct == "Beta" else "α cells", fontsize=7, color=TYPE_COLORS[ct], pad=2)
    if legend:
        hs = [ax.plot([], [], mk, ms=3.4, mfc=fc, mec=ec, mew=0.8, label=name)[0] for _, name, mk, fc, ec in PARTS]
        ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(1.45, 1.07), ncol=3, fontsize=6.5,
                  frameon=False, handletextpad=0.3, columnspacing=1.4)
    return pd.DataFrame(rows)


COND = [("pc50", "50 expression components", "09e", "prop_pc50"),
        ("dml", "Cross-fitted ridge, 2,000 genes", "09e", "prop_dml"),
        ("between3", "Donor means of 3 components", "09f", "prop_between3"),
        ("between5", "Donor means of 5 components", "09f", "prop_between5"),
        ("sig_int", "T2D signature, held-out donors", "09f", "prop_signature"),
        ("sig_ext", "T2D signature, external atlas", "09f", "prop_external_signature"),
        ("panel", "58 exocytosis and granule genes", "09g", "prop_panel_pcs"),
        ("etv1", "ETV1", "09g", "prop_candidate")]
SERIES = [("Beta", "exocytosis", "β total", "o", None), ("Beta", "late_exocytosis", "β late", "^", "white"),
          ("Alpha", "exocytosis", "α total", "o", None)]
XLIM = (-1.0, 1.5)


def share_table(e9e: pd.DataFrame, e9f: pd.DataFrame, e9g: pd.DataFrame) -> pd.DataFrame:
    src = {"09e": e9e, "09f": e9f, "09g": e9g}
    rows = []
    for key, lab, s, q in COND:
        for ct, ep, slab, _, _ in SERIES:
            d = src[s]
            d = d[(d.endpoint == ep) & (d.part == "positive responses") & (d.quantity == q)]
            if "cell_type" in d:
                d = d[d.cell_type == ct]
            elif ct != "Beta":
                continue
            if d.empty:
                continue
            r = d.iloc[0]
            rows.append({"conditioning": lab, "series": slab, "cell_type": ct, "endpoint": ep,
                         "cells": "positive responses", "share_carried": r.estimate, "lo": r.lo, "hi": r.hi,
                         "source_stage": s})
    return pd.DataFrame(rows)


def shares(ax, t: pd.DataFrame) -> None:
    labs = [lab for _, lab, _, _ in COND]
    for k, (ct, ep, slab, mk, fc) in enumerate(SERIES):
        col = TYPE_COLORS[ct]
        for _, r in t[t.series == slab].iterrows():
            y = labs.index(r.conditioning) + (k - 1) * 0.24
            ax.plot([max(r.lo, XLIM[0]), min(r.hi, XLIM[1])], [y, y], color=col, lw=0.8)
            ax.plot(r.share_carried, y, mk, ms=3.4, mfc=col if fc is None else fc, mec=col, mew=0.8)
    ax.axvline(0, color="#999999", lw=0.6, zorder=0)
    ax.axvline(1, color="#555555", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=6.5)
    ax.set_ylim(len(labs) - 0.5, -0.5)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Share of T2D difference in positive exocytotic responses carried")
    hs = [ax.plot([], [], mk, ms=3.4, mfc=TYPE_COLORS[ct] if fc is None else fc, mec=TYPE_COLORS[ct], mew=0.8,
                  label=slab)[0] for ct, _, slab, mk, fc in SERIES]
    ax.legend(handles=hs, loc="lower right", fontsize=6.5, frameon=True, handletextpad=0.3, borderpad=0.4)


def main() -> None:
    apply_style()
    e = pd.read_csv(RES / "effects.tsv", sep="\t")
    e9f = pd.read_csv(ROOT / "results/09f_t2d_robust/conditioning.tsv", sep="\t")
    e9g = pd.read_csv(ROOT / "results/09g_t2d_mechanism/panel_conditioning.tsv", sep="\t")
    fig = plt.figure(figsize=(170 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.36, height_ratios=[1.15, 0.85], top=0.92, bottom=0.08)
    r1 = gs[0].subgridspec(1, 2, wspace=0.95)
    a, b = fig.add_subplot(r1[0, 0]), fig.add_subplot(r1[0, 1])
    r2 = gs[1].subgridspec(1, 3, width_ratios=[0.17, 1.0, 0.04])
    c = fig.add_subplot(r2[0, 1])
    save_source(forest(a, e, "Beta", True), ROOT, "Fig6", "a_beta_conditioned")
    save_source(forest(b, e, "Alpha", False), ROOT, "Fig6", "b_alpha_conditioned")
    t = share_table(e, e9f, e9g)
    shares(c, t)
    save_source(t, ROOT, "Fig6", "c_share_by_representation")
    for ax, lab in zip((a, b, c), "abc"):
        panel_label(ax, lab)
    save_fig(fig, ROOT, "Fig6")


if __name__ == "__main__":
    main()
