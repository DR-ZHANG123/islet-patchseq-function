"""补充图 S12：只用 ND 供体训练的模型的预测值中的 T2D 差异（实测、预测与两者之差），以及这些模型在留出 ND 供体中的供体内精度。
只读 results/09c_t2d_hidden 与 results/09g_t2d_mechanism 的结果表，并把每个面板的输入写到 figures/source_data/。

a, b  β（a）与 α（b）细胞：T2D − ND 的实测值、转录组预测值与残差（ND 标准差单位；供体等权线性模型，
      调整前驱糖尿病、葡萄糖浓度与来源中心；按状态 × 中心分层重抽样供体 95% 置信区间）；右侧为实测效应的置换 P 值。
c     预测值捕捉到的实测 T2D 差异比例（预测效应 / 实测效应），胞吐终点（有反应细胞）。
d     只用 ND 供体训练的岭回归在留出 ND 供体中的供体内 r：训练目标为细胞值、供体内差异或供体（× 葡萄糖）均值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, panel_label, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/09c_t2d_hidden"
ROWS = {"Beta": [("exocytosis", "positive responses", "Total exo. (pos.)"), ("late_exocytosis", "positive responses", "Late exo. (pos.)"),
                 ("early_exocytosis", "positive responses", "Early exo. (pos.)"), ("calcium_entry", "all cells", "Ca$^{2+}$ entry"),
                 ("late_ca_current", "all cells", "Late Ca$^{2+}$"), ("sodium_current", "all cells", "Na$^{+}$ peak"),
                 ("size", "all cells", "Size")],
        "Alpha": [("exocytosis", "positive responses", "Total exo. (pos.)"), ("early_exocytosis", "positive responses", "Early exo. (pos.)"),
                  ("late_exocytosis", "positive responses", "Late exo. (pos.)"), ("calcium_entry", "all cells", "Ca$^{2+}$ entry"),
                  ("late_ca_current", "all cells", "Late Ca$^{2+}$"), ("sodium_current", "all cells", "Na$^{+}$ peak"),
                  ("size", "all cells", "Size")]}
PARTS = [("beta_y", "Measured", "o", "#C0392B", "#C0392B"), ("beta_p", "Predicted from transcriptome", "s", "white", "#2C6FB0"),
         ("beta_r", "Measured − predicted", "D", "#9A9A9A", "#9A9A9A")]


def forest(ax, h: pd.DataFrame, ct: str, legend: bool) -> pd.DataFrame:
    rows = []
    for i, (ep, part, lab) in enumerate(ROWS[ct]):
        s = h[(h.cell_type == ct) & (h.endpoint == ep) & (h.part == part)].set_index("quantity")
        for k, (q, name, mk, fc, ec) in enumerate(PARTS):
            r = s.loc[q]
            y = i + (k - 1) * 0.24
            ax.plot([r.lo_sd, r.hi_sd], [y, y], color=ec, lw=0.8)
            ax.plot(r.estimate_sd, y, mk, ms=3.4, mfc=fc, mec=ec, mew=0.8)
            rows.append({"cell_type": ct, "endpoint": ep, "cells": part, "quantity": name, "effect_sd": r.estimate_sd,
                         "lo_sd": r.lo_sd, "hi_sd": r.hi_sd, "p_perm": r.p_perm, "n_nd_donors": r.n_nd_donors,
                         "n_t2d_donors": r.n_t2d_donors})
        p = s.loc["beta_y", "p_perm"]
        ax.text(1.02, i, f"{p:.3f}" if p < 0.01 else f"{p:.2f}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=6)
    ax.text(1.02, -0.85, "P", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6)
    ax.axvline(0, color="#999999", lw=0.6, zorder=0)
    ax.set_yticks(range(len(ROWS[ct])))
    ax.set_yticklabels([l for _, _, l in ROWS[ct]], fontsize=6.5)
    ax.set_ylim(len(ROWS[ct]) - 0.5, -0.5)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xlabel(f"{'β' if ct == 'Beta' else 'α'} cells, T2D − ND (ND SD units)")
    ax.set_title("β cells" if ct == "Beta" else "α cells", fontsize=7, color=TYPE_COLORS[ct], pad=2)
    if legend:
        hs = [ax.plot([], [], mk, ms=3.4, mfc=fc, mec=ec, mew=0.8, label=name)[0] for _, name, mk, fc, ec in PARTS]
        ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(1.3, 1.07), ncol=3, fontsize=6.5,
                  frameon=False, handletextpad=0.3, columnspacing=1.4)
    return pd.DataFrame(rows)


CAPT = [("Beta", "exocytosis", "β total"), ("Beta", "late_exocytosis", "β late"),
        ("Alpha", "exocytosis", "α total"), ("Alpha", "early_exocytosis", "α early")]


def captured(ax, h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, (ct, ep, lab) in enumerate(CAPT):
        r = h[(h.cell_type == ct) & (h.endpoint == ep) & (h.part == "positive responses")
              & (h.quantity == "captured_fraction")].iloc[0]
        ax.bar(x, r.estimate, width=0.55, color=TYPE_COLORS[ct], alpha=0.85)
        ax.plot([x, x], [r.lo, r.hi], color="#333333", lw=0.8)
        rows.append({"cell_type": ct, "endpoint": ep, "cells": "positive responses", "captured_fraction": r.estimate,
                     "lo": r.lo, "hi": r.hi})
    ax.axhline(1, color="#555555", lw=0.7, ls="--")
    ax.text(len(CAPT) - 0.55, 1.03, "fully captured", ha="right", va="bottom", fontsize=6, color="#555555")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(range(len(CAPT))); ax.set_xticklabels([l for _, _, l in CAPT], fontsize=6.5)
    ax.set_xlim(-0.6, len(CAPT) - 0.4)
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("Share of T2D difference\ncaptured by predictions")
    ax.set_xlabel("Exocytosis, positive responses")
    return pd.DataFrame(rows)


TARGETS = [("pooled", "Cell values", "o", None), ("within", "Within-donor differences", "s", "white"),
           ("between", "Donor means", "D", "#BBBBBB")]
DIAG = [("exocytosis", "Total exo."), ("early_exocytosis", "Early exo."), ("late_exocytosis", "Late exo."),
        ("size", "Size")]


def targets(ax) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results/09g_t2d_mechanism/nd_trained_targets.tsv", sep="\t")
    d = d[d.status == "ND"]
    col = TYPE_COLORS["Beta"]
    for x, (ep, _) in enumerate(DIAG):
        for k, (tg, _, mk, fc) in enumerate(TARGETS):
            r = d[(d.endpoint == ep) & (d.training_target == tg)].iloc[0]
            xx = x + (k - 1) * 0.24
            ax.plot([xx, xx], [r.lo, r.hi], color=col, lw=0.8)
            ax.plot(xx, r.r, mk, ms=3.4, mfc=col if fc is None else fc, mec=col, mew=0.8)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(range(len(DIAG))); ax.set_xticklabels([l for _, l in DIAG], fontsize=6.5)
    ax.set_xlim(-0.6, len(DIAG) - 0.4)
    ax.set_ylabel("Within-donor r, held-out\nND donors (ND-trained ridge)")
    hs = [ax.plot([], [], mk, ms=3.4, mfc=col if fc is None else fc, mec=col, mew=0.8, label=lab)[0]
          for _, lab, mk, fc in TARGETS]
    ax.legend(handles=hs, loc="upper left", fontsize=6, frameon=False, handletextpad=0.3, title="Training target",
              title_fontsize=6)
    return d


def main() -> None:
    apply_style()
    h = pd.read_csv(RES / "hidden_effects.tsv", sep="\t")
    fig = plt.figure(figsize=(170 * MM, 130 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.55, height_ratios=[1.15, 0.85], top=0.90, bottom=0.10)
    r1 = gs[0].subgridspec(1, 2, wspace=0.55)
    a, b = fig.add_subplot(r1[0, 0]), fig.add_subplot(r1[0, 1])
    g2 = gs[1].subgridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.45)
    c = fig.add_subplot(g2[0, 0])
    dd = fig.add_subplot(g2[0, 1])
    save_source(forest(a, h, "Beta", True), ROOT, "FigS12", "a_beta_measured_predicted")
    save_source(forest(b, h, "Alpha", False), ROOT, "FigS12", "b_alpha_measured_predicted")
    save_source(captured(c, h), ROOT, "FigS12", "c_captured_fraction")
    save_source(targets(dd), ROOT, "FigS12", "d_nd_trained_targets")
    for ax, l in zip((a, b, c, dd), "abcd"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "FigS12")


if __name__ == "__main__":
    main()
