"""补充图 S11：同一批供体的胰岛胰岛素分泌（灌流与静态 GSIS）与胰岛胰岛素含量。只读 results/11_islet_secretion 与 results/09g_t2d_mechanism。

a     T2D 与 ND 供体的胰岛分泌之比，以及分离记录中的胰岛胰岛素含量（每 IEQ、胰岛素/DNA）之比（对数尺度差的指数；按供体重抽样 95% 置信区间；状态标签置换 P 值）。
b–d   供体层面 β 细胞正值总胞吐、偏移效率与转录组预测胞吐 vs 30 mM KCl 刺激的胰岛分泌（按糖尿病状态着色；
      标注调整糖尿病状态的 Spearman ρ 与置换 P 值）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, STATUS_COLORS, STATUS_LABELS, apply_style, panel_label, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/11_islet_secretion"
OUTCOMES = [("peri_auc_baseline_3mmgluc", "Perifusion, 3 mM glucose"), ("peri_auc_gluc_6mmgluc", "Perifusion, 6 mM glucose"),
            ("peri_auc_gluc_15mmgluc", "Perifusion, 15 mM glucose"), ("peri_auc_gluc_30mmkcl", "Perifusion, 30 mM KCl"),
            ("peri_auc_leu_5mmleu", "Perifusion, 5 mM leucine"), ("peri_auc_olp_1p5mmolp", "Perifusion, oleate/palmitate"),
            ("peri_glucose_fold", "Perifusion, 15 mM/3 mM glucose"), ("gsis_log_stim_index", "Static GSIS, stimulation index"),
            ("gsis_log_secreted_percent", "Static GSIS, % of content")]
FEATURES = [("positive_total", "β-cell positive total exocytosis\n(donor mean, log)"),
            ("coupling_total", "β-cell coupling efficiency\n(donor mean)"),
            ("predicted_exocytosis", "β-cell exocytosis predicted\nfrom transcriptome (donor mean)")]
KCL = "peri_auc_gluc_30mmkcl"


INSULIN = [("insulinperieq", "Islet insulin content per IEQ"), ("insulindnaratio", "Islet insulin:DNA ratio")]


def ratios(ax, s: pd.DataFrame, ins: pd.DataFrame) -> pd.DataFrame:
    s = s.set_index("outcome").loc[[o for o, _ in OUTCOMES]]
    i2 = ins.set_index("measure").loc[[m for m, _ in INSULIN]].rename(columns={"ratio": "ratio", "lo": "ratio_lo",
                                                                               "hi": "ratio_hi"})
    s = pd.concat([s, i2[["ratio", "ratio_lo", "ratio_hi", "p_perm", "n_nd", "n_t2d"]]])
    y = np.arange(len(s))
    ax.plot([s.ratio_lo, s.ratio_hi], [y, y], color=STATUS_COLORS["T2D"], lw=0.8)
    ax.plot(s.ratio, y, "o", ms=3.4, color=STATUS_COLORS["T2D"])
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(1.02, i, f"{r.p_perm:.3f}" if r.p_perm < 0.01 else f"{r.p_perm:.2f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6)
    ax.text(1.02, -0.9, "P", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6)
    ax.axvline(1, color="#999999", lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2"])
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in OUTCOMES + INSULIN], fontsize=6.5)
    ax.axhline(len(OUTCOMES) - 0.5, color="#DDDDDD", lw=0.6)
    ax.set_ylim(len(s) - 0.5, -0.5)
    ax.set_xlabel("Islet insulin secretion and content,\nT2D/ND ratio")
    return s.reset_index().rename(columns={"index": "outcome"})


def scatter(ax, f: pd.DataFrame, feat: str, xlabel: str, legend: bool) -> pd.DataFrame:
    d = f[(f.cell_type == "Beta") & (f.feature == feat) & (f.outcome == KCL)]
    for st in ("ND", "preT2D", "T2D"):
        g = d[d.status == st]
        ax.plot(g.x, g.y, "o", ms=3.2, color=STATUS_COLORS[st], alpha=0.85, mec="white", mew=0.3,
                label=STATUS_LABELS[st])
    r = d.iloc[0]
    title = (f"ρ = {r.rho_adj:.2f} (95% CI {r.rho_adj_lo:.2f} to {r.rho_adj_hi:.2f}), P = {r.rho_adj_p:.2f}\n"
             f"{int(r.n_donors)} donors")
    ax.set_title(title.replace("-", "−"), fontsize=6, pad=3)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.set_ylabel("Perifusion, 30 mM KCl\n(log area under curve)", fontsize=6.5)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.42), ncol=3, fontsize=6, frameon=False,
                  handletextpad=0.2, columnspacing=1.0)
    return d


def main() -> None:
    apply_style()
    s = pd.read_csv(RES / "status_differences.tsv", sep="\t")
    f = pd.read_csv(RES / "fig_source.tsv", sep="\t")
    fig = plt.figure(figsize=(170 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.75, height_ratios=[1.0, 0.85], top=0.95, bottom=0.17)
    a = fig.add_subplot(gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.6])[0, 0])
    r2 = gs[1].subgridspec(1, 3, wspace=0.65)
    axs = [fig.add_subplot(r2[0, i]) for i in range(3)]
    ins = pd.read_csv(ROOT / "results/09g_t2d_mechanism/insulin_content.tsv", sep="\t")
    save_source(ratios(a, s, ins), ROOT, "FigS11", "a_islet_secretion_t2d")
    for i, (ax, (feat, lab)) in enumerate(zip(axs, FEATURES)):
        save_source(scatter(ax, f, feat, lab, i == 0), ROOT, "FigS11", f"{'bcd'[i]}_{feat}_kcl")
    fig._col_tol = 0.16
    for ax, lab in zip([a] + axs, "abcd"):
        panel_label(ax, lab)
    save_fig(fig, ROOT, "FigS11")


if __name__ == "__main__":
    main()
