"""Fig. 7：独立图谱的 T2D 转录特征在 Patch-seq β 细胞中复现；携带比例的零分布校准；该检验能识别哪些携带者。
只读 results/09f_t2d_robust、results/09g_t2d_mechanism 与 results/09e_t2d_conditioned 的结果表，
并把每个面板的输入写到 figures/source_data/。

a  外部（Bandesh 等 2026 液滴图谱）β 细胞 T2D 特征在各供体 β 细胞中的平均得分（同葡萄糖 ND 细胞为参照的 ND 细胞 SD 单位），
   按糖代谢状态；标注 T2D − ND 差（供体等权模型，供体重抽样 95% 置信区间，中心内状态置换 P）。
b  正值胞吐反应 T2D 差异由 50 个主成分携带的比例：实测值，与两种零分布的 95% 范围——供体均值主成分在供体间置换
   （安慰剂：去掉供体层面转录组与功能的对应）和结局在供体 × 葡萄糖组内置换（去掉细胞层面对应、保留供体层面结构）。
c  模拟：把 β 细胞正值总胞吐的 T2D 差异的 50% 或 100% 换成由不同转录携带者承担，50 个主成分找回的比例；
   红线为实测比例。携带者：细胞层面 T2D 特征、20 基因特征、10 个 T2D 差异最大的胞吐/颗粒基因、
   供体均值加一半供体内特征与非转录噪声、只有供体均值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, STATUS_COLORS, STATUS_LABELS, TYPE_COLORS, apply_style, panel_label, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R9F = ROOT / "results/09f_t2d_robust"
R9G = ROOT / "results/09g_t2d_mechanism"
STATUS = ["ND", "preT2D", "T2D"]


def scores(ax) -> pd.DataFrame:
    d = pd.read_csv(R9G / "donor_scores.tsv", sep="\t")
    d = d[d.signature == "external"]
    rng = np.random.default_rng(7)
    for i, st in enumerate(STATUS):
        v = d[d.status == st].score.values
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=STATUS_COLORS[st], edgecolor="white", lw=0.3,
                   zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#222222", lw=1.0, zorder=4)
    e = pd.read_csv(R9F / "external_signature.tsv", sep="\t")
    r = e[(e.signature == "external") & (e.quantity == "score_t2d_nd_cell_sd")].iloc[0]
    ax.text(0.03, 0.97, f"T2D − ND: +{r.estimate:.2f} SD\n(95% CI {r.lo:.2f} to {r.hi:.2f})\nP = {r.p_perm:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(range(len(STATUS)))
    ax.set_xticklabels([STATUS_LABELS[s].replace(" ", "\n", 1) for s in STATUS], fontsize=6)
    ax.set_xlim(-0.6, len(STATUS) - 0.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.5 * (hi - lo))
    ax.set_ylabel("External T2D signature score,\ndonor mean (ND SD units)")
    return d.assign(t2d_minus_nd=r.estimate, lo=r.lo, hi=r.hi, p_perm=r.p_perm)


NULLS = [("Beta", "exocytosis", "β total"), ("Beta", "late_exocytosis", "β late"),
         ("Alpha", "exocytosis", "α total"), ("Alpha", "early_exocytosis", "α early")]


def nulls(ax) -> pd.DataFrame:
    pl = pd.read_csv(R9F / "placebo.tsv", sep="\t")
    wp = pd.read_csv(R9F / "within_permutation_null.tsv", sep="\t")
    rows = []
    for x, (ct, ep, lab) in enumerate(NULLS):
        p = pl[(pl.cell_type == ct) & (pl.endpoint == ep) & (pl.part == "positive responses")].iloc[0]
        w = wp[(wp.cell_type == ct) & (wp.endpoint == ep) & (wp.part == "positive responses") & (wp.n_pcs == 50)].iloc[0]
        ax.plot([x - 0.13] * 2, [p.placebo_lo, p.placebo_hi], color="#D3D3D3", lw=6, solid_capstyle="butt", zorder=1)
        ax.plot([x + 0.13] * 2, [w.null_lo, w.null_hi], color="#8C8C8C", lw=6, solid_capstyle="butt", zorder=1)
        ax.plot(x, p.observed_share, "o", ms=4.4, color=TYPE_COLORS[ct], mec="white", mew=0.5, zorder=3)
        ax.text(x, max(p.placebo_hi, w.null_hi, p.observed_share) + 0.06,
                f"P = {p.p_placebo_ge_observed:.3f}\n({w.p_null_ge_observed:.3f})", ha="center", va="bottom",
                fontsize=6, color="#333333")
        rows.append({"cell_type": ct, "endpoint": ep, "cells": "positive responses", "observed_share": p.observed_share,
                     "placebo_lo": p.placebo_lo, "placebo_hi": p.placebo_hi, "p_placebo": p.p_placebo_ge_observed,
                     "n_placebo": p.n_placebo, "within_null_lo": w.null_lo, "within_null_hi": w.null_hi,
                     "within_null_median": w.null_median, "p_within_null": w.p_null_ge_observed, "n_perm": w.n_perm})
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.set_xticks(range(len(NULLS)))
    ax.set_xticklabels([lab for _, _, lab in NULLS], fontsize=6.5)
    ax.set_xlim(-0.6, len(NULLS) - 0.4)
    ax.set_ylim(-0.5, 1.05)
    ax.set_ylabel("Share carried by\n50 expression components")
    hs = [ax.plot([], [], "o", ms=4.4, color="#555555", label="Observed")[0],
          Patch(facecolor="#D3D3D3", edgecolor="none", label="Donor means permuted"),
          Patch(facecolor="#8C8C8C", edgecolor="none", label="Outcome permuted within donors")]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, frameon=False,
              handletextpad=0.5, columnspacing=1.4, handlelength=1.2)
    return pd.DataFrame(rows)


CARRIERS = [("09f", "signature_cell", "Signature,\ncell level"), ("09f", "small_geneset_cell", "20 signature\ngenes"),
            ("09g", "panel10", "10 exocytosis/\ngranule genes"),
            ("09f", "signature_noisy_rho0.5", "Signature, half\nwithin donors"),
            ("09f", "signature_donor", "Signature,\ndonor means")]
YLIM = (-0.7, 2.2)


def simulations(ax) -> pd.DataFrame:
    s9f = pd.read_csv(R9F / "simulations.tsv", sep="\t")
    s9f = s9f[(s9f.endpoint == "exocytosis") & (s9f.part == "positive responses") & (s9f.estimator == "full50")]
    s9g = pd.read_csv(R9G / "panel_simulation.tsv", sep="\t")
    s9g = s9g[(s9g.endpoint == "exocytosis") & (s9g.part == "positive responses") & (s9g.estimator == "full50")]
    obs = pd.read_csv(ROOT / "results/09e_t2d_conditioned/effects.tsv", sep="\t")
    obs = obs[(obs.cell_type == "Beta") & (obs.endpoint == "exocytosis") & (obs.part == "positive responses")
              & (obs.quantity == "prop_pc50")].iloc[0]
    rows = []
    col = TYPE_COLORS["Beta"]
    for x, (src, key, lab) in enumerate(CARRIERS):
        for k, (pi, mk) in enumerate(((0.5, "^"), (1.0, "s"))):
            d = s9g[s9g.pi == pi] if src == "09g" else s9f[(s9f.carrier == key) & (s9f.pi == pi)]
            r = d.iloc[0]
            xx = x + (k - 0.5) * 0.28
            ax.plot([xx, xx], [max(r.lo, YLIM[0]), min(r.hi, YLIM[1])], color=col, lw=0.8)
            ax.plot(xx, r.estimate, mk, ms=3.8, mfc="white" if pi == 0.5 else col, mec=col, mew=0.8)
            rows.append({"carrier": lab.replace("\n", " "), "simulated_share": pi, "recovered_share": r.estimate,
                         "lo": r.lo, "hi": r.hi, "share_boot_above_0": r.share_boot_above_0,
                         "within_r_outcome_carrier": r.within_r_y_carrier, "source_stage": src})
    for v in (0.5, 1.0):
        ax.axhline(v, color="#777777", lw=0.6, ls="--", zorder=0)
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.axhline(obs.estimate, color="#C0392B", lw=1.0, zorder=0)
    ax.text(len(CARRIERS) - 0.45, obs.estimate - 0.05, "observed", ha="right", va="top", fontsize=6, color="#C0392B")
    ax.set_xticks(range(len(CARRIERS)))
    ax.set_xticklabels([lab for _, _, lab in CARRIERS], fontsize=6.3)
    ax.set_xlim(-0.6, len(CARRIERS) - 0.4)
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Recovered share,\n50 expression components")
    hs = [ax.plot([], [], "^", ms=3.8, mfc="white", mec="#555555", mew=0.8, label="50% of defect carried")[0],
          ax.plot([], [], "s", ms=3.8, color="#555555", label="100% of defect carried")[0]]
    ax.legend(handles=hs, loc="upper right", fontsize=6, frameon=True, handletextpad=0.3, borderpad=0.4)
    rows.append({"carrier": "observed data", "simulated_share": np.nan, "recovered_share": obs.estimate,
                 "lo": obs.lo, "hi": obs.hi, "share_boot_above_0": np.nan, "within_r_outcome_carrier": np.nan,
                 "source_stage": "09e"})
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(170 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.42, height_ratios=[1.0, 1.0], top=0.9, bottom=0.1)
    r1 = gs[0].subgridspec(1, 2, wspace=0.42, width_ratios=[0.8, 1.2])
    a, b = fig.add_subplot(r1[0, 0]), fig.add_subplot(r1[0, 1])
    c = fig.add_subplot(gs[1])
    save_source(scores(a), ROOT, "Fig7", "a_external_signature_donors")
    save_source(nulls(b), ROOT, "Fig7", "b_share_null_distributions")
    save_source(simulations(c), ROOT, "Fig7", "c_simulated_carriers")
    for ax, lab in zip((a, b, c), "abc"):
        panel_label(ax, lab)
    save_fig(fig, ROOT, "Fig7")


if __name__ == "__main__":
    main()
