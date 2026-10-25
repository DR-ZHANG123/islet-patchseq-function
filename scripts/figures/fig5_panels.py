"""Fig. 5 的各个面板（只读 results/09_t2d_function、09g_t2d_mechanism、09h_t2d_glucose 的结果表）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from figstyle import STATUS_COLORS, TYPE_COLORS

SHORT = {"size": "Size", "sodium_current": "Na$^{+}$ peak", "early_ca_current": "Early Ca$^{2+}$",
         "late_ca_current": "Late Ca$^{2+}$", "calcium_entry": "Ca$^{2+}$ entry", "exocytosis": "Total exo.",
         "early_exocytosis": "Early exo.", "late_exocytosis": "Late exo."}
EPS = ["exocytosis", "early_exocytosis", "late_exocytosis", "calcium_entry", "sodium_current", "size"]
TYPE_SYM = {"Beta": "β", "Alpha": "α"}


def _stars(q):
    return "**" if q < 0.01 else ("*" if q < 0.05 else "")


def _ratio_axis(ax, lo, hi, ticks):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axvline(1, color="#999999", lw=0.6, zorder=0)


def forest(ax, t: pd.DataFrame, effect: str, xlabel: str, lims, ticks, labels: bool):
    """各终点（行）× 细胞类型（同一行内上下错开）的比值与 95% 置信区间；右侧标注 BH 校正后的显著性。"""
    for k, ep in enumerate(EPS):
        for ct, dy in (("Beta", -0.17), ("Alpha", 0.17)):
            r = t[(t.cell_type == ct) & (t.endpoint == ep) & (t.effect == effect)]
            if r.empty or not np.isfinite(r.estimate.iloc[0]):
                continue
            r = r.iloc[0]
            ax.errorbar(r.estimate, k + dy, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", ms=3,
                        color=TYPE_COLORS[ct], elinewidth=0.8, capsize=0,
                        label=f"{TYPE_SYM[ct]} cells" if k == 0 else None)
            s = _stars(r.p_bh)
            if s:
                ax.text(lims[1] * 0.97, k + dy, s, ha="right", va="center", fontsize=6.5, color=TYPE_COLORS[ct])
    _ratio_axis(ax, *lims, ticks)
    ax.set_yticks(range(len(EPS)))
    ax.set_yticklabels([SHORT[e] for e in EPS] if labels else [])
    ax.set_ylim(len(EPS) - 0.5, -0.5)
    ax.set_xlabel(xlabel)


def shape(ax, t: pd.DataFrame):
    """上尾增强（q90 比值 / q50 比值，正值部分）与供体内离散度之比；=1 即等比例变化。"""
    for k, ep in enumerate(EPS):
        for ct, dy in (("Beta", -0.17), ("Alpha", 0.17)):
            for eff, mk, off in (("tail_pos_ratio", "o", -0.07), ("sd_ratio", "D", 0.07)):
                r = t[(t.cell_type == ct) & (t.endpoint == ep) & (t.effect == eff)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.estimate, k + dy + off, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt=mk,
                            ms=2.6, mfc=TYPE_COLORS[ct] if mk == "o" else "white", color=TYPE_COLORS[ct],
                            elinewidth=0.7, capsize=0, mew=0.7)
    _ratio_axis(ax, 0.4, 2.5, [0.5, 1, 2])
    ax.set_yticks(range(len(EPS)))
    ax.set_yticklabels([])
    ax.set_ylim(len(EPS) - 0.5, -0.5)
    ax.set_xlabel("T2D / ND ratio")
    ax.errorbar([], [], fmt="o", ms=2.6, color="#555555", label="P90 / P50 effect")
    ax.errorbar([], [], fmt="D", ms=2.6, mfc="white", color="#555555", mew=0.7, label="Within-donor SD")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6, handletextpad=0.2,
              borderpad=0.3)


PROFILE_EPS = {"exocytosis": "#B2182B", "early_exocytosis": "#E7877B", "late_exocytosis": "#7F0000",
               "calcium_entry": "#E08214", "sodium_current": "#2C6FB0"}


def profile(ax, t: pd.DataFrame, qs):
    """β 细胞正值部分各百分位的 T2D/ND 比（每位供体等权）。"""
    b = t[t.cell_type == "Beta"]
    for k, (ep, col) in enumerate(PROFILE_EPS.items()):
        r = b[b.endpoint == ep].set_index("effect")
        est = np.array([r.loc[f"q{int(q * 100)}_pos_ratio", "estimate"] for q in qs])
        lo = np.array([r.loc[f"q{int(q * 100)}_pos_ratio", "lo"] for q in qs])
        hi = np.array([r.loc[f"q{int(q * 100)}_pos_ratio", "hi"] for q in qs])
        x = np.arange(len(qs)) + (k - 2) * 0.09
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="-o", ms=2.6, lw=0.9, elinewidth=0.6, color=col,
                    label=SHORT[ep])
    ax.set_yscale("log")
    ax.set_ylim(0.25, 6.5)
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(["0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.axhline(1, color="#999999", lw=0.6, zorder=0)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"P{int(q * 100)}" for q in qs])
    ax.set_xlabel("Percentile of positive responses")
    ax.set_ylabel("β cells, T2D / ND ratio")
    ax.legend(loc="upper left", fontsize=6, ncol=2, columnspacing=0.6, handlelength=1.2, handletextpad=0.3,
              borderpad=0.3)


ROBUST_ORDER = ["all donors", "ADI IsletCore donors", "HPAP donors", "glucose 1", "glucose 5", "glucose 10",
                "adjusted for age, sex and BMI", "clinically diagnosed T2D only",
                "ADI IsletCore, adjusted for isolation and culture", "cells weighted equally"]
ROBUST_SHORT = ["All donors", "IsletCore only", "HPAP only", "1 mM glucose", "5 mM glucose", "10 mM glucose",
                "+ age, sex, BMI", "Clinical T2D", "IsletCore + isolation", "Cells equal weight"]


def robustness(ax, t: pd.DataFrame, rob: pd.DataFrame, glu: pd.DataFrame) -> pd.DataFrame:
    """β 细胞正值反应几何均值之比（总胞吐与 Na⁺ 电流）在不同供体子集、葡萄糖浓度与调整下的估计。"""
    g = glu.assign(analysis="glucose " + glu.glucose.astype(str))
    rob = rob[rob.analysis != "5 mM glucose only"]
    allr = pd.concat([t[t.cell_type == "Beta"], rob[rob.cell_type == "Beta"], g], ignore_index=True)
    used = []
    for ep, col, dy in (("exocytosis", "#B2182B", -0.15), ("sodium_current", "#2C6FB0", 0.15)):
        for k, an in enumerate(ROBUST_ORDER):
            r = allr[(allr.analysis == an) & (allr.endpoint == ep) & (allr.effect == "positive_ratio")]
            if r.empty:
                continue
            r = r.iloc[0]
            used.append(r)
            ax.errorbar(r.estimate, k + dy, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", ms=2.8,
                        color=col, elinewidth=0.8, capsize=0, label=SHORT[ep] if k == 0 else None)
    _ratio_axis(ax, 0.3, 3.0, [0.5, 1, 2])
    ax.set_yticks(range(len(ROBUST_ORDER)))
    ax.set_yticklabels(ROBUST_SHORT, fontsize=6)
    ax.set_ylim(len(ROBUST_ORDER) - 0.5, -0.5)
    ax.set_xlabel("β cells, ratio of positive-response\ngeometric means, T2D / ND")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handletextpad=0.2,
              borderpad=0.3, columnspacing=0.8)
    return pd.DataFrame(used)


CA_ROWS = [("ca_t2d", "Ca$^{2+}$ entry"), ("total", "Exocytosis"), ("direct_ca_lam1", "+ Ca$^{2+}$"),
           ("direct_ca_lam0.5", "+ Ca$^{2+}$, reliab. 0.5"),
           ("direct_ca_lam0.2", "+ Ca$^{2+}$, reliab. 0.2"), ("direct_ca_between", "+ donor-mean Ca$^{2+}$")]
CA_EPS = [("exocytosis", "Total exo.", "#B2182B", "o", None), ("late_exocytosis", "Late exo.", "#7F0000", "s", "white")]


def calcium(ax, ca: pd.DataFrame, el: pd.DataFrame) -> pd.DataFrame:
    """β 细胞正值反应：同一组细胞中 Ca²⁺ 进入的 T2D 比、胞吐的 T2D 比，及按 Ca²⁺ 进入（含测量误差校正与供体均值）调整后的比；
    末行为晚期与早期 T2D 比之比（同一供体抽样）。"""
    rows = []
    for k, (ep, lab, col, mk, fc) in enumerate(CA_EPS):
        s = ca[(ca.endpoint == ep) & (ca.part == "positive responses")].set_index("quantity")
        for i, (q, _) in enumerate(CA_ROWS):
            r = s.loc[q]
            y = i + (k - 0.5) * 0.3
            ax.plot(np.exp([r.lo, r.hi]), [y, y], color=col, lw=0.8)
            ax.plot(np.exp(r.estimate), y, mk, ms=3.0, mfc=col if fc is None else fc, mec=col, mew=0.8,
                    label=lab if i == 0 else None)
            rows.append({"endpoint": ep, "quantity": q, "ratio": np.exp(r.estimate), "lo": np.exp(r.lo),
                         "hi": np.exp(r.hi), "n_cells": r.n_cells, "n_donors": r.n_donors})
    r = el.set_index("quantity").loc["late_minus_early"]
    y = len(CA_ROWS)
    ax.plot(np.exp([r.lo, r.hi]), [y, y], color="#555555", lw=0.8)
    ax.plot(np.exp(r.estimate), y, "D", ms=3.0, color="#555555")
    ax.text(1.12, y, f"P = {r.p_boot_two_sided:.3f}", va="center", ha="left", fontsize=6)
    rows.append({"endpoint": "late / early", "quantity": "late_minus_early", "ratio": np.exp(r.estimate),
                 "lo": np.exp(r.lo), "hi": np.exp(r.hi), "p_boot": r.p_boot_two_sided})
    _ratio_axis(ax, 0.3, 2.0, [0.5, 1, 2])
    ax.set_yticks(range(len(CA_ROWS) + 1))
    ax.set_yticklabels([lab for _, lab in CA_ROWS] + ["Late / early"], fontsize=6)
    ax.set_ylim(len(CA_ROWS) + 0.5, -0.5)
    ax.set_xlabel("β cells, T2D / ND ratio,\npositive responses")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handletextpad=0.2,
              borderpad=0.3, columnspacing=0.8)
    return pd.DataFrame(rows)


def shares(ax, t: pd.DataFrame):
    """β 细胞总胞吐：高于 ND 上分位数的细胞比例——ND、T2D 实测，与每个正值反应都按估计比值降低时 T2D 的预期比例。"""
    b = t[(t.cell_type == "Beta") & (t.endpoint == "exocytosis")].set_index("effect")
    cols = [("nd", STATUS_COLORS["ND"], "ND observed", None), ("t2d", STATUS_COLORS["T2D"], "T2D observed", None),
            ("expected", "white", "T2D, expected if every\npositive response were\nreduced by the\nestimated ratio", STATUS_COLORS["T2D"])]
    for g, thr in enumerate((75, 90)):
        for k, (key, fc, lab, ec) in enumerate(cols):
            r = b.loc[f"share{thr}_{key}"]
            x = g * 1.2 + (k - 1) * 0.3
            ax.bar(x, r.estimate * 100, width=0.27, color=fc, edgecolor=ec or fc, lw=0.8, hatch="////" if ec else None,
                   label=lab if g == 0 else None)
            ax.errorbar(x, r.estimate * 100, yerr=[[(r.estimate - r.lo) * 100], [(r.hi - r.estimate) * 100]],
                        fmt="none", ecolor="#333333", elinewidth=0.6)
        ex = b.loc[f"share{thr}_excess"]
        top = b.loc[[f"share{thr}_{k}" for k in ("nd", "t2d", "expected")], "hi"].max() * 100
        ax.text(g * 1.2, top + 2.0, f"Observed vs\nexpected:\nP = {ex.p_perm:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1.2])
    ax.set_xticklabels(["Above\nND P75", "Above\nND P90"])
    ax.set_ylabel("β cells, total exocytosis (%)")
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlim(-0.85, 2.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.03, 1.03), fontsize=6, handletextpad=0.3, borderpad=0.3,
              frameon=False)


def coupling(ax, d: pd.DataFrame, margin: float):
    """只用 ND 供体训练的模型：T2D − ND 供体内精度之差（点、95% 与 90% 区间；阴影为 ±等效性界值）。"""
    d = d[d.quantity == "diff_T2D"]
    order = [("Beta", e) for e in ("sodium_current", "size", "calcium_entry", "late_ca_current")] + \
        [("Alpha", e) for e in ("sodium_current", "size", "early_ca_current", "late_ca_current", "exocytosis",
                                "early_exocytosis")]
    ax.axhspan(-margin, margin, color="#EEEEEE", zorder=0, lw=0)
    ax.axhline(0, color="#999999", lw=0.6)
    for x, (ct, ep) in enumerate(order):
        r = d[(d.cell_type == ct) & (d.endpoint == ep)].iloc[0]
        ax.plot([x, x], [r.lo, r.hi], color=TYPE_COLORS[ct], lw=0.7)
        ax.plot([x, x], [r.lo90, r.hi90], color=TYPE_COLORS[ct], lw=2.0, solid_capstyle="butt")
        ax.plot(x, r.estimate, "o", ms=3.2, color="white", mec=TYPE_COLORS[ct], mew=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{TYPE_SYM[c]} {SHORT[e]}" for c, e in order], rotation=35, ha="right", fontsize=6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.42, 0.32)
    ax.set_ylabel("Within-donor r,\nT2D − ND")
