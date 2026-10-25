"""Supplementary Fig. S5：外部供体中的绝对尺度与跨平台迁移（a、b 的排序迁移见 Fig. 3c、d，函数 panel_ext 供其调用）。只读结果表。

a  外部供体的 R²，以训练集均值（可部署的常数基线）为参照；0 = 与直接报告训练均值相同。灰色短横线为开发交叉验证。
b  外部供体的校准斜率（实测值对预测值回归；1 = 绝对尺度校准）。
c  供体层面跨平台比较，只用两种输入都有的供体：上为 Patch-seq 与液滴汇总表达谱的 Spearman（配对连线），
   下为配对差值（Patch-seq − 液滴）及按供体配对重抽样 95% 置信区间。
d  示例：α 细胞早期 Ca²⁺ 电流的供体平均实测值与预测值（Patch-seq 汇总表达谱）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (MM, MODEL_COLORS, TYPE_COLORS, model_label, apply_style, panel_label, save_fig,  # noqa: E402
                      save_source)
from Fig2_variance import ORDER, SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEY = [(ct, ep) for ct in ("Beta", "Alpha") for ep in ORDER]
MODS = ["knn_predictive_set", "ridge", "gradient_boosting", "context_model"]
CAL_MODS = ["context_model", "cell_branch"]     # 外部应用的模型（含供体参照）与不含供体参照的集成


def load():
    ext = pd.concat([pd.read_csv(ROOT / "results/06_external_hpap" / f"metrics_{ct}.tsv", sep="\t")
                     for ct in ("Beta", "Alpha")])
    ceil = pd.read_csv(ROOT / "results/06b_external_ceiling/metrics_internal.tsv", sep="\t")
    ceil = ceil[ceil.model == "context_model"].groupby(["cell_type", "endpoint"])[
        ["cell_spearman", "within_pearson"]].mean().reset_index()
    dl = pd.read_csv(ROOT / "results/06c_external_donor_level/donor_level_summary.tsv", sep="\t")
    dp = pd.read_csv(ROOT / "results/06c_external_donor_level/donor_level_predictions.tsv", sep="\t")
    cal = ROOT / "results/06d_calibration"
    ce = pd.read_csv(cal / "external_calibration.tsv", sep="\t")
    cd = pd.read_csv(cal / "development_calibration.tsv", sep="\t")
    cal_tab = pd.concat([ce, cd])
    cal_tab = cal_tab[cal_tab.model.isin(CAL_MODS)]
    xp = pd.read_csv(cal / "crossplatform_common_donors.tsv", sep="\t")
    return ext, ceil, dl, dp, cal_tab, xp


def panel_ext(ax, ext, ceil, met, lev, ylabel):
    x = 0
    ticks, labs = [], []
    for ct, ep in KEY:
        for k, mod in enumerate(MODS):
            r = ext[(ext.cell_type == ct) & (ext.endpoint == ep) & (ext.model == mod)].iloc[0]
            xx = x + (k - 1.5) * 0.19
            ax.errorbar([xx], [r[met]], yerr=[[r[met] - r[f"{lev}_lo"]], [r[f"{lev}_hi"] - r[met]]], fmt="o",
                        ms=2.8, color=MODEL_COLORS[mod], elinewidth=0.7, capsize=0)
        c = ceil[(ceil.cell_type == ct) & (ceil.endpoint == ep)][met].iloc[0]
        ax.plot([x - 0.42, x + 0.42], [c, c], color="#222222", lw=1.0, ls="--", zorder=0)
        ticks.append(x); labs.append(SHORT[ep])
        x += 1
        if ep == ORDER[-1] and ct == "Beta":
            x += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=6)
    ax.set_xlim(-0.6, x - 0.4)
    for ct, xc in (("Beta", 4), ("Alpha", 13.6)):
        ax.text(xc, 0.71, "β cells" if ct == "Beta" else "α cells", ha="center", va="top", fontsize=7,
                color=TYPE_COLORS[ct])
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.3, 0.72)


def _strip_x(ax, groups=True):
    """c–e 的横轴：β 与 α 各 9 个终点，中间留空。返回 {(细胞类型, 终点): x}。"""
    xs, x = {}, 0.0
    for ct, ep in KEY:
        xs[(ct, ep)] = x
        x += 1
        if ep == ORDER[-1] and ct == "Beta":
            x += 0.6
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([SHORT[ep] for _, ep in KEY], rotation=50, ha="right", fontsize=6)
    ax.set_xlim(-0.6, x - 0.4)
    tr = ax.get_xaxis_transform()
    for ct, xc in (("Beta", 4), ("Alpha", 13.6)) if groups else ():
        ax.text(xc, 1.02, "β cells" if ct == "Beta" else "α cells", transform=tr, ha="center", va="bottom",
                fontsize=6, color=TYPE_COLORS[ct])
    return xs


def panel_cal(ax, cal, met, ylabel, refs, groups=True):
    """外部供体的校准指标（两种集成模型，95% 置信区间）与开发交叉验证参照（灰色短横线，含供体参照的集成）。"""
    xs = _strip_x(ax, groups)
    for ref in refs:
        ax.axhline(ref, color="#999999", lw=0.6, ls=":" if ref == 0 and len(refs) > 1 else "-", zorder=0)
    ext = cal[cal.set == "external"].set_index(["cell_type", "endpoint", "model"])
    dev = cal[cal.set == "development"].set_index(["cell_type", "endpoint", "model"])
    for (ct, ep), x in xs.items():
        v = dev.loc[(ct, ep, "context_model"), met]
        ax.plot([x - 0.38, x + 0.38], [v, v], color="#9A9A9A", lw=1.1, zorder=1, solid_capstyle="butt")
        for k, mod in enumerate(CAL_MODS):
            r = ext.loc[(ct, ep, mod)]
            ax.errorbar([x + (k - 0.5) * 0.3], [r[met]], yerr=[[r[met] - r[f"{met}_lo"]], [r[f"{met}_hi"] - r[met]]],
                        fmt="o", ms=2.6, color=MODEL_COLORS[mod], elinewidth=0.7, capsize=0, zorder=3)
    ax.set_ylabel(ylabel)


def panel_common(ax, ax2, xp):
    """共同供体上的跨平台比较：上轴两种输入的 Spearman（配对连线），下轴配对差值及 95% 置信区间。"""
    xs = _strip_x(ax)
    _strip_x(ax2, groups=False)
    for (ct, ep), x in xs.items():
        r = xp[(xp.cell_type == ct) & (xp.endpoint == ep)].iloc[0]
        ax.plot([x - 0.14, x + 0.14], [r.rho_patchseq, r.rho_droplet], color="#BBBBBB", lw=0.8, zorder=1)
        ax.plot([x - 0.14], [r.rho_patchseq], "o", ms=3.0, color=TYPE_COLORS[ct], zorder=3)
        ax.plot([x + 0.14], [r.rho_droplet], "s", ms=2.8, mfc="white", mec="#666666", mew=0.8, zorder=3)
        ax2.errorbar([x], [r["diff"]], yerr=[[r["diff"] - r.diff_lo], [r.diff_hi - r["diff"]]], fmt="o", ms=2.8,
                     color=TYPE_COLORS[ct], elinewidth=0.7, capsize=0)
    for a_ in (ax, ax2):
        a_.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.set_ylim(-0.85, 1.0)
    ax2.set_ylim(-1.6, 1.75)
    ax.set_ylabel("Donor-level ρ")
    ax2.set_ylabel("Δρ, Patch-seq\nminus droplet")
    nb, na = (int(xp[xp.cell_type == c].n_common_donors.max()) for c in ("Beta", "Alpha"))
    h1 = ax.plot([], [], "o", ms=3, color="#444444", label="Patch-seq profiles")[0]
    h2 = ax.plot([], [], "s", ms=3, mfc="white", mec="#666666", label="Droplet profiles")[0]
    h3 = ax.plot([], [], " ", label=f"({nb} β / {na} α donors with both)")[0]
    ax.legend(handles=[h1, h2, h3], loc="lower left", bbox_to_anchor=(0.0, 1.1), fontsize=6, ncol=3, frameon=False,
              columnspacing=1.2, handletextpad=0.3)


def panel_scatter(ax, dp):
    s = dp[(dp.cell_type == "Alpha") & (dp.endpoint == "early_ca_current") & (dp.input == "Patch-seq profile")]
    ax.scatter(s.predicted, s.measured, s=10 + s.n_cells / 3, color=TYPE_COLORS["Alpha"], edgecolor="white", lw=0.4)
    rho = spearmanr(s.predicted, s.measured).statistic
    ax.text(0.04, 0.96, f"ρ = {rho:.2f}\n{len(s)} donors\npoint size: cells per donor", transform=ax.transAxes,
            ha="left", va="top", fontsize=6)
    ax.set_xlabel("Predicted donor mean\n(SD units)")
    ax.set_ylabel("Measured donor mean (SD units)")
    ax.set_title("α early Ca$^{2{+}}$ current", fontsize=7)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    return s


def main() -> None:
    apply_style()
    ext, ceil, dl, dp, cal, xp = load()
    save_source(cal, ROOT, "FigS5", "ab_calibration")
    save_source(xp, ROOT, "FigS5", "c_common_donors")
    fig = plt.figure(figsize=(170 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.62)
    gcd = gs[0].subgridspec(2, 1, hspace=0.32)
    c1 = fig.add_subplot(gcd[0]); c2 = fig.add_subplot(gcd[1])
    panel_cal(c1, cal, "r2_train_mean", "R² relative to\ntraining mean", [0])
    panel_cal(c2, cal, "calib_slope", "Calibration\nslope", [0, 1], groups=False)
    c1.tick_params(labelbottom=False)
    hc = [c1.errorbar([], [], fmt="o", ms=3, color=MODEL_COLORS[m], label=model_label(m, short=True))
          for m in CAL_MODS]
    hc.append(c1.plot([], [], color="#9A9A9A", lw=1.1, label="Development donors, cross-validation "
                      "(Ensemble + donor ref.)")[0])
    c1.legend(handles=hc, loc="lower left", bbox_to_anchor=(0.0, 1.1), fontsize=6, ncol=3, columnspacing=1.2,
              handletextpad=0.3, frameon=False)
    gef = gs[1].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.32)
    gee = gef[0, 0].subgridspec(2, 1, hspace=0.14)
    e1 = fig.add_subplot(gee[0]); e2 = fig.add_subplot(gee[1])
    f = fig.add_subplot(gef[0, 1])
    panel_common(e1, e2, xp)
    e1.tick_params(labelbottom=False)
    s = panel_scatter(f, dp)
    save_source(s, ROOT, "FigS5", "d_alpha_early_ca")
    fig._col_tol = 0.15
    for ax, l in zip((c1, c2, e1, f), "abcd"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "FigS5")


if __name__ == "__main__":
    main()
