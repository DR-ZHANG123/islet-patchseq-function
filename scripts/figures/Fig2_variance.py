"""Fig. 2：电生理变异主要发生在供体内部；按细胞随机划分会高估预测精度。只读结果表。

a  每位供体的 Patch-seq 细胞数（β、α），按糖代谢状态着色。
b  各终点的组内相关（ICC）及 95% 置信区间。
c  β 细胞终点之间的供体内相关。
d–e  按细胞随机划分与按供体留出两种评价下的精度（细胞层面与供体均值层面）。
f  供体层面相关的置换零分布（供体平均偏移在供体间互换，完整交叉验证流程重跑 100 次）：岭回归与 Ensemble 的零分布
   均值与 95% 范围及实测值（β 20 次、α 5 次重复的均值）。留出供体时零分布均值系统性低于 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ENDPOINT_LABELS, MM, MODEL_COLORS, model_label, STATUS_COLORS, STATUS_LABELS,  # noqa: E402
                      TYPE_COLORS, apply_style, panel_label, save_fig, save_source)

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["size", "sodium_current", "na_half_inactivation", "early_ca_current", "late_ca_current", "calcium_entry",
         "exocytosis", "early_exocytosis", "late_exocytosis"]
SHORT = {"size": "Size", "sodium_current": "Na$^{{+}}$ peak", "na_half_inactivation": "Na$^{{+}}$ V$_{1/2}$",
         "early_ca_current": "Early Ca$^{2{+}}$", "late_ca_current": "Late Ca$^{2{+}}$", "calcium_entry": "Ca$^{2{+}}$ entry",
         "exocytosis": "Total exo.", "early_exocytosis": "Early exo.", "late_exocytosis": "Late exo."}


def panel_cells(ax):
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    cells = cells[cells.qc_pass]
    d = cells.groupby(["cell_type", "donor", "status"]).size().rename("n").reset_index()
    save_source(d, ROOT, "Fig2", "a_cells_per_donor")
    rng = np.random.default_rng(1)
    for i, ct in enumerate(("Beta", "Alpha")):
        s = d[d.cell_type == ct]
        for st in ("ND", "preT2D", "T2D", "T1D"):
            v = s[s.status == st].n.values
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=7, color=STATUS_COLORS[st], alpha=0.85,
                       lw=0, label=STATUS_LABELS[st] if i == 0 else None)
        ax.plot([i - 0.33, i + 0.33], [np.median(s.n)] * 2, color="#222222", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["β cells", "α cells"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Cells per donor")
    ax.legend(loc="upper left", fontsize=6, markerscale=1.2, handletextpad=0.1, ncol=1, borderpad=0.3,
              labelspacing=0.25)
    ax.set_ylim(0.7, 6000)


def panel_icc(ax):
    icc = pd.read_csv(ROOT / "results/02_variance/icc.tsv", sep="\t")
    save_source(icc, ROOT, "Fig2", "b_icc")
    y = np.arange(len(ORDER))
    for j, ct in enumerate(("Beta", "Alpha")):
        s = icc[icc.cell_type == ct].set_index("endpoint").loc[ORDER]
        off = -0.17 if ct == "Beta" else 0.17
        ax.errorbar(s.icc, y + off, xerr=[s.icc - s.icc_lo, s.icc_hi - s.icc], fmt="o", ms=3.5, color=TYPE_COLORS[ct],
                    elinewidth=0.8, capsize=0, label="β cells" if ct == "Beta" else "α cells")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[e] for e in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Between-donor share of variance (ICC)")
    ax.legend(loc="lower right", fontsize=6)


def panel_corr(ax):
    c = pd.read_csv(ROOT / "results/02_variance/within_endpoint_correlation.tsv", sep="\t")
    c = c[c.cell_type == "Beta"].pivot(index="a", columns="b", values="within_r").loc[ORDER, ORDER]
    save_source(c.reset_index(), ROOT, "Fig2", "c_within_correlation_beta")
    # 只画下三角：行 = 第 2–9 个终点，列 = 第 1–8 个终点
    v = c.values[1:, :-1]
    m = np.where(np.tril(np.ones_like(v, dtype=bool)), v, np.nan)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(v.shape[0]):
        for j in range(i + 1):
            ax.text(j, i, f"{v[i, j]:.2f}".replace("0.", ".").replace("-", "−"), ha="center", va="center",
                    fontsize=6, color="white" if abs(v[i, j]) > 0.6 else "#222222")._overlap_exempt = True
    ax.set_xticks(range(len(ORDER) - 1))
    ax.set_xticklabels([SHORT[e] for e in ORDER[:-1]], rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(ORDER) - 1)); ax.set_yticklabels([SHORT[e] for e in ORDER[1:]], fontsize=6)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Within-donor r (β cells)", fontsize=6); cb.ax.tick_params(labelsize=6)


def panel_split(ax_c, ax_d):
    m = pd.read_csv(ROOT / "results/03c_summary/metrics_cv.tsv", sep="\t")
    eps = ["size", "sodium_current", "late_ca_current"]
    models = ["knn_predictive_set", "ridge", "context_model"]
    s = m[m.endpoint.isin(eps) & m.model.isin(models)]
    save_source(s, ROOT, "Fig2", "de_split_comparison")
    for ax, met, lab in ((ax_c, "cell_spearman", "Cell-level ρ"), (ax_d, "donor_spearman", "Donor-mean ρ")):
        x = 0
        ticks, labels = [], []
        for ct in ("Beta", "Alpha"):
            for ep in eps:
                for k, mod in enumerate(models):
                    r = s[(s.cell_type == ct) & (s.endpoint == ep) & (s.model == mod)].set_index("split")[met]
                    xx = x + (k - 1) * 0.24
                    ax.plot([xx, xx], [r["donors"], r["cells"]], color=MODEL_COLORS[mod], lw=0.9)
                    ax.scatter([xx], [r["cells"]], s=12, facecolor="white", edgecolor=MODEL_COLORS[mod], lw=0.9, zorder=3)
                    ax.scatter([xx], [r["donors"]], s=12, color=MODEL_COLORS[mod], zorder=3)
                ticks.append(x); labels.append(f"{'β' if ct == 'Beta' else 'α'} {SHORT[ep]}")
                x += 1
            x += 0.4
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(lab)
        ax.axhline(0, color="#999999", lw=0.5, ls=":")
        ax.set_ylim(-0.45 if met == "donor_spearman" else -0.05, 1.05)
    hs = [ax_c.scatter([], [], s=12, color=MODEL_COLORS[mm], label=model_label(mm, short=True)) for mm in models]
    hs += [ax_c.scatter([], [], s=12, color="#555555", label="Donors held out"),
           ax_c.scatter([], [], s=12, facecolor="white", edgecolor="#555555", label="Cells split at random")]
    ax_d.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6, frameon=True)


NULL_MODELS = ["ridge", "cell_branch"]


def panel_null(ax):
    c = pd.read_csv(ROOT / "results/04c_donor_null/calibration.tsv", sep="\t")
    c = c[(c.null == "donor") & (c.grid == "default") & (c.level == "donor_spearman") & c.model.isin(NULL_MODELS)]
    keep = ["cell_type", "endpoint", "model", "n_obs_repeats", "observed", "null_mean", "null_q025", "null_q975",
            "p_perm", "n_perm"]
    save_source(c[keep], ROOT, "Fig2", "f_donor_null")
    x = 0
    ticks, labs = [], []
    for ct in ("Beta", "Alpha"):
        for ep in ORDER:
            for k, mod in enumerate(NULL_MODELS):
                r = c[(c.cell_type == ct) & (c.endpoint == ep) & (c.model == mod)].iloc[0]
                xx = x + (k - 0.5) * 0.36
                ax.plot([xx, xx], [r.null_q025, r.null_q975], color="#C8C8C8", lw=3.2, solid_capstyle="butt",
                        zorder=1)
                ax.plot([xx - 0.13, xx + 0.13], [r.null_mean] * 2, color="#555555", lw=0.9, zorder=2)
                ax.scatter([xx], [r.observed], s=13, color=MODEL_COLORS[mod], zorder=3, edgecolor="white", lw=0.3)
            ticks.append(x); labs.append(SHORT[ep])
            x += 1
        x += 0.6
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
    ax.set_xlim(-0.6, x - 0.9)
    ax.set_ylim(-0.75, 1.05)
    ax.set_ylabel("Donor-level ρ (donors held out)")
    for ct, x0 in (("Beta", 4), ("Alpha", 4 + 9.6)):
        ax.text(x0, 1.0, "β cells" if ct == "Beta" else "α cells", ha="center", va="top", fontsize=7,
                color=TYPE_COLORS[ct])
    hs = [ax.scatter([], [], s=13, color=MODEL_COLORS[m], label=f"{model_label(m, short=True)}, observed")
          for m in NULL_MODELS]
    hs.append(ax.plot([], [], color="#C8C8C8", lw=3.2, label="Permutation null, 95% range")[0])
    hs.append(ax.plot([], [], color="#555555", lw=0.9, label="Permutation null, mean")[0])
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=6, frameon=False,
              handletextpad=0.6, handlelength=1.6, columnspacing=1.6)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(170 * MM, 200 * MM))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.95, 0.85], hspace=0.62, wspace=0.55, width_ratios=[0.95, 1, 1.3])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    gs2 = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.35)
    d, e = fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])
    f = fig.add_subplot(gs[2, :])
    panel_cells(a); panel_icc(b); panel_corr(c); panel_split(d, e); panel_null(f)
    for ax, l in zip((a, b, c, d, e, f), "abcdef"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "Fig2")


if __name__ == "__main__":
    main()
