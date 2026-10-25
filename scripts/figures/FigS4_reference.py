"""补充图 S3：供体参照。只读结果表。

a–b  参照细胞数（1、2、3、5、10、全部）与换用其他供体参照时，供体参照单独给出的供体层面 Spearman ρ（β、α；
     results/04_context_tests，5 次重复均值）。
c    全部终点：本供体参照与其他供体参照的配对比较。
d    加入供体参照带来的供体层面 ρ 增益（β 20 次、α 5 次交叉验证重复；按供体重抽样 95% 置信区间；results/04b），
     分别相对 Ridge (within-donor term) 与 Ensemble；灰条 = 置换零分布下同一增益的 95% 范围（results/04c）。
e    留一参照（参照均值不含细胞自身）减标准参照的供体层面 ρ 差（Ensemble + donor ref.，前 5 次重复）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import ENDPOINT_LABELS, MM, TYPE_COLORS, apply_style, panel_label, save_fig, save_source  # noqa: E402
from Fig2_variance import ORDER, SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SIZES = ["k1", "k2", "k3", "k5", "k10", "full"]
XT = ["1", "2", "3", "5", "10", "All"]
KEY = {"Beta": ["sodium_current", "size", "early_exocytosis"],
       "Alpha": ["sodium_current", "size", "early_ca_current", "late_ca_current"]}
LINE = ["#B2182B", "#2C6FB0", "#1A9850", "#8E6BB0"]
GAINS = [("linear_context_model", "linear_cell_branch", "Ridge (within-donor term)"),
         ("context_model", "cell_branch", "Ensemble")]
GAIN_COL = {"linear_cell_branch": "#2C6FB0", "cell_branch": "#B2182B"}


def load_context():
    rows = []
    for ct in ("Beta", "Alpha"):
        m = pd.read_csv(ROOT / "results/04_context_tests" / f"context_tests_{ct}.tsv", sep="\t")
        m = m[m.component == "donor_branch"].groupby(["endpoint", "context"]).donor_spearman.agg(["mean", "std"])
        rows.append(m.reset_index().assign(cell_type=ct))
    return pd.concat(rows)


def panel_size(ax, d, ct):
    for i, ep in enumerate(KEY[ct]):
        s = d[(d.cell_type == ct) & (d.endpoint == ep)].set_index("context")
        ax.plot(range(len(SIZES)), s.loc[SIZES, "mean"].values, "-o", color=LINE[i], ms=3.2, lw=1.0,
                label=ENDPOINT_LABELS[ep])
        off = 0.16 * (i - (len(KEY[ct]) - 1) / 2)
        ax.plot([len(SIZES) + 0.7 + off], [s.loc["mismatched", "mean"]], "D", color=LINE[i], ms=3.8, mfc="white",
                mew=1.0)
    ax.set_xticks(list(range(len(SIZES))) + [len(SIZES) + 0.7])
    ax.set_xticklabels(XT + ["Another\ndonor"])
    ax.set_xlim(-0.4, len(SIZES) + 1.4)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xlabel("Reference cells per donor")
    ax.set_ylabel("Donor-level ρ")
    ax.set_title("β cells, reference alone" if ct == "Beta" else "α cells, reference alone", fontsize=8, color=TYPE_COLORS[ct])
    ax.set_ylim(-0.8, 0.8)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handlelength=1.4, handletextpad=0.4,
              borderaxespad=0.3, labelspacing=0.3)


def panel_paired(ax, d):
    xs, labels = [], []
    pos = 0
    for ct in ("Beta", "Alpha"):
        s = d[d.cell_type == ct].pivot(index="endpoint", columns="context", values="mean").loc[ORDER]
        for ep, r in s.iterrows():
            ax.plot([pos - 0.12, pos + 0.12], [r["full"], r["mismatched"]], color="#BBBBBB", lw=0.8, zorder=1)
            ax.scatter([pos + 0.12], [r["mismatched"]], s=14, facecolor="white", edgecolor=TYPE_COLORS[ct], zorder=3,
                       lw=0.9)
            ax.scatter([pos - 0.12], [r["full"]], s=14, color=TYPE_COLORS[ct], zorder=4, edgecolor="white", lw=0.4)
            xs.append(pos); labels.append(SHORT[ep])
            pos += 1
        pos += 0.8
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xlim(-0.6, pos - 1.2)
    ax.set_ylim(-0.6, 0.95)
    ax.set_ylabel("Donor-level ρ,\nreference alone")
    for ct, xc in (("Beta", 4), ("Alpha", 13.8)):
        ax.text(xc, 0.93, "β cells" if ct == "Beta" else "α cells", ha="center", va="top", fontsize=7,
                color=TYPE_COLORS[ct])
    h1 = ax.scatter([], [], s=14, color="#555555", label="Own donor's cells")
    h2 = ax.scatter([], [], s=14, facecolor="white", edgecolor="#555555", label="Another donor's cells")
    ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.5, frameon=False)


def x_layout():
    pos, x = [], 0.0
    for ct in ("Beta", "Alpha"):
        for ep in ORDER:
            pos.append((ct, ep, x)); x += 1
        x += 0.8
    return pos, x


def endpoint_axis(ax, ylabel):
    pos, x = x_layout()
    ax.set_xticks([p for *_, p in pos]); ax.set_xticklabels([SHORT[ep] for _, ep, _ in pos], rotation=40, ha="right",
                                                          fontsize=6.5)
    ax.set_xlim(-0.6, x - 1.2)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_ylabel(ylabel)
    return pos


def panel_gain(ax):
    p = pd.read_csv(ROOT / "results/04b_robustness/paired_differences.tsv", sep="\t")
    p = p[(p.comparison == "standard_all_repeats") & (p.level == "donor")]
    c = pd.read_csv(ROOT / "results/04c_donor_null/calibration_differences.tsv", sep="\t")
    c = c[(c["null"] == "donor") & (c.grid == "default") & (c.level == "donor_spearman")]
    pos = endpoint_axis(ax, "Gain in donor-level ρ\nfrom the donor reference")
    rows = []
    for ct, ep, x in pos:
        for k, (a, b, name) in enumerate(GAINS):
            xx = x + (k - 0.5) * 0.34
            r = p[(p.cell_type == ct) & (p.endpoint == ep) & (p.model_a == a) & (p.model_b == b)].iloc[0]
            n = c[(c.cell_type == ct) & (c.endpoint == ep) & (c.model_a == a) & (c.model_b == b)].iloc[0]
            ax.plot([xx, xx], [n.null_q025, n.null_q975], color="#D5D5D5", lw=3.0, solid_capstyle="butt", zorder=1)
            ax.errorbar([xx], [r.difference], yerr=[[r.difference - r.lo], [r.hi - r.difference]], fmt="o", ms=2.6,
                        color=GAIN_COL[b], elinewidth=0.7, capsize=0, zorder=3)
            rows.append({"cell_type": ct, "endpoint": ep, "relative_to": name, "gain": r.difference, "lo": r.lo,
                         "hi": r.hi, "p_boot": r.p_boot, "null_mean": n.null_mean, "null_q025": n.null_q025,
                         "null_q975": n.null_q975, "p_perm": n.p_perm})
    ax.set_ylim(-0.75, 0.95)
    hs = [ax.errorbar([], [], fmt="o", ms=2.8, color=GAIN_COL[b], label=f"Relative to {name}") for _, b, name in GAINS]
    hs.append(ax.plot([], [], color="#D5D5D5", lw=3.0, label="Permutation null, 95% range")[0])
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.5, frameon=False,
              handlelength=1.6, columnspacing=1.4)
    return pd.DataFrame(rows)


def panel_loo(ax):
    p = pd.read_csv(ROOT / "results/04b_robustness/paired_differences.tsv", sep="\t")
    p = p[(p.model_a == "context_model@loo_reference") & (p.model_b == "context_model@standard") & (p.level == "donor")]
    pos = endpoint_axis(ax, "Leave-one-out −\nstandard reference")
    rows = []
    for ct, ep, x in pos:
        r = p[(p.cell_type == ct) & (p.endpoint == ep)].iloc[0]
        ax.errorbar([x], [r.difference], yerr=[[r.difference - r.lo], [r.hi - r.difference]], fmt="o", ms=2.8,
                    color=TYPE_COLORS[ct], elinewidth=0.7, capsize=0)
        rows.append({"cell_type": ct, "endpoint": ep, "difference": r.difference, "lo": r.lo, "hi": r.hi,
                     "p_boot": r.p_boot})
    ax.set_ylim(-0.5, 0.5)
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    d = load_context()
    save_source(d, ROOT, "FigS4", "abc_reference_cells")
    fig = plt.figure(figsize=(170 * MM, 222 * MM))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 0.85, 0.9, 0.75], hspace=0.75, wspace=0.3)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c, e1, e2 = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])
    panel_size(a, d, "Beta"); panel_size(b, d, "Alpha"); panel_paired(c, d)
    save_source(panel_gain(e1), ROOT, "FigS4", "d_reference_gain")
    save_source(panel_loo(e2), ROOT, "FigS4", "e_leave_one_out_reference")
    for ax, l in zip((a, b, c, e1, e2), "abcde"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "FigS4")


if __name__ == "__main__":
    main()
