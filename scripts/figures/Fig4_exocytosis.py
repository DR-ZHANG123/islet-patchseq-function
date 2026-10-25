"""Fig. 4：β 细胞胞吐不由单细胞转录组编码。只读结果表。

a  全部方法在 β 与 α 细胞胞吐（总、早期、晚期）上的供体内准确率（点 + 按供体重抽样 95% 置信区间），
   方法按估计值排序；横线为该设计下 80% 功效可检测的最小 r（results/04d_reliability）。
   供体参照对同一供体的所有细胞平移相同的量，带供体参照的模型与其细胞层面模型的供体内预测相同，只计一次。
b  Ensemble 的供体内 r 随训练供体数的变化（训练折中按供体抽取 20%–100%，每个比例 10 次；按供体重抽样 95% 置信区间；
   results/04g_learning_curve）：β 大小、β Na⁺ 电流、α 总胞吐、β 早期与总胞吐。
c  α 细胞下采样到 β 细胞规模（相同供体数，每供体细胞数按秩匹配）后胞吐的供体内 r（每次下采样一点）；
   菱形 = α 全部数据，方块 = β 细胞（results/04e_training_size）。
d  β 细胞早期胞吐与总胞吐的供体层面 ρ：实测值（20 次重复均值）对置换零分布（供体平均偏移在供体间互换，
   完整交叉验证重跑 100 次；results/04c_donor_null）。
e, f  胞吐分解为 Ca²⁺ 可解释部分与偶联效率后的可预测性（fig4_coupling.py；results/04j_coupling）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coupling_panels as FC  # noqa: E402
from figstyle import (MM, TYPE_COLORS, apply_style, model_color, model_label, panel_label, save_fig,  # noqa: E402
                      save_source)

ROOT = Path(__file__).resolve().parents[2]
EXO = ["exocytosis", "early_exocytosis", "late_exocytosis"]
EXO_LAB = {"exocytosis": "Total", "early_exocytosis": "Early", "late_exocytosis": "Late"}
ENS = "cell_branch"
NULL_MODELS = ["ridge", "cell_branch"]


def distinct_methods(models):
    """供体内预测互不相同的方法：去掉带供体参照的版本（与其细胞层面模型相同）与单独的供体参照（供体内无定义）。"""
    drop = {"context_model", "context_model_full_gate", "linear_context_model", "donor_branch"}
    return [m for m in models if m not in drop and not m.endswith("_context") and not m.endswith("_donor_branch")]


def panel_all(ax):
    m = pd.read_csv(ROOT / "results/03e_sota/metrics_cv.tsv", sep="\t")
    mods = distinct_methods(sorted(m.model.unique()))
    m = m[m.model.isin(mods) & m.endpoint.isin(EXO)]
    det = pd.read_csv(ROOT / "results/04d_reliability/detectable_and_upper_bounds.tsv", sep="\t")
    rows = []
    x0, ticks, labs = 0.0, [], []
    width = 0.8
    for ct in ("Beta", "Alpha"):
        for ep in EXO:
            s = m[(m.cell_type == ct) & (m.endpoint == ep)].sort_values("within_pearson").reset_index(drop=True)
            xs = x0 - width / 2 + width * (np.arange(len(s)) + 0.5) / len(s)
            for xx, (_, r) in zip(xs, s.iterrows()):
                ours = r.model == ENS
                col = "#B2182B" if ours else "#9A9A9A"
                ax.plot([xx, xx], [r.within_lo, r.within_hi], color=col, lw=0.8 if ours else 0.45,
                        alpha=1 if ours else 0.8, zorder=3 if ours else 2)
                ax.scatter([xx], [r.within_pearson], s=11 if ours else 4, color=col, zorder=4 if ours else 2, lw=0)
                rows.append({"cell_type": ct, "endpoint": ep, "method": model_label(r.model), "model_id": r.model,
                             "within_r": r.within_pearson, "lo": r.within_lo, "hi": r.within_hi, "x": xx})
            d = det[(det.cell_type == ct) & (det.endpoint == ep)].min_detectable_r_80.iloc[0]
            ax.plot([x0 - 0.47, x0 + 0.47], [d, d], color="#2C6FB0", lw=0.9, ls="--", zorder=1)
            rows.append({"cell_type": ct, "endpoint": ep, "method": "Smallest detectable r (80% power)",
                         "model_id": "min_detectable_r_80", "within_r": d, "lo": np.nan, "hi": np.nan, "x": np.nan})
            ticks.append(x0); labs.append(EXO_LAB[ep])
            x0 += 1
        x0 += 0.4
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7)
    ax.set_xlim(-0.6, x0 - 0.8)
    ax.set_ylim(-0.22, 0.36)
    ax.set_ylabel("Within-donor r (95% CI)")
    for ct, xc in (("Beta", 1.0), ("Alpha", 4.4)):
        ax.text(xc, 0.35, "β cells, exocytosis" if ct == "Beta" else "α cells, exocytosis", ha="center", va="top",
                fontsize=7, color=TYPE_COLORS[ct])
    n = len(mods)
    hs = [ax.scatter([], [], s=11, color="#B2182B", label="Ensemble (± donor ref.)"),
          ax.scatter([], [], s=5, color="#9A9A9A", label=f"Other methods ({n - 1})"),
          ax.plot([], [], color="#2C6FB0", ls="--", lw=0.9, label="Smallest detectable r (80% power)")[0]]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.5, frameon=False)
    return pd.DataFrame(rows)


CURVE = [("Beta", "size", "β size", "#1B6F66", "s", "-"),
         ("Beta", "sodium_current", "β Na$^{+}$ current", "#2C6FB0", "^", "-"),
         ("Alpha", "exocytosis", "α total exo.", "#C8811B", "o", "--"),
         ("Beta", "early_exocytosis", "β early exo.", "#E7877B", "D", "-"),
         ("Beta", "exocytosis", "β total exo.", "#B2182B", "o", "-")]


def panel_curve(ax):
    """Ensemble 的供体内 r 随训练供体数的变化（每个训练比例 10 次抽样均值；按供体重抽样 95% 置信区间）。"""
    c = pd.read_csv(ROOT / "results/04g_learning_curve/curve.tsv", sep="\t")
    c = c[(c.model == ENS) & (c.level == "within")]
    rows = []
    for k, (ct, ep, lab, col, mk, ls) in enumerate(CURVE):
        s = c[(c.cell_type == ct) & (c.endpoint == ep)].sort_values("n_train_donors")
        x = s.n_train_donors.values + (k - 2) * 1.2
        ax.errorbar(x, s.estimate, yerr=[s.estimate - s.lo, s.hi - s.estimate], fmt=mk, ls=ls, ms=3.2, lw=0.9,
                    elinewidth=0.6, capsize=0, color=col, mfc="white" if ct == "Alpha" else col, mew=0.8,
                    label=lab, zorder=3)
        for _, r in s.iterrows():
            rows.append({"cell_type": ct, "endpoint": ep, "series": lab.replace("$^{+}$", "+"), "model_id": ENS,
                         "fraction_of_training_donors": r.fraction, "n_training_donors": r.n_train_donors,
                         "within_r": r.estimate, "lo": r.lo, "hi": r.hi, "n_draws": r.n_draws})
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xlim(5, 66)
    ax.set_xticks([12, 24, 36, 48, 60])
    ax.set_ylim(-0.1, 0.72)
    ax.set_xlabel("Training donors (n)")
    ax.set_ylabel("Within-donor r (95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, frameon=False,
              handletextpad=0.3, columnspacing=0.8, handlelength=1.8)
    return pd.DataFrame(rows)


def panel_size(ax):
    d = pd.read_csv(ROOT / "results/04e_training_size/metrics_by_draw.tsv", sep="\t")
    d = d[(d.model == "context_model") & d.endpoint.isin(EXO)]
    s = pd.read_csv(ROOT / "results/04e_training_size/summary.tsv", sep="\t")
    s = s[(s.model == "context_model") & s.endpoint.isin(EXO)].set_index("endpoint")
    rng = np.random.default_rng(3)
    rows = []
    for i, ep in enumerate(EXO):
        v = d[d.endpoint == ep].within_pearson.values
        xs = i + rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(xs, v, s=9, color=TYPE_COLORS["Alpha"], alpha=0.75, lw=0, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=TYPE_COLORS["Alpha"], lw=1.2, zorder=2)
        ax.scatter([i + 0.32], [s.loc[ep, "within_full_Alpha"]], marker="D", s=14, color="white",
                   edgecolor=TYPE_COLORS["Alpha"], lw=0.9, zorder=3)
        ax.scatter([i + 0.32], [s.loc[ep, "within_full_Beta"]], marker="s", s=14, color=TYPE_COLORS["Beta"], zorder=3)
        for dr, val in zip(d[d.endpoint == ep].draw, v):
            rows.append({"endpoint": ep, "series": "Alpha subsampled to beta size", "draw": dr, "within_r": val})
        rows += [{"endpoint": ep, "series": "Alpha, all cells", "draw": np.nan, "within_r": s.loc[ep, "within_full_Alpha"]},
                 {"endpoint": ep, "series": "Beta, all cells", "draw": np.nan, "within_r": s.loc[ep, "within_full_Beta"]}]
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels([EXO_LAB[e] for e in EXO], fontsize=7)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-0.08, 0.32)
    ax.set_xlabel("Exocytosis")
    ax.set_ylabel("Within-donor r")
    hs = [ax.scatter([], [], s=9, color=TYPE_COLORS["Alpha"], label="α, β-sized data"),
          ax.scatter([], [], marker="D", s=14, color="white", edgecolor=TYPE_COLORS["Alpha"], lw=0.9,
                     label="α, all data"),
          ax.scatter([], [], marker="s", s=14, color=TYPE_COLORS["Beta"], label="β cells")]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5, frameon=False,
              handletextpad=0.2, borderaxespad=0.2)
    return pd.DataFrame(rows)


def panel_null(ax):
    n = pd.read_csv(ROOT / "results/04c_donor_null/null_metrics.tsv", sep="\t")
    c = pd.read_csv(ROOT / "results/04c_donor_null/calibration.tsv", sep="\t")
    sel = lambda t: t[(t.cell_type == "Beta") & (t["null"] == "donor") & (t.grid == "default")]  # noqa: E731
    n, c = sel(n), sel(c)
    c = c[c.level == "donor_spearman"]
    rows = []
    pos, ticks, labs = 0, [], []
    for ep in ("early_exocytosis", "exocytosis"):
        for k, mod in enumerate(NULL_MODELS):
            v = n[(n.endpoint == ep) & (n.model == mod)].donor_spearman.values
            obs = c[(c.endpoint == ep) & (c.model == mod)].iloc[0]
            vp = ax.violinplot([v], positions=[pos], widths=0.75, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor("#D5D5D5"); b.set_edgecolor("none"); b.set_alpha(1)
            ax.plot([pos - 0.2, pos + 0.2], [v.mean()] * 2, color="#555555", lw=0.9)
            ax.scatter([pos], [obs.observed], s=22, color=model_color(mod), zorder=3, edgecolor="white", lw=0.4)
            rows += [{"endpoint": ep, "method": model_label(mod), "series": "permutation", "value": x} for x in v]
            rows.append({"endpoint": ep, "method": model_label(mod), "series": "observed", "value": obs.observed,
                         "p_perm": obs.p_perm})
            ticks.append(pos); labs.append(f"P =\n{obs.p_perm:.2f}")
            pos += 1
        pos += 0.4
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=6)
    ax.set_xlim(-0.6, pos - 0.9)
    ax.set_ylim(-0.9, 0.8)
    ax.set_ylabel("Donor-level ρ, β cells")
    for xc, ep in ((0.5, "Early"), (2.9, "Total")):
        ax.text(xc, 0.78, ep, ha="center", va="top", fontsize=6.5)
    ax.set_xlabel("β-cell exocytosis")
    hs = [ax.scatter([], [], s=22, color=model_color(m_), label=f"{model_label(m_, short=True)}")
          for m_ in NULL_MODELS]
    hs.append(ax.fill_between([], [], color="#D5D5D5", label="Permutation null"))
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5, frameon=False,
              handletextpad=0.3)
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(170 * MM, 225 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.05, 1.0], hspace=0.62)
    a = fig.add_subplot(gs[0])
    g2 = gs[1].subgridspec(1, 3, width_ratios=[1.0, 0.85, 0.9], wspace=0.55)
    b, c, d = (fig.add_subplot(g2[0, k]) for k in range(3))
    save_source(panel_all(a), ROOT, "Fig4", "a_exocytosis_all_methods")
    save_source(panel_curve(b), ROOT, "Fig4", "b_learning_curve")
    save_source(panel_size(c), ROOT, "Fig4", "c_alpha_subsampled")
    save_source(panel_null(d), ROOT, "Fig4", "d_donor_null_beta_exocytosis")
    g3 = gs[2].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.3)
    e, f = fig.add_subplot(g3[0, 0]), fig.add_subplot(g3[0, 1])
    save_source(FC.panel_within(e, ROOT), ROOT, "Fig4", "e_coupling_within_donor")
    save_source(FC.panel_donor(f, ROOT), ROOT, "Fig4", "f_coupling_donor_level")
    fig._col_tol = 0.15
    for ax, l in zip((a, b, c, d, e, f), "abcdef"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "Fig4")


if __name__ == "__main__":
    main()
