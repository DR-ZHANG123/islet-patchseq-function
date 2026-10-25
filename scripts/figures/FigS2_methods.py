"""Supplementary Fig. S2：各方法族代表与岭–提升集成的供体内准确率配对差（同一批留出细胞，按供体重抽样 95% 置信区间）。

代表的固定选取规则见 figures/source_data/Fig3_NOTE_selection_rules.txt：经典方法 = 预测基因集（原始流程重建，
results/03g_ps_faithful）、岭回归、调参梯度提升；神经网络 = 调参单细胞 MLP；学习表示 = scVI 与参数量最大的基础模型（UCE 33-layer）。
只读结果表。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, model_label, save_fig, save_source  # noqa: E402
from Fig2_variance import ORDER  # noqa: E402
from Fig3_prediction import REPS, ylabels  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def panel_diff(axs, d):
    rows = []
    for ax, mod in zip(axs, REPS):
        s = d[(d.method == mod) & (d.level == "within")].set_index(["cell_type", "endpoint"])
        for k, ct in enumerate(("Beta", "Alpha")):
            col = TYPE_COLORS[ct]
            for i, ep in enumerate(ORDER):
                r = s.loc[(ct, ep)]
                sig = r.hi < 0 or r.lo > 0
                ax.errorbar([r.difference], [i + (k - 0.5) * 0.4], xerr=[[r.difference - r.lo], [r.hi - r.difference]],
                            fmt="o", ms=2.6, color=col, elinewidth=0.6, capsize=0, mew=0.7,
                            mfc=col if sig else "white")
                rows.append({"method": model_label(mod), "cell_type": ct, "endpoint": ep,
                             "difference_vs_ensemble": r.difference, "lo": r.lo, "hi": r.hi, "p_boot": r.p_boot,
                             "ci_excludes_0": sig})
        ax.axvline(0, color="#777777", lw=0.6)
        ax.set_xlim(-0.42, 0.2)
        ax.set_xticks([-0.4, -0.2, 0]); ax.set_xticklabels(["−0.4", "−0.2", "0"])
        ax.set_title(model_label(mod, short=True).replace(" (", "\n("), fontsize=6.5)
        ylabels(ax, ax is axs[0])
    hs = [axs[0].errorbar([], [], fmt="o", ms=2.8, color=TYPE_COLORS[ct], label="β cells" if ct == "Beta" else "α cells")
          for ct in ("Beta", "Alpha")]
    hs.append(axs[0].errorbar([], [], fmt="o", ms=2.8, color="#777777", mfc="white", mew=0.7,
                              label="95% CI includes 0"))
    return pd.DataFrame(rows), hs


def main() -> None:
    apply_style()
    d = pd.read_csv(ROOT / "results/03e_sota/paired_differences.tsv", sep="\t")
    d = d[d.reference == "context_model"]
    dp = pd.read_csv(ROOT / "results/03g_ps_faithful/paired_differences.tsv", sep="\t")
    dp = dp[(dp.split == "donors") & (dp.test_cells == "all_test_cells") & (dp.reference == "cell_branch")
            & (dp.method == "ps_faithful_nd")]
    d = pd.concat([d, dp.drop(columns=["split", "test_cells"])], ignore_index=True)
    fig = plt.figure(figsize=(170 * MM, 80 * MM))
    gc = fig.add_gridspec(1, 6, wspace=0.14)
    cx = [fig.add_subplot(gc[0, k]) for k in range(6)]
    hc, hs = panel_diff(cx, d)
    for ax in cx:
        ax.set_xlabel(" ")
    cx[2].set_xlabel("Within-donor r, method − Ensemble (95% CI)", fontsize=7)
    cx[2].xaxis.set_label_coords(1.07, -0.1)
    cx[5].legend(handles=hs, loc="upper left", bbox_to_anchor=(-5.4, -0.16), ncol=3, fontsize=6, frameon=False)
    save_source(hc, ROOT, "FigS2", "paired_within_differences")
    save_fig(fig, ROOT, "FigS2")


if __name__ == "__main__":
    main()
