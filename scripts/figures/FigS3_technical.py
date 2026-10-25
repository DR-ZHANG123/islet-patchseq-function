"""补充图 S2：技术与批次特征作基线（results/04f_technical/metrics_boot.tsv）。只读结果表。

行 = 细胞类型（β 上、α 下），列 = 评价层面（跨细胞、供体均值、供体内），每行 9 个终点。
方法：只用质控特征、只用批次特征（GEO 系列、来源中心）、质控 + 批次的梯度提升树（固定规则：技术基线一律取
梯度提升头，岭回归头的结果在作图输入表中一并给出），Ensemble + donor ref.，以及表达特征先对质控特征回归取残差后的
Ensemble + donor ref.。批次特征在供体内恒定，没有供体内预测。点 = 估计值，线 = 按供体重抽样 95% 置信区间。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, panel_label, save_fig, save_source  # noqa: E402
from Fig2_variance import ORDER, SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODS = [("qc_only_boost", "QC features only", "#A6A6A6"),
        ("batch_only_boost", "Batch only", "#6B6B6B"),
        ("qc_batch_boost", "QC + batch", "#2E2E2E"),
        ("context_model_qc_partialled", "Ensemble + donor ref. (QC-adjusted)", "#F4A582"),
        ("context_model", "Ensemble + donor ref.", "#B2182B")]
LEVELS = [("cell", "Across cells (ρ)"), ("donor", "Donor means (ρ)"), ("within", "Within donors (r)")]
LABELS = {"qc_only_ridge": "QC features only (ridge)", "batch_only_ridge": "Batch only (ridge)",
          "qc_batch_ridge": "QC + batch (ridge)", "ridge": "Ridge", "gradient_boosting": "Gradient boosting (fixed)",
          "cell_branch": "Ensemble", "ridge_qc_partialled": "Ridge, QC-adjusted expression",
          "cell_branch_qc_partialled": "Ensemble, QC-adjusted expression"}


def main() -> None:
    apply_style()
    m = pd.read_csv(ROOT / "results/04f_technical/metrics_boot.tsv", sep="\t")
    lab = dict(LABELS, **{k: v for k, v, _ in MODS})
    save_source(m.assign(method=m.model.map(lab)), ROOT, "FigS3", "technical_baselines")
    fig = plt.figure(figsize=(170 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.12, hspace=0.32)
    axes = {}
    for i, ct in enumerate(("Beta", "Alpha")):
        for j, (lev, title) in enumerate(LEVELS):
            ax = fig.add_subplot(gs[i, j]); axes[(i, j)] = ax
            s = m[(m.cell_type == ct) & (m.level == lev)].set_index(["endpoint", "model"])
            for k, (mod, _, col) in enumerate(MODS):
                for y, ep in enumerate(ORDER):
                    if (ep, mod) not in s.index or pd.isna(s.loc[(ep, mod), "estimate"]):
                        continue
                    r = s.loc[(ep, mod)]
                    yy = y + (k - 2) * 0.15
                    ax.errorbar([r.estimate], [yy], xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o",
                                ms=2.4, color=col, elinewidth=0.6, capsize=0)
            ax.axvline(0, color="#999999", lw=0.5, ls=":")
            ax.set_yticks(range(len(ORDER)))
            ax.set_yticklabels([SHORT[e] for e in ORDER] if j == 0 else [], fontsize=6.5)
            ax.set_ylim(len(ORDER) - 0.5, -0.5)
            for y in range(1, len(ORDER)):
                ax.axhline(y - 0.5, color="#EEEEEE", lw=0.5, zorder=0)
            ax.set_xlim(-1.0, 1.0) if lev == "donor" else ax.set_xlim(-0.45, 0.85)
            ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel("β cells" if ct == "Beta" else "α cells", color=TYPE_COLORS[ct], fontsize=8)
    hs = [axes[(0, 0)].errorbar([], [], fmt="o", ms=3, color=col, label=name) for _, name, col in MODS]
    axes[(1, 1)].legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.5,
                        frameon=False, columnspacing=1.2, handletextpad=0.3)
    panel_label(axes[(0, 0)], "a"); panel_label(axes[(1, 0)], "b")
    save_fig(fig, ROOT, "FigS3")


if __name__ == "__main__":
    main()
