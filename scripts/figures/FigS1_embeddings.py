"""补充图 S1：单细胞基础模型冻结嵌入的性质（results/03d_embed/embedding_summary.tsv）。只读结果表。

a  α/β 细胞类型可分性（嵌入上的平衡准确率）。
b  嵌入空间最近邻中来自同一供体的比例（虚线 = 随机期望），标注相对随机的富集倍数。
c  嵌入空间最近邻中来自同一采集中心的比例。
以对数归一化表达的主成分（PCA）作参照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, REPRESENTATIONS, apply_style, model_color, panel_label, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = dict(REPRESENTATIONS, reference_pca_lognorm="PCA of expression\n(reference)")
COLORS = {"reference_pca_lognorm": "#222222"}


def dots(ax, e, col, xlabel, xlim):
    y = np.arange(len(e))
    for i, (_, r) in enumerate(e.iterrows()):
        c = COLORS.get(r.model, model_color(r.model + "_ridge"))
        ax.plot([xlim[0], r[col]], [i, i], color="#DDDDDD", lw=1.0, zorder=1)
        ax.scatter([r[col]], [i], s=22, color=c, zorder=3, marker="D" if r.model in COLORS else "o")
    ax.set_yticks(y)
    ax.set_ylim(len(e) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


def main() -> None:
    apply_style()
    e = pd.read_csv(ROOT / "results/03d_embed/embedding_summary.tsv", sep="\t")
    keep = ["model", "setting", "dim", "celltype_balanced_acc", "knn_same_donor_frac", "knn_same_donor_expected",
            "knn_donor_enrichment", "knn_same_site_frac"]
    e = e[keep].reset_index(drop=True)
    save_source(e.assign(label=e.model.map(lambda m: NAMES[m].replace("\n", " "))), ROOT, "FigS1",
                "abc_embedding_summary")
    fig = plt.figure(figsize=(170 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.12)
    a, b, c = (fig.add_subplot(gs[0, k]) for k in range(3))
    dots(a, e, "celltype_balanced_acc", "α/β balanced accuracy", (0.8, 1.0))
    a.set_yticklabels([NAMES[m] for m in e.model], fontsize=6.5)
    dots(b, e, "knn_same_donor_frac", "Nearest neighbours\nfrom the same donor", (0, 0.4))
    b.axvline(e.knn_same_donor_expected.iloc[0], color="#2C6FB0", lw=0.8, ls="--")
    for i, (_, r) in enumerate(e.iterrows()):
        b.text(r.knn_same_donor_frac + 0.018, i, f"{r.knn_donor_enrichment:.1f}×", va="center", ha="left",
               fontsize=6)
    b.text(e.knn_same_donor_expected.iloc[0] + 0.01, len(e) - 0.55, "random", fontsize=6, color="#2C6FB0",
           va="bottom", ha="left")
    dots(c, e, "knn_same_site_frac", "Nearest neighbours\nfrom the same centre", (0.4, 0.85))
    for ax in (b, c):
        ax.set_yticklabels([])
    for ax, l in zip((a, b, c), "abc"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "FigS1")


if __name__ == "__main__":
    main()
