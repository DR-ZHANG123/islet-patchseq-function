"""Fig. 1：研究设计（矢量图）。

b 栏：细胞自身的表达谱 → 35 种方法（经典模型、神经网络、学习表示含基础模型嵌入）→ 主模型岭回归–梯度提升集成；
供体参照（同一供体其他同类型细胞的平均表达）作为检验过的附加项，以小的虚线分支画出。
文字、方框、箭头与示意曲线由代码生成（文字为可编辑文本）；图标为代码绘制的矢量图形（icons_vector.py），输出与自查见 schematic.py。
子图字母大小写由 FIG_PANEL_CASE 决定，输出目录由 FIG_OUT_DIR 决定（见 figstyle）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import TYPE_COLORS, apply_style, fig_out_dir, panel_case  # noqa: E402
from icons_vector import ephys_traces, expression_bars  # noqa: E402
from schematic import Canvas  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H = 170.0, 112.0                     # mm
TEAL, AMBER, RED, BLUE, GREY = TYPE_COLORS["Beta"], TYPE_COLORS["Alpha"], "#B2182B", "#2C6FB0", "#555555"
# 方法族：名称、示例方法、填充色、边线色
FAMILIES = [("Classical models", "gene set, ridge, PLS, sRRR, boosting", "#F2F2F2", "#8C8C8C"),
            ("Neural networks", "multilayer perceptrons", "#FBF0E1", "#C8811B"),
            ("Learned representations", "scVI, scGPT, Geneformer, UCE", "#EFEAF5", "#7B5EA7")]


def panel_a(c: Canvas):
    c.text(2, 4.5, panel_case("a"), 11, weight="bold")
    c.text(8, 4.8, "Paired single-cell measurements", 7.5, weight="bold")
    c.icon("donor", 3.0, 12.0, 21.0); c.icon("islet", 24.0, 13.0, 19.0); c.icon("patch", 6.0, 37.0, 30.0)
    c.arrow(22.5, 22.5, 24.9, 22.5)
    c.arrow(33.5, 32.5, 27.0, 38.5)
    c.text(3.5, 34.5, "96 donors", 6.2, color=GREY)
    c.text(43.5, 23.0, "Dispersed\nislet cells", 6.0, color=GREY)
    expression_bars(c.ax, H, 6, 78, 18, 6, 1.35, 1.55, TEAL, seed=4, lo=0.15)
    c.text(6, 81.5, "Transcriptome\n(Smart-seq2)", 6.2, va="top")
    ephys_traces(c.ax, H, 36, 16, 69, 78, RED, BLUE)
    c.text(35.0, 81.5, "Currents and\nexocytosis", 6.2, va="top")
    c.text(3.0, 92.5, "4,297 α and β cells\n9 electrophysiological measures", 6.5, va="top")


def panel_b(c: Canvas):
    cx = 84.5                                           # b 栏中线
    c.text(56, 4.5, panel_case("b"), 11, weight="bold")
    c.text(62, 4.8, "Prediction from expression", 7.5, weight="bold")
    # 输入：单个细胞的表达谱
    c.icon("cell", 61.0, 10.0, 16.0)
    c.text(79.5, 15.0, "The cell's own expression", 6.5, weight="bold")
    c.text(79.5, 19.6, "Smart-seq2 profile of the\nrecorded cell", 6.0, color=GREY, va="center")
    c.arrow(cx, 26.8, cx, 31.6)
    # 35 种方法，按方法族分组
    c.box(58.0, 32.0, 53.0, 44.0, "#FAFAFA", "#B0B0B0", lw=0.6)
    c.text(cx, 35.6, "35 approaches compared", 6.8, ha="center", weight="bold")
    for k, (name, examples, fc, ec) in enumerate(FAMILIES):
        y = 39.2 + k * 12.0
        c.box(60.5, y, 48.0, 10.6, fc, ec, lw=0.7, r=1.2)
        c.text(cx, y + 3.3, name, 6.5, ha="center", weight="bold", color="#333333")
        c.text(cx, y + 7.4, examples, 6.0, ha="center", color="#333333")
    c.arrow(cx, 76.3, cx, 80.8, color=RED, lw=1.0)
    # 主模型
    c.box(66.0, 81.2, 37.0, 11.4, "#FBEAEA", RED, lw=1.1)
    c.text(cx, 84.8, "Ridge–boosting ensemble", 6.8, ha="center", weight="bold", color=RED)
    c.text(cx, 89.2, "main model", 6.0, ha="center", color="#333333")
    # 检验过的附加项：供体参照（小分支、虚线）
    c.icon("reference", 58.5, 95.2, 13.0)
    c.box(73.0, 97.0, 38.0, 11.2, "white", "#7A9CC6", lw=0.6, r=1.0, ls=(0, (2.2, 1.4)))
    c.text(92.0, 100.0, "Tested addition: donor reference", 6.0, ha="center", color=BLUE)
    c.text(92.0, 104.6, "mean of the donor's other cells", 6.0, ha="center", color=GREY)
    c.arrow(92.0, 96.8, 92.0, 93.0, color="#7A9CC6", lw=0.7, ls=(0, (2.0, 1.2)), scale=6)


def mini_scatter(c: Canvas, x, y, kind):
    ax = c.add_axes(x, y, 14, 14)
    rng = np.random.default_rng(7)
    cols = ["#2A9D8F", "#C8811B", "#8E6BB0"]
    if kind == "cells":
        v = rng.normal(size=60); ax.scatter(v, 0.5 * v + rng.normal(size=60) * 0.8, s=2, color="#777777", lw=0)
    elif kind == "donors":
        v = rng.normal(size=12); ax.scatter(v, 0.7 * v + rng.normal(size=12) * 0.5, s=6, color="#555555", lw=0)
    else:
        for k, col in enumerate(cols):
            v = rng.normal(size=10)
            ax.scatter(v + k * 3.5, 0.6 * v + rng.normal(size=10) * 0.5 + k * 0.3, s=3, color=col, lw=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def panel_c(c: Canvas):
    c.text(116, 4.5, panel_case("c"), 11, weight="bold")
    c.text(122, 4.8, "Evaluation in unseen donors", 7.5, weight="bold")
    for x, lab, kind in zip([118.5, 135.5, 152.5], ["Across cells", "Donor means", "Within donors"],
                            ["cells", "donors", "within"]):
        mini_scatter(c, x, 19.0, kind)
        c.text(x + 7, 36.5, lab, 6.0, ha="center")
    c.box(118, 40.5, 49, 11.5, "#F4F4F4", GREY, lw=0.6)
    c.text(142.5, 43.8, "69 development donors: donors held out", 6.0, ha="center")
    c.text(142.5, 48.6, "27 external donors: tested once", 6.0, ha="center")


def panel_d(c: Canvas):
    """以转录组为条件的检验：T2D 差异若经转录状态而来，以细胞自身转录组为条件后应缩小。"""
    c.text(116, 58.5, panel_case("d"), 11, weight="bold")
    c.text(122, 58.8, "Is the T2D defect in the transcriptome?", 7.0, weight="bold")
    c.box(117.5, 72.5, 14.0, 8.0, "#FBEAEA", "#C0392B", lw=0.8, r=1.0)
    c.text(124.5, 76.5, "T2D", 6.5, ha="center", weight="bold", color="#C0392B")
    c.box(134.5, 63.5, 20.0, 9.5, "#EAF4F2", TEAL, lw=0.8, r=1.0)
    c.text(144.5, 68.25, "Transcriptional\nstate", 6.0, ha="center", color="#1F6F66")
    c.box(154.0, 72.5, 14.0, 8.0, "#F4F4F4", GREY, lw=0.8, r=1.0)
    c.text(161.0, 76.5, "Exocytosis", 6.0, ha="center")
    c.arrow(131.6, 73.6, 134.4, 71.4, color=TEAL, lw=0.8, scale=6)
    c.arrow(154.6, 68.5, 160.5, 72.3, color=TEAL, lw=0.8, scale=6)
    c.arrow(131.7, 78.6, 153.9, 78.6, color=GREY, lw=0.8, scale=6)
    c.text(142.8, 81.0, "not through transcripts", 6.0, ha="center", color=GREY)
    # 实测差异与以转录组为条件后的差异
    c.text(117.5, 87.0, "T2D − ND, measured", 6.0)
    c.box(146.0, 85.6, 20.0, 2.8, "#C0392B", "#C0392B", lw=0.4, r=0.3)
    c.text(117.5, 92.0, "conditioned on transcriptome", 6.0)
    c.box(146.0, 90.6, 12.0, 2.8, BLUE, BLUE, lw=0.4, r=0.3)
    c.box(158.0, 90.6, 8.0, 2.8, "white", TEAL, lw=0.6, r=0.3, ls=(0, (1.6, 1.0)))
    c.text(162.0, 96.3, "carried", 6.0, ha="center", color="#1F6F66")
    c.text(117.5, 100.6, "Calibrated with alpha cells, shuffled\ncontrols and simulations", 6.0, va="top",
           color=GREY)


def main() -> None:
    apply_style()
    c = Canvas(W, H)
    panel_a(c); panel_b(c); panel_c(c); panel_d(c)
    c.line([55.0, 55.0], [3, H - 3]); c.line([114.5, 114.5], [3, H - 3]); c.line([116, W - 2], [55.0, 55.0])
    issues = c.check(min_pt=6.0)
    if issues:
        raise SystemExit("Fig1 layout problems:\n" + "\n".join(issues))
    c.compose(fig_out_dir(ROOT) / "Fig1", png_px=int(W / 25.4 * 300))
    print("Fig1 written; layout checks passed")


if __name__ == "__main__":
    main()
