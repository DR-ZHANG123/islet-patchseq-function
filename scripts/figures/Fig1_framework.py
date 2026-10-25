"""Fig. 1：研究设计（矢量图），以 T2D 胞吐缺陷为主线的四个面板。

a 同一细胞的两种读数（供体 → 胰岛 → 膜片钳细胞；转录组与电流/胞吐）；b 转录组编码什么（35 种模型、三个层面、留出供体）；
c T2D 缺陷的问题（早期已引发颗粒 vs 晚期持续释放，Ca2+ 进入）；d 缺陷是否由转录组携带（独立图谱的 T2D 特征在同一批
Patch-seq β 细胞中打分；实测差异与以转录组/胞吐基因为条件后的差异，只画问题、不画结果；校准）。
文字、方框、箭头与示意曲线由代码生成（文字为可编辑文本）；图标为代码绘制的矢量图形（icons_vector.py），输出与自查见 schematic.py。
子图字母大小写由 FIG_PANEL_CASE 决定，输出目录由 FIG_OUT_DIR 决定（见 figstyle）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import TYPE_COLORS, apply_style, fig_out_dir, panel_case  # noqa: E402
from icons_vector import ephys_traces  # noqa: E402
from schematic import Canvas  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H = 170.0, 118.0                     # mm
TEAL, AMBER, RED, BLUE, GREY = TYPE_COLORS["Beta"], TYPE_COLORS["Alpha"], "#B2182B", "#2C6FB0", "#555555"
T2D, ND = "#C0392B", "#3A6EA5"


def mini_scatter(c: Canvas, x, y, kind):
    ax = c.add_axes(x, y, 15, 13)
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
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def panel_a(c: Canvas):
    c.text(2, 4.5, panel_case("a"), 11, weight="bold")
    c.text(8, 4.8, "Same cells, two readouts", 7.5, weight="bold")
    c.icon("donor", 3.0, 9.5, 19.0); c.icon("islet", 26.0, 10.5, 17.0); c.icon("patch", 50.0, 7.5, 24.0)
    c.arrow(22.0, 19.0, 25.6, 19.0); c.arrow(43.4, 19.0, 49.6, 19.0)
    c.text(12.5, 31.0, "Donor", 6.0, ha="center", color=GREY)
    c.text(34.5, 31.0, "Islet", 6.0, ha="center", color=GREY)
    c.text(62.0, 33.0, "Patch-clamped cell", 6.0, ha="center", color=GREY)
    c.arrow(56.0, 36.0, 26.0, 40.5, color=TEAL, lw=0.8, scale=6)
    c.arrow(64.0, 36.0, 64.0, 39.6, color=BLUE, lw=0.8, scale=6)
    c.icon("mrna", 8.0, 39.0, 13.0)
    c.text(23.0, 43.5, "Transcriptome\n(Smart-seq2)", 6.0, va="top")
    ephys_traces(c.ax, H, 46.0, 30.0, 45.0, 52.5, RED, BLUE)
    c.text(46.0, 41.6, "Currents, Ca$^{2+}$ entry, exocytosis", 6.0)


def panel_b(c: Canvas):
    c.text(2, 62.5, panel_case("b"), 11, weight="bold")
    c.text(8, 62.8, "What the transcriptome encodes", 7.5, weight="bold")
    c.text(5.0, 70.0, "35 models, including single-cell foundation models", 6.2)
    for x, lab, kind in zip([6.0, 31.0, 56.0], ["Across cells", "Donor means", "Within donors"],
                            ["cells", "donors", "within"]):
        mini_scatter(c, x, 75.0, kind)
        c.text(x + 7.5, 91.8, lab, 6.0, ha="center")
    c.box(4.5, 97.0, 72.0, 13.0, "#F4F4F4", GREY, lw=0.6)
    c.text(40.5, 100.6, "Donors held out from training", 6.0, ha="center")
    c.text(40.5, 105.6, "27 external donors tested once", 6.0, ha="center")


def panel_c(c: Canvas):
    c.text(86, 4.5, panel_case("c"), 11, weight="bold")
    c.text(92, 4.8, "The exocytotic defect of T2D", 7.5, weight="bold")
    c.icon("granules", 88.0, 9.0, 30.0)
    c.text(121.0, 12.6, "Ca$^{2+}$ entry through channels", 6.0, color=BLUE)
    c.text(121.0, 18.6, "Early release: docked, primed granules", 6.0)
    c.text(121.0, 24.6, "Late release: granule supply during", 6.0)
    c.text(121.0, 28.4, "sustained stimulation", 6.0)
    c.text(121.0, 35.0, "Which step is reduced in T2D?", 6.2, weight="bold", color=T2D)


def panel_d(c: Canvas):
    c.text(86, 46.5, panel_case("d"), 11, weight="bold")
    c.text(92, 46.8, "Is the defect carried by the transcriptome?", 7.5, weight="bold")
    c.icon("droplet_atlas", 88.0, 51.5, 17.0)
    c.text(107.0, 56.6, "T2D signature from an", 6.0)
    c.text(107.0, 60.4, "independent droplet atlas", 6.0)
    c.arrow(140.0, 58.6, 145.6, 58.6, color=GREY)
    c.icon("cell", 146.5, 51.0, 15.0)
    c.text(154.0, 68.8, "Scored in the same", 6.0, ha="center")
    c.text(154.0, 72.6, "Patch-seq β cells", 6.0, ha="center")
    c.line([88, 168], [77.0, 77.0])
    c.text(88.5, 81.6, "T2D − ND, measured", 6.0)
    c.box(138.0, 80.2, 26.0, 2.8, T2D, T2D, lw=0.4, r=0.3)
    c.text(88.5, 87.6, "Conditioned on the transcriptome", 6.0)
    c.box(138.0, 86.2, 26.0, 2.8, "white", BLUE, lw=0.6, r=0.3, ls=(0, (1.6, 1.0)))
    c.text(166.0, 87.6, "?", 7.0, ha="center", weight="bold", color=BLUE)
    c.text(88.5, 93.6, "Conditioned on exocytosis genes", 6.0)
    c.box(138.0, 92.2, 26.0, 2.8, "white", BLUE, lw=0.6, r=0.3, ls=(0, (1.6, 1.0)))
    c.text(166.0, 93.6, "?", 7.0, ha="center", weight="bold", color=BLUE)
    c.text(88.5, 100.6, "Share carried = shrinkage of the T2D difference", 6.0, color=GREY)
    c.text(88.5, 107.6, "Calibration: alpha cells, shuffled controls,", 6.0, color=GREY)
    c.text(88.5, 111.4, "simulated carriers", 6.0, color=GREY)


def main() -> None:
    apply_style()
    c = Canvas(W, H)
    panel_a(c); panel_b(c); panel_c(c); panel_d(c)
    c.line([83.5, 83.5], [3, H - 3]); c.line([3, 80], [58.5, 58.5]); c.line([86, W - 2], [41.5, 41.5])
    issues = c.check(min_pt=6.0)
    if issues:
        raise SystemExit("Fig1 layout problems:\n" + "\n".join(issues))
    c.compose(fig_out_dir(ROOT) / "Fig1", png_px=int(W / 25.4 * 300))
    print("Fig1 written; layout checks passed")


if __name__ == "__main__":
    main()
