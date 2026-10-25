"""补充图 S6：β 与 α 细胞总胞吐的可靠性参考值与可检测效应（原 Fig. 4b）。只读 results/04d_reliability。

β 细胞（实心）与 α 细胞（空心）总胞吐：Ensemble + donor ref. 的观测供体内 r、单侧 95% 上界、全部方法 97.5% 上界的
最大值、按测量可靠性校正后的上界、该设计下 80% 功效可检测的最小 r，以及测量可靠性上限
（早期 + 晚期胞吐内部一致性给出的可达上限）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, save_fig, save_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def panel_bounds(ax):
    det = pd.read_csv(ROOT / "results/04d_reliability/detectable_and_upper_bounds.tsv", sep="\t")
    con = pd.read_csv(ROOT / "results/04d_reliability/within_consistency.tsv", sep="\t")
    con = con[con.subset == "all_cells"].set_index("cell_type")
    items = [("context_model_within", "Observed r"),
             ("context_model_upper95_one_sided", "Upper 95% bound"),
             ("max_hi975_all_methods_03c", "Highest upper bound,\nany method"),
             ("disattenuated_upper95_exocytosis", "Upper bound corrected\nfor reliability"),
             ("min_detectable_r_80", "Smallest detectable r\n(80% power)"),
             ("ceiling", "Reliability ceiling")]
    rows = []
    for k, ct in enumerate(("Beta", "Alpha")):
        r = det[(det.cell_type == ct) & (det.endpoint == "exocytosis")].iloc[0]
        col = TYPE_COLORS[ct]
        for i, (key, lab) in enumerate(items):
            v = con.loc[ct, "ceiling_within_total"] if key == "ceiling" else r[key]
            ax.scatter([v], [i + (k - 0.5) * 0.32], s=18 if k == 0 else 14, color=col if k == 0 else "white",
                       edgecolor=col, lw=0.9, zorder=3)
            rows.append({"cell_type": ct, "endpoint": "exocytosis", "quantity": lab.replace("\n", " "), "value": v})
    ax.set_yticks(range(len(items))); ax.set_yticklabels([lab for _, lab in items], fontsize=6.5)
    ax.set_ylim(len(items) - 0.5, -0.5)
    for i in range(1, len(items)):
        ax.axhline(i - 0.5, color="#EEEEEE", lw=0.5, zorder=0)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Within-donor r, total exocytosis")
    hs = [ax.scatter([], [], s=18, color=TYPE_COLORS["Beta"], label="β cells"),
          ax.scatter([], [], s=14, color="white", edgecolor=TYPE_COLORS["Alpha"], lw=0.9, label="α cells")]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(110 * MM, 78 * MM))
    save_source(panel_bounds(ax), ROOT, "FigS9", "reliability_bounds")
    save_fig(fig, ROOT, "FigS9")


if __name__ == "__main__":
    main()
