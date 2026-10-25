"""Fig. 5：T2D 中 β 细胞正值胞吐反应的几何均值降低（非零反应比例未见改变），晚期胞吐受损多于早期，且不由 Ca²⁺ 进入解释。
只读 results/09_t2d_function、results/09g_t2d_mechanism 与 results/09h_t2d_glucose 的结果表，并把每个面板的输入写到 figures/source_data/。

a  两部分模型的正值部分：T2D/ND 正值反应几何均值之比（供体随机截距；95% 置信区间；* BH 校正 q < 0.05）。
b  两部分模型的零值部分：非零反应的比值比。
c  β 细胞正值反应：同一组细胞中 Ca²⁺ 进入的 T2D 比；胞吐的 T2D 比，及按细胞 Ca²⁺ 进入调整（未校正、按供体内信度 0.5 与 0.2
   回归校准、供体均值）后的比；晚期与早期 T2D 比之比（供体等权线性模型，按状态 × 中心分层重抽样供体）。
d  β 细胞非零值各百分位的 T2D/ND 比（每位供体等权）。
e  β 细胞正值部分比值在各中心、各葡萄糖浓度、协变量调整与分离/培养调整下的估计。
f  高于 ND 上分位数的 β 细胞比例：实测，与“每个正值反应都按估计比值降低”时的预期。
（分布形状——P90/P50 效应比与供体内离散度之比——见 Additional file 4。）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, apply_style, panel_label, save_fig, save_source  # noqa: E402
import fig5_panels as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/09_t2d_function"


def main() -> None:
    apply_style()
    tc = yaml.safe_load(open(ROOT / "config/run.yaml"))["t2d"]
    t = pd.read_csv(RES / "t2d_effects.tsv", sep="\t")
    rob = pd.read_csv(RES / "t2d_robustness.tsv", sep="\t")
    save_source(t[t.effect.isin(["positive_ratio", "nonzero_odds_ratio"])], ROOT, "Fig5", "ab_two_part")
    save_source(t[(t.cell_type == "Beta") & t.effect.str.match(r"q\d+_pos_ratio")], ROOT, "Fig5", "d_profile")
    save_source(t[(t.cell_type == "Beta") & (t.endpoint == "exocytosis") & t.effect.str.startswith("share")],
                ROOT, "Fig5", "f_shares")

    fig = plt.figure(figsize=(170 * MM, 140 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.62, height_ratios=[1, 1.05])
    r1 = gs[0].subgridspec(1, 3, wspace=0.9, width_ratios=[1, 0.8, 0.95])
    r2 = gs[1].subgridspec(1, 3, wspace=0.95, width_ratios=[1, 0.85, 1])
    a, b, c = (fig.add_subplot(r1[0, i]) for i in range(3))
    P.forest(a, t, "positive_ratio", "Ratio of positive-response\ngeometric means, T2D / ND", (0.3, 2.0), [0.5, 1, 2], True)
    a.legend(loc="lower left", fontsize=6, handletextpad=0.2, borderpad=0.3)
    P.forest(b, t, "nonzero_odds_ratio", "Odds ratio, non-zero response", (0.15, 8.0), [0.25, 1, 4], False)
    ca = pd.read_csv(ROOT / "results/09g_t2d_mechanism/calcium_log_adjustment.tsv", sep="\t")
    el = pd.read_csv(ROOT / "results/09g_t2d_mechanism/early_late.tsv", sep="\t")
    save_source(P.calcium(c, ca, el), ROOT, "Fig5", "c_calcium_early_late")
    d, e, f = (fig.add_subplot(r2[0, i]) for i in range(3))
    P.profile(d, t, tc["quantiles"])
    glu = pd.read_csv(ROOT / "results/09h_t2d_glucose/glucose_two_part.tsv", sep="\t")
    save_source(P.robustness(e, t, rob, glu), ROOT, "Fig5", "e_robustness")
    P.shares(f, t)
    for ax, l in zip((a, b, c, d, e, f), "abcdef"):
        panel_label(ax, l)
    save_fig(fig, ROOT, "Fig5")


if __name__ == "__main__":
    main()
