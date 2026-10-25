"""Fig. 3：新供体与外部队列中的预测。只读结果表（results/03e_sota 包含 Stage 03/03b 的全部方法与新增对照，折划分相同）。

a  供体内准确率（Pearson r，留出供体）热图（以 0 为中心的发散色标）：行 = 方法（经典、神经网络、学习表示、Ensemble），列 = 9 终点 × 2 细胞类型。
   Ensemble + donor ref. 的供体内预测与 Ensemble 相同（供体参照对同一供体的细胞平移相同），不单列一行。
b  Ensemble 在细胞、供体均值、供体内三个层面的准确率及按供体重抽样 95% 置信区间。
c、d  外部供体（HPAP）的细胞层面 ρ 与供体内 r（开发供体训练；横线 = 外部队列内部交叉验证）。
各方法族代表 − Ensemble 的供体内配对差见 Supplementary Fig. S2（FigS2_methods.py）。

固定的选取规则（写入 source_data 说明文件，不按结果挑选）：
- 学习表示（scVI 与各基础模型嵌入）一律取岭回归头；
- 神经网络取 8 个 MLP 变体（4 种结构 × 固定/调参）中 18 列平均供体内 r 最高的 2 个（规则见说明文件）；
- c 中方法族代表：经典方法 = 预测基因集（原始流程重建，results/03g_ps_faithful）、岭回归、调参梯度提升；神经网络 = 调参单细胞 MLP；
  学习表示 = scVI 与参数量最大的基础模型（UCE 33-layer）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, TYPE_COLORS, apply_style, model_color, model_label, panel_label, save_fig, save_source  # noqa: E402
from Fig2_variance import ORDER, SHORT  # noqa: E402
import FigS5_external as EXT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL = ["ps_faithful_nd", "knn_predictive_set", "elastic_net", "pls", "ridge", "srrr", "gradient_boosting",
             "gradient_boosting_tuned"]
MLPS = ["mlp_cell", "mlp_concat", "mlp_two", "mlp_two_within", "mlp_cell_tuned", "mlp_concat_tuned",
        "mlp_two_tuned", "mlp_two_within_tuned"]
REPR = ["scvi_ridge", "scgpt_human_ridge", "geneformer_v2_104m_ridge", "geneformer_v2_316m_ridge",
        "uce_4layer_ridge", "uce_33layer_ridge"]
OURS = ["cell_branch"]
REPS = ["ps_faithful_nd", "ridge", "gradient_boosting_tuned", "mlp_cell_tuned", "scvi_ridge", "uce_33layer_ridge"]
COLS = [(ct, ep) for ct in ("Beta", "Alpha") for ep in ORDER]
LEVELS = [("cell_spearman", "cell", "Across cells (ρ)"), ("donor_spearman", "donor", "Donor means (ρ)"),
          ("within_pearson", "within", "Within donors (r)")]
NOTE = """Fig. 3 source data: fixed rules for choosing methods and heads
- Every learned representation (scVI latent space; scGPT, Geneformer and UCE embeddings) is shown with the ridge-regression head.
- Neural networks: the two of the eight MLP variants (four architectures, fixed or tuned settings) with the highest within-donor r averaged over all 18 cell type x measure columns: {mlps}.
- The predictive set (original procedure) row reconstructs the published predictive-gene-set kNN procedure inside each training fold (gene selection, one gene set for all nine measures, kNN with k = 5), trained on cells from donors without diabetes as in the original description and evaluated on the same held-out-donor folds; the outcome-specific kNN row selects genes separately for each measure.
- Ensemble + donor ref. shares its within-donor predictions with the Ensemble (the donor reference shifts all cells of a donor equally), so it is not shown as a separate row in panel a.
- Supplementary Fig. S2 family representatives: predictive set (original procedure), ridge and tuned gradient boosting (classical); tuned single-cell MLP (neural network); scVI and UCE 33-layer, the largest foundation model (learned representations). Differences are method minus Ensemble + donor ref., identical within donors to method minus Ensemble.
"""


def col_label(ct, ep):
    return SHORT[ep]


def load():
    m = pd.read_csv(ROOT / "results/03e_sota/metrics_cv.tsv", sep="\t")
    ps = pd.read_csv(ROOT / "results/03g_ps_faithful/metrics_cv.tsv", sep="\t")
    ps = ps[(ps.model == "ps_faithful_nd") & (ps.split == "donors") & (ps.test_cells == "all_test_cells")]
    m = pd.concat([m, ps.drop(columns=["split", "test_cells"])], ignore_index=True)
    mlp_rank = m[m.model.isin(MLPS)].groupby("model").within_pearson.mean().sort_values(ascending=False)
    mlps = list(mlp_rank.index[:2])
    rows = CLASSICAL + mlps + REPR + OURS
    groups = [("Classical", len(CLASSICAL)), ("Neural\nnetworks", 2), ("Learned\nrepresentations", len(REPR)),
              ("This study", 1)]
    return m, rows, groups, mlps, mlp_rank


def panel_heat(ax, m, rows, groups):
    t = np.array([[m[(m.cell_type == ct) & (m.endpoint == ep) & (m.model == r)].within_pearson.iloc[0]
                   for ct, ep in COLS] for r in rows])
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.6)
    im = ax.imshow(t, cmap="RdBu_r", norm=norm, aspect="auto")
    for j in range(t.shape[1]):
        best = np.nanmax(t[:, j])
        for i in range(t.shape[0]):
            v = t[i, j]
            txt = ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                          fontsize=6, color="white" if (v > 0.4 or v < -0.2) else "#222222",
                          fontweight="bold" if np.isclose(v, best) else "normal")
            txt._overlap_exempt = True
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([col_label(*c) for c in COLS], rotation=60, ha="right",
                                                        fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([model_label(r, short=True) for r in rows], fontsize=6)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.axvline(8.5, color="white", lw=2.0)
    y = -0.5
    for name, n in groups[:-1]:
        y += n
        ax.axhline(y, color="white", lw=1.6)
    for k, ct in enumerate(("Beta", "Alpha")):
        ax.text(4 + 9 * k, -0.75, "β cells" if ct == "Beta" else "α cells", ha="center", va="bottom", fontsize=7,
                color=TYPE_COLORS[ct])
    # 方法组：热图右侧的括号与组名
    y0 = -0.5
    for name, n in groups:
        ax.plot([18.0, 18.0], [y0 + 0.12, y0 + n - 0.12], color="#555555", lw=0.7, clip_on=False)
        ax.text(18.3, y0 + n / 2, name, ha="left", va="center", fontsize=6, color="#333333")
        y0 += n
    ax.set_xlim(-0.5, 17.5)
    rows_out = [{"method": model_label(r), "model_id": r, "cell_type": ct, "endpoint": ep, "within_r": t[i, j]}
                for i, r in enumerate(rows) for j, (ct, ep) in enumerate(COLS)]
    return im, pd.DataFrame(rows_out)


def ylabels(ax, show):
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[e] for e in ORDER] if show else [], fontsize=6)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    for k in range(1, len(ORDER)):
        ax.axhline(k - 0.5, color="#EEEEEE", lw=0.5, zorder=0)


OFF_B = {("Beta", "cell_branch"): -0.15, ("Alpha", "cell_branch"): 0.15}


def panel_levels(axs, m):
    rows = []
    for ax, (met, lev, title) in zip(axs, LEVELS):
        for mod in OURS:
            s = m[m.model == mod].set_index(["cell_type", "endpoint"])
            for ct in ("Beta", "Alpha"):
                for i, ep in enumerate(ORDER):
                    r = s.loc[(ct, ep)]
                    col = TYPE_COLORS[ct]
                    ax.errorbar([r[met]], [i + OFF_B[(ct, mod)]],
                                xerr=[[r[met] - r[f"{lev}_lo"]], [r[f"{lev}_hi"] - r[met]]], fmt="o", ms=2.6,
                                color=col, elinewidth=0.6, capsize=0, mew=0.7)
                    rows.append({"method": model_label(mod), "cell_type": ct, "endpoint": ep, "level": lev,
                                 "estimate": r[met], "lo": r[f"{lev}_lo"], "hi": r[f"{lev}_hi"]})
        ax.axvline(0, color="#999999", lw=0.5, ls=":")
        ax.set_title(title, fontsize=7)
        if lev == "donor":
            ax.set_xlim(-0.8, 1.0); ax.set_xticks([-0.5, 0, 0.5])
        else:
            ax.set_xlim(-0.25, 0.85); ax.set_xticks([0, 0.25, 0.5, 0.75])
        ylabels(ax, ax is axs[0])
    hs = []
    for ct in ("Beta", "Alpha"):
        for mod in OURS:
            col = TYPE_COLORS[ct]
            hs.append(axs[0].errorbar([], [], fmt="o", ms=2.8, color=col, mew=0.7,
                                      label=f"{'β' if ct == 'Beta' else 'α'} cells, {model_label(mod, short=True)}"))
    axs[1].legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False,
                  columnspacing=1.2, handletextpad=0.3)
    return pd.DataFrame(rows)


def main() -> None:
    apply_style()
    m, rows, groups, mlps, mlp_rank = load()
    fig = plt.figure(figsize=(170 * MM, 225 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.12, 0.8, 1.0], hspace=0.5)
    a = fig.add_subplot(gs[0])
    im, ha = panel_heat(a, m, rows, groups)
    cb = fig.colorbar(im, ax=a, fraction=0.018, pad=0.155)
    cb.set_ticks([-0.3, -0.2, -0.1, 0, 0.2, 0.4, 0.6])
    cb.set_ticklabels(["−0.3", "−0.2", "−0.1", "0", "0.2", "0.4", "0.6"])
    cb.set_label("Within-donor r", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    gb = gs[1].subgridspec(1, 3, wspace=0.2)
    bx = [fig.add_subplot(gb[0, k]) for k in range(3)]
    hb = panel_levels(bx, m)
    ext, ceil, *_ = EXT.load()
    gcd = gs[2].subgridspec(2, 1, hspace=0.3)
    c1 = fig.add_subplot(gcd[0]); c2 = fig.add_subplot(gcd[1])
    EXT.panel_ext(c1, ext, ceil, "cell_spearman", "cell", "External donors,\ncell-level ρ")
    EXT.panel_ext(c2, ext, ceil, "within_pearson", "within", "External donors,\nwithin-donor r")
    c1.tick_params(labelbottom=False)
    hs = [c1.errorbar([], [], fmt="o", ms=3, color=EXT.MODEL_COLORS[x], label=model_label(x, short=True))
          for x in EXT.MODS]
    hs.append(c1.plot([], [], color="#222222", ls="--", lw=1.0,
                      label="Within-cohort training (Ensemble + donor ref.)")[0])
    c1.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6, ncol=3, columnspacing=1.2,
              handletextpad=0.3, frameon=False)
    save_source(ha, ROOT, "Fig3", "a_within_heatmap")
    save_source(hb, ROOT, "Fig3", "b_ensemble_levels")
    save_source(ext, ROOT, "Fig3", "cd_external_cells")
    save_source(ceil, ROOT, "Fig3", "cd_internal_ceiling")
    save_source(mlp_rank.rename("mean_within_r_18_columns").reset_index().assign(
        label=lambda x: x.model.map(model_label)), ROOT, "Fig3", "a_mlp_rule")
    (ROOT / "figures/source_data/Fig3_NOTE_selection_rules.txt").write_text(
        NOTE.format(mlps=", ".join(model_label(x) for x in mlps)))
    fig._col_tol = 0.2
    panel_label(a, "a"); panel_label(bx[0], "b"); panel_label(c1, "c"); panel_label(c2, "d")
    save_fig(fig, ROOT, "Fig3")


if __name__ == "__main__":
    main()
