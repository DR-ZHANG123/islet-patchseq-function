"""Fig. 6e,f：胞吐分解为 Ca²⁺ 可解释部分与偶联效率（Ca²⁺ 内流之外的残差）后，转录组对各部分的可预测性。
只读 results/04j_coupling 的结果表。

e  供体内 r（岭回归–梯度提升集成，留出供体；按供体重抽样 95% 置信区间）：Ca²⁺ 内流、实测胞吐、
   Ca²⁺ 可解释部分与偶联效率，β 与 α 细胞的总胞吐和早期胞吐。
f  β 细胞供体层面 ρ：实测值对置换零分布（供体平均偏移在全部供体间互换；灰条 95% 范围，短横为均值），
   标注全部供体间与同一来源中心内置换的 P 值。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from figstyle import TYPE_COLORS

ENS = "cell_branch"
COMP = {"calcium_entry": ("Ca$^{2+}$ entry", "#2C6FB0", "s"),
        "measured": ("Measured exocytosis", "#555555", "o"),
        "ca_explained": ("Ca$^{2+}$-explained part", "#1B9E77", "D"),
        "coupling": ("Coupling efficiency", "#B2182B", "^")}
TARGET = {"total": "exocytosis", "early": "early_exocytosis"}


def _endpoint(comp: str, tgt: str) -> str:
    if comp == "calcium_entry":
        return "calcium_entry"
    if comp == "measured":
        return TARGET[tgt]
    return f"{comp}_{tgt}"


def panel_within(ax, root: Path) -> pd.DataFrame:
    m = pd.read_csv(root / "results/04j_coupling/metrics.tsv", sep="\t")
    m = m[(m.predictor == ENS) & (m.level == "within")]
    rows, ticks, labs = [], [], []
    x = 0.0
    for ct in ("Beta", "Alpha"):
        start = x
        r = m[(m.cell_type == ct) & (m.endpoint == "calcium_entry")].iloc[0]
        groups = [("entry", ["calcium_entry"])] + [(t, ["measured", "ca_explained", "coupling"]) for t in TARGET]
        for tgt, comps in groups:
            xs = [x + 0.22 * (k - (len(comps) - 1) / 2) for k in range(len(comps))]
            for xx, comp in zip(xs, comps):
                ep = _endpoint(comp, "total" if tgt == "entry" else tgt)
                r = m[(m.cell_type == ct) & (m.endpoint == ep)].iloc[0]
                lab, col, mk = COMP[comp]
                ax.plot([xx, xx], [r.lo, r.hi], color=col, lw=0.8)
                ax.plot(xx, r.estimate, mk, ms=3.6, color=col, mec="white", mew=0.3)
                rows.append({"cell_type": ct, "exocytosis": "" if tgt == "entry" else tgt, "component": comp,
                             "endpoint": ep, "within_r": r.estimate, "lo": r.lo, "hi": r.hi, "n_donors": r.n_donors})
            ticks.append(x); labs.append({"entry": "Ca$^{2+}$\nentry", "total": "Total\nexo.", "early": "Early\nexo."}[tgt])
            x += 0.9
        ax.text((start + x - 0.9) / 2, 0.47, "β cells" if ct == "Beta" else "α cells", ha="center", va="top",
                fontsize=7, color=TYPE_COLORS[ct])
        x += 0.35
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_xlim(-0.45, x - 0.8)
    ax.set_ylim(-0.1, 0.48)
    ax.set_ylabel("Within-donor r (95% CI)")
    hs = [ax.plot([], [], COMP[c][2], ms=3.6, color=COMP[c][1], label=COMP[c][0])[0] for c in COMP]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, frameon=False,
              handletextpad=0.2, columnspacing=0.9)
    return pd.DataFrame(rows)


DONOR = [("calcium_entry", "Ca$^{2+}$\nentry"), ("early_exocytosis", "Early\nexo."),
         ("ca_explained_early", "Early,\nCa$^{2+}$-\nexpl."), ("coupling_early", "Early,\ncoupling"),
         ("coupling_total", "Total,\ncoupling")]


def panel_donor(ax, root: Path) -> pd.DataFrame:
    c = pd.read_csv(root / "results/04j_coupling/donor_null_calibration.tsv", sep="\t")
    c = c[(c.cell_type == "Beta") & (c.model == ENS)]
    rows = []
    for x, (ep, _) in enumerate(DONOR):
        d = c[(c.endpoint == ep) & (c["null"] == "donor")].iloc[0]
        s = c[(c.endpoint == ep) & (c["null"] == "site")].iloc[0]
        ax.plot([x, x], [d.null_q025, d.null_q975], color="#D0D0D0", lw=5, solid_capstyle="butt", zorder=1)
        ax.plot([x - 0.18, x + 0.18], [d.null_mean] * 2, color="#555555", lw=0.9, zorder=2)
        col = COMP["coupling"][1] if ep.startswith("coupling") else "#555555"
        ax.plot(x, d.observed, "o", ms=4.2, color=col, mec="white", mew=0.4, zorder=3)
        ax.text(x, max(d.observed, d.null_q975) + 0.05, f"P = {d.p_perm:.2f}\n({s.p_perm:.2f})", ha="center",
                va="bottom", fontsize=6, color="#333333")
        rows.append({"endpoint": ep, "observed": d.observed, "null_mean": d.null_mean, "null_q025": d.null_q025,
                     "null_q975": d.null_q975, "p_all_donors": d.p_perm, "p_within_centre": s.p_perm,
                     "n_perm": d.n_perm})
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(range(len(DONOR))); ax.set_xticklabels([l for _, l in DONOR], fontsize=6)
    ax.set_xlim(-0.6, len(DONOR) - 0.4)
    ax.set_ylim(-0.5, 0.75)
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Donor-level ρ, β cells")
    hs = [ax.plot([], [], "o", ms=4.2, color="#555555", label="Observed")[0],
          ax.plot([], [], color="#D0D0D0", lw=5, label="Permutation null (95%)")[0]]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, frameon=False,
              handletextpad=0.5, columnspacing=1.6, handlelength=1.4)
    return pd.DataFrame(rows)
