"""Stage 04d：测量可靠性上限、供体内检测功效与 β 细胞胞吐供体内准确度的置信上界（开发供体）。

- 供体均值可靠性（ICC 与 split-half 两种估计）及其给出的供体层面相关上限，与 Stage 04b 观测值比较；
- 细胞内一致性：总胞吐 = 早期 + 晚期胞吐，两部分 Cronbach α；各终点之间的供体内相关；
- 供体内功效：在实际细胞/供体结构下可检测（80% 功效、按供体重抽样 95% 置信区间下界 > 0）的最小 r；
- 置信上界：Stage 04b 20 次重复平均预测的单侧 95% 上界，以及 Stage 03c 全部方法 97.5% 上界的最大值。
输出 results/04d_reliability/*.tsv。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import robust_reliability as rr  # noqa: E402

STAGE = "04d_reliability"
RAW = {"exocytosis": "exocytosis", "early_exocytosis": "early_exocytosis", "late_exocytosis": "late_exocytosis"}


def donor_reliability(td, cfg, k, obs, rng):
    rc = cfg["robustness"]["reliability"]
    mg = cfg["cv"]["min_group"]
    z = rr.glucose_std(td.y[:, k], td.glucose)
    ok = ~np.isnan(z)
    icc = rr.icc1(z[ok], td.donor[ok])
    n = pd.Series(td.donor[ok]).value_counts()
    n = n[n >= mg].values.astype(float)
    rel = lambda i: float(np.mean(n * i / (1 + (n - 1) * i)))  # noqa: E731
    donors = np.unique(td.donor[ok])
    bi = []
    for _ in range(rc["n_boot"]):
        pick = rng.choice(donors, len(donors), replace=True)
        m = np.concatenate([np.where(td.donor[ok] == d)[0] for d in pick])
        lab = np.concatenate([[f"{d}#{j}"] * int((td.donor[ok] == d).sum()) for j, d in enumerate(pick)])
        bi.append(rr.icc1(z[ok][m], lab))
    bi = np.array(bi)
    ds, h1, h2, ns = rr.split_halves(z, td.donor, rc["min_cells_split"], rc["split_draws"], rng)
    shp, shlo, shhi = rr.split_half_reliability(h1, h2, rc["n_boot"], rng, rank=False)
    shr, shrlo, shrhi = rr.split_half_reliability(h1, h2, rc["n_boot"], rng, rank=True)
    row = {"icc": icc, "icc_lo": np.percentile(bi, 2.5), "icc_hi": np.percentile(bi, 97.5),
           "n_donors_eval": len(n), "median_cells": float(np.median(n)),
           "reliability_icc": rel(icc), "reliability_icc_lo": rel(np.percentile(bi, 2.5)),
           "reliability_icc_hi": rel(np.percentile(bi, 97.5)),
           "split_half_donors": len(ds), "split_half_median_cells": float(np.median(ns)),
           "reliability_split": shp, "reliability_split_lo": shlo, "reliability_split_hi": shhi,
           "reliability_split_rank": shr, "reliability_split_rank_lo": shrlo, "reliability_split_rank_hi": shrhi}
    for key in ("icc", "split", "split_rank"):
        for suf in ("", "_lo", "_hi"):
            v = row[f"reliability_{key}{suf}"]
            row[f"ceiling_{key}{suf}"] = float(np.sqrt(max(v, 0)))
    for m in ("context_model", "cell_branch", "ridge", "linear_cell_branch"):
        if m in obs.index:
            row[f"observed_{m}"] = float(obs[m])
            row[f"fraction_of_ceiling_{m}"] = float(obs[m] / row["ceiling_icc"]) if row["ceiling_icc"] > 0 else np.nan
    return row


def within_consistency(td, cfg):
    """细胞内：早期与晚期胞吐（原始尺度）的供体内一致性。负电容值在数据中截为 0，截断后总量不再严格等于两部分之和，
    因此分别报告全部细胞与“总量 = 早期 + 晚期”（偏差 < 0.05 fF/pF）成立的细胞。"""
    c = td.cells
    e, l, t = (c[RAW[x]].values for x in ("early_exocytosis", "late_exocytosis", "exocytosis"))
    base = ~(np.isnan(e) | np.isnan(l) | np.isnan(t))
    consistent = base & (np.abs(t - e - l) < 0.05)
    rows = []
    for name, ok in (("all_cells", base), ("sum_consistent_cells", consistent)):
        g = td.group[ok]
        sizes = pd.Series(g).map(pd.Series(g).value_counts()).values >= 2
        ee, ll, gg = e[ok][sizes], l[ok][sizes], g[sizes]
        ec, lc = rr.centred(ee, gg), rr.centred(ll, gg)
        a = rr.alpha_two(ee, ll, gg)
        rows.append({"measure": "exocytosis_early_plus_late", "subset": name, "n_cells": int(len(ee)),
                     "fraction_sum_consistent": float(consistent.sum() / base.sum()),
                     "within_pearson_early_late_raw": float(pearsonr(ec, lc)[0]),
                     "within_spearman_early_late_raw": float(spearmanr(ec, lc).statistic),
                     "within_pearson_early_late_log1p": float(pearsonr(rr.centred(np.log1p(ee), gg),
                                                                        rr.centred(np.log1p(ll), gg))[0]),
                     "cronbach_alpha_total_raw": a, "ceiling_within_total": float(np.sqrt(max(a, 0)))})
    return pd.DataFrame(rows)


def measure_correlations(td):
    """各终点之间的合并供体内 Pearson（变换后尺度，比较组内去均值）。"""
    rows = []
    K = len(td.endpoints)
    for i in range(K):
        for j in range(i + 1, K):
            a, b = td.y[:, i], td.y[:, j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 20:
                continue
            ac, bc = rr.centred(a[ok], td.group[ok]), rr.centred(b[ok], td.group[ok])
            rows.append({"endpoint_a": td.endpoints[i], "endpoint_b": td.endpoints[j], "n_cells": int(ok.sum()),
                         "within_pearson": float(pearsonr(ac, bc)[0])})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rc = cfg["robustness"]["reliability"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(ROOT / "results/04b_robustness/metrics_by_repeat.tsv", sep="\t")
    boot = pd.read_csv(ROOT / "results/04b_robustness/metrics_boot.tsv", sep="\t")
    s03 = pd.read_csv(ROOT / "results/03c_summary/metrics_cv.tsv", sep="\t")
    dons, wins, cors, pows, mdes = [], [], [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        rng = np.random.default_rng([cfg["seed"], len(ct)])
        r = rep[(rep.cell_type == ct) & (rep.variant == "standard")]
        for k, ep in enumerate(td.endpoints):
            obs = r[r.endpoint == ep].groupby("model").donor_spearman.mean()
            dons.append({"cell_type": ct, "endpoint": ep} | donor_reliability(td, cfg, k, obs, rng))
            y = td.y[:, k]
            ok = ~np.isnan(y)
            curve = rr.power_curve(y[ok], td.group[ok], td.donor[ok], rc["power_r"], rc["power_sims"],
                                   rc["power_boot"], rng)
            pows.append(curve.assign(cell_type=ct, endpoint=ep))
            b = boot[(boot.cell_type == ct) & (boot.endpoint == ep) & (boot.comparison == "standard_all_repeats")
                     & (boot.level == "within")].set_index("model")
            o3 = s03[(s03.cell_type == ct) & (s03.split == "donors") & (s03.endpoint == ep)]
            best = o3.loc[o3.within_hi.idxmax()] if len(o3) else None
            mdes.append({"cell_type": ct, "endpoint": ep,
                         "min_detectable_r_80": rr.min_detectable(curve, rc["power_target"]),
                         "context_model_within": b.at["context_model", "estimate"],
                         "context_model_upper95_one_sided": b.at["context_model", "upper95_one_sided"],
                         "context_model_hi975": b.at["context_model", "hi"],
                         "cell_branch_upper95_one_sided": b.at["cell_branch", "upper95_one_sided"],
                         "max_hi975_all_methods_03c": float(best.within_hi) if best is not None else np.nan,
                         "method_max_hi975_03c": best.model if best is not None else "",
                         "max_estimate_all_methods_03c": float(o3.within_pearson.max()) if len(o3) else np.nan})
        wins.append(within_consistency(td, cfg).assign(cell_type=ct))
        cors.append(measure_correlations(td).assign(cell_type=ct))
    win = pd.concat(wins)
    mde = pd.DataFrame(mdes)
    a = win[win.subset == "all_cells"].set_index("cell_type").cronbach_alpha_total_raw
    mde["disattenuated_upper95_exocytosis"] = [
        r.context_model_upper95_one_sided / np.sqrt(a[r.cell_type]) if r.endpoint == "exocytosis" and a[r.cell_type] > 0
        else np.nan for r in mde.itertuples()]
    tabs = {"reliability_donor.tsv": pd.DataFrame(dons), "within_consistency.tsv": win,
            "measure_correlations_within.tsv": pd.concat(cors), "power_within.tsv": pd.concat(pows),
            "detectable_and_upper_bounds.tsv": mde}
    outs = []
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/04b_robustness/metrics_boot.tsv",
                             ROOT / "results/03c_summary/metrics_cv.tsv"], outs, seed=cfg["seed"])
    print(tabs["reliability_donor.tsv"].round(3).to_string())
    print(win.round(3).to_string())
    print(mde.round(3).to_string())


if __name__ == "__main__":
    main()
