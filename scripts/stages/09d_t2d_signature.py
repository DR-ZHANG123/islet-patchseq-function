"""Stage 09d：β 细胞 T2D 转录特征是否携带胞吐缺陷。

1. 留出供体推导 T2D 转录特征（只用折外开发供体，HPAP 不参与推导）并给每个 β 细胞打分。
2. (i) 得分的 T2D − ND 差；(ii) 得分与各终点的供体内与供体层面关联；
   (iii) 中介：在 Stage 09c 的供体等权模型中加入特征得分、ND 训练的预测值（Stage 09c）或两者，
   T2D 系数下降的比例。胞吐类终点另做仅正值反应部分（并用随机截距模型复核）。
推断：状态 × 中心分层的供体重抽样（区间，得分固定）；中心内 ND/T2D 标签置换（P 值，每次置换重新推导特征）。
公开文献的 T2D 基因列表未纳入：无法在本流程内可核查地下载其补充表。
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from t2d_hidden import _boot, parts  # noqa: E402
from t2d_signature import association, derive_signature, fold_scores, mediation, safe  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09d_t2d_signature"
MAIN_SITES = ["ADI IsletCore", "HPAP"]
LMM_PARTS = {("exocytosis", "positive responses"), ("late_exocytosis", "positive responses")}
TESTED = ["total", "score_t2d", "indirect_signature", "indirect_predicted", "indirect_both"]


def frames(cfg, td, status, site, s) -> dict:
    """按 Stage 09c 的行顺序重建每个终点的细胞表，并接上 09c 预测值与特征得分。"""
    pred = pd.read_csv(ROOT / "results/09c_t2d_hidden/predictions.tsv.gz", sep="\t")
    pred = pred[pred.cell_type == td.cells.cell_type.iloc[0]]
    out = {}
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        ok = np.isin(status, ["ND", "preT2D", "T2D"]) & ~np.isnan(y)
        d = pd.DataFrame({"y": y, "s": s, "g": td.group, "status": status, "donor": td.donor, "site": site,
                          "glucose": td.glucose, "cell_row": np.arange(len(y))})[ok]
        d = d[d.groupby("g").y.transform("size") >= 2].reset_index(drop=True)
        pe = pred[pred.endpoint == ep].reset_index(drop=True)
        if len(pe) != len(d) or not np.allclose(pe.y.values, d.y.values) or (pe.donor.values != d.donor.values).any():
            raise RuntimeError(f"row alignment with Stage 09c failed for {ep}")
        d["p"] = pe.p.values
        d = d.dropna(subset=["p", "s"])
        d["t2d"] = (d.status == "T2D").astype(float)
        d["pre"] = (d.status == "preT2D").astype(float)
        for part, sub in parts(d, ep).items():
            out[(ep, part)] = sub.reset_index(drop=True)
    return out


def point_stats(df, key) -> dict:
    return safe(mediation, df, lmm=key in LMM_PARTS) | safe(association, df)


def permuted_status(status, site, donor, rng) -> np.ndarray:
    dtab = pd.DataFrame({"status": status, "site": site}, index=donor).groupby(level=0).first()
    lab = dtab.status.copy()
    for _, sub in dtab[dtab.status.isin(["ND", "T2D"])].groupby("site"):
        lab.loc[sub.index] = rng.permutation(sub.status.values)
    return pd.Series(donor).map(lab).values


def one_perm(td, status, site, site_main, dev, sc, nf, seed, fr) -> dict:
    rng = np.random.default_rng(seed)
    st = permuted_status(status, site_main, td.donor, rng)
    s, _ = fold_scores(td, st, site, dev, sc["n_genes"], sc["min_cpm"], nf, seed=sc["fold_seed"])
    s_of = pd.Series(s, index=np.arange(len(s)))
    lab = pd.Series(st, index=td.donor).groupby(level=0).first()
    res = {}
    for key, df in fr.items():
        d = df.copy()
        d["status"] = d.donor.map(lab).values
        d["t2d"] = (d.status == "T2D").astype(float)
        d["s"] = s_of.loc[d.cell_row].values
        m = safe(mediation, d)
        res[key] = {k: m.get(k, np.nan) for k in TESTED}
    return res


def boot_stats(fr, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for key, df in fr.items():
        rows = [point_stats(_boot(df, rng), key) for _ in range(n_boot)]
        out[key] = pd.DataFrame(rows)
    return out


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    sc = cfg["t2d_signature"] | {"fold_seed": cfg["seed"]}
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    td = load_type(cfg, sc["cell_type"], "all")
    status = td.cells.status.values
    site = td.cells.site.values
    site_main = np.where(np.isin(site, MAIN_SITES), site, "Other")
    dev = site != cfg["cv"]["external_site"]
    nf = cfg["cv"]["outer_folds"]

    s, sets = fold_scores(td, status, site, dev, sc["n_genes"], sc["min_cpm"], nf, cfg["seed"], keep_sets=True)
    td.cells.assign(signature_score=s)[["cell_id", "donor", "status", "site", "glucose", "signature_score"]].to_csv(
        out / "cell_scores.tsv.gz", sep="\t", index=False)
    gs = [pd.DataFrame({"fold": f, "direction": dr, "gene": td.symbols[ix]})
          for f, up, down in sets for dr, ix in (("up", up), ("down", down))]
    gs = pd.concat(gs, ignore_index=True)
    gs.to_csv(out / "signature_genes_by_fold.tsv", sep="\t", index=False)
    # 描述用：全部开发供体上的 T2D 差异表达（不用于打分）
    up, down, t = derive_signature(td.counts, td.donor, status, site, np.unique(td.donor[dev]), sc["n_genes"],
                                   sc["min_cpm"])
    de = pd.DataFrame({"gene": td.symbols[t.index.values], "t": t.values}).sort_values("t")
    de["in_signature_folds"] = de.gene.map(gs.groupby("gene").fold.nunique()).fillna(0).astype(int)
    de.to_csv(out / "de_all_dev_donors.tsv", sep="\t", index=False)
    print(f"signature + scores: {time.time() - t0:.0f}s", flush=True)

    fr = frames(cfg, td, status, site_main, s)
    # 全部打分细胞：得分本身的 T2D − ND
    ok = np.isin(status, ["ND", "preT2D", "T2D"]) & ~np.isnan(s)
    sc_df = pd.DataFrame({"y": s[ok], "s": s[ok], "p": s[ok], "g": td.group[ok], "status": status[ok],
                          "donor": td.donor[ok], "site": site_main[ok], "glucose": td.glucose[ok],
                          "cell_row": np.where(ok)[0]})
    sc_df["t2d"] = (sc_df.status == "T2D").astype(float)
    sc_df["pre"] = (sc_df.status == "preT2D").astype(float)
    fr[("signature_score", "all cells")] = sc_df

    point = {k: point_stats(df, k) for k, df in fr.items()}
    print(f"point: {time.time() - t0:.0f}s", flush=True)
    keys = list(fr)
    boots = Parallel(n_jobs=min(len(keys), sc["n_jobs"]))(
        delayed(boot_stats)({k: fr[k]}, sc["n_boot"], cfg["seed"] + i) for i, k in enumerate(keys))
    boots = {k: b[k] for b, k in zip(boots, keys)}
    print(f"bootstrap: {time.time() - t0:.0f}s", flush=True)
    perms = Parallel(n_jobs=sc["n_jobs"])(
        delayed(one_perm)(td, status, site, site_main, dev, sc, nf, cfg["seed"] + 10_000 + i, fr)
        for i in range(sc["n_perm"]))
    print(f"permutations: {time.time() - t0:.0f}s", flush=True)

    rows = []
    for key in keys:
        ep, part = key
        b = boots[key]
        pm = pd.DataFrame([p[key] for p in perms])
        n_donors = fr[key].groupby("status").donor.nunique()
        for q, v in point[key].items():
            bb = b[q].replace([np.inf, -np.inf], np.nan).dropna().values if q in b else np.array([])
            row = {"endpoint": ep, "part": part, "quantity": q, "estimate": v,
                   "lo": np.percentile(bb, 2.5) if len(bb) else np.nan,
                   "hi": np.percentile(bb, 97.5) if len(bb) else np.nan, "n_boot": len(bb), "p_perm": np.nan,
                   "n_perm": 0, "n_cells": len(fr[key]), "n_nd_donors": int(n_donors.get("ND", 0)),
                   "n_t2d_donors": int(n_donors.get("T2D", 0))}
            if q in TESTED and np.isfinite(v):
                pv = pm[q].dropna().values
                row["p_perm"] = (1 + np.sum(np.abs(pv) >= abs(v) - 1e-12)) / (1 + len(pv))
                row["n_perm"] = len(pv)
            rows.append(row)
    tab = pd.DataFrame(rows)
    # 得分本身只报告 T2D − ND 差，并换算为 ND 细胞（葡萄糖内去均值）得分 SD 单位
    is_s = tab.endpoint == "signature_score"
    tab = tab[~is_s | (tab.quantity == "score_t2d")].copy()
    nd = sc_df[sc_df.status == "ND"]
    sd = float((nd.s - nd.groupby("glucose").s.transform("mean")).std())
    srow = tab[tab.endpoint == "signature_score"].iloc[0].copy()
    for c in ("estimate", "lo", "hi"):
        srow[c] = srow[c] / sd
    srow["quantity"] = "score_t2d_nd_sd_units"
    tab = pd.concat([tab, srow.to_frame().T], ignore_index=True)
    tab.to_csv(out / "mediation.tsv", sep="\t", index=False)

    # 敏感性：特征基因数（点估计与重抽样区间，不做置换）
    sens = []
    for n in sc["sensitivity_n_genes"]:
        s2, _ = fold_scores(td, status, site, dev, n, sc["min_cpm"], nf, cfg["seed"])
        f2 = frames(cfg, td, status, site_main, s2)
        for i, (key, df) in enumerate(f2.items()):
            pt = point_stats(df, key)
            bt = boot_stats({key: df}, sc["n_boot"] // 4, cfg["seed"] + 500 + i)[key]
            for q in ("total", "score_t2d", "prop_signature", "indirect_signature", "within_r_all"):
                bb = bt[q].replace([np.inf, -np.inf], np.nan).dropna().values
                sens.append({"n_genes": n, "endpoint": key[0], "part": key[1], "quantity": q,
                             "estimate": pt.get(q, np.nan), "lo": np.percentile(bb, 2.5), "hi": np.percentile(bb, 97.5),
                             "n_boot": len(bb)})
    pd.DataFrame(sens).to_csv(out / "sensitivity_n_genes.tsv", sep="\t", index=False)

    show = tab[tab.quantity.isin(["total", "score_t2d", "prop_signature", "indirect_signature", "prop_predicted",
                                  "prop_both", "lmm_prop_signature", "within_r_all", "donor_rho_all"])]
    with pd.option_context("display.width", 200):
        print(show[["endpoint", "part", "quantity", "estimate", "lo", "hi", "p_perm"]].round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/09c_t2d_hidden/predictions.tsv.gz"],
                     [out / f for f in ("cell_scores.tsv.gz", "signature_genes_by_fold.tsv", "de_all_dev_donors.tsv",
                                        "mediation.tsv", "sensitivity_n_genes.tsv")], seed=cfg["seed"])
    print(f"done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
