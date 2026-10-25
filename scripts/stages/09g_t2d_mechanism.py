"""Stage 09g：T2D β 细胞胞吐缺陷的定位与候选转录携带者。

1. 精选胞吐/颗粒基因集（GO：Ca2+ 调控胞吐、胰岛素分泌、囊泡停靠与引发、致密核心颗粒运输，加上点名的 SNARE、
   颗粒与 ETV1 等基因）为条件的携带比例：基因集主成分、其供体均值、交叉拟合岭回归、ETV1 单基因。
2. ETV1：T2D 差与 ND、T2D 细胞中与胞吐的供体内相关。
3. 模拟：缺陷由基因集中 T2D 差异最大的 10 个基因携带时，各估计量找回的比例。
4. 对数尺度 Ca2+ 进入：同一组正值反应细胞中 Ca2+ 进入的 T2D 比，及按供体内信度回归校准后的胞吐 T2D 比。
5. 早期与晚期正值反应 T2D 比之差；各葡萄糖浓度内的 T2D 比。
6. ND 训练模型供体内负相关的来源（混合、供体内、供体间三种训练目标）。
6b. 允许特征得分–胞吐关系在 T2D 与 ND 细胞中不同（交互项）时的自然间接效应与携带比例。
7. 胰岛胰岛素含量（每 IEQ、胰岛素/DNA 比）的 T2D 与 ND 比。
输出 results/09g_t2d_mechanism/*.tsv。
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
from t2d_mech import (boot_panel, candidate_stats, early_late, glucose_strata, panel_carrier, panel_genes,  # noqa: E402
                      panel_pcs, read_gmt, ridge_share, simulate_panel, standardise)
from t2d_mech_diag import insulin_content, nd_trained_variants, within_r_boot  # noqa: E402
from t2d_mediation import boot_decompose  # noqa: E402
from t2d_robust import boot_generic, boot_shares, ca_models, ca_reliability  # noqa: E402
from t2d_robust_data import prepare  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09g_t2d_mechanism"
KEYS = [("exocytosis", "positive responses"), ("late_exocytosis", "positive responses"),
        ("exocytosis", "all cells"), ("early_exocytosis", "positive responses")]


def tag(rows, **kw):
    return [kw | r for r in rows]


def donor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """供体平均特征得分（外部与留出供体特征），以同葡萄糖 ND 细胞为参照的 ND 细胞 SD 单位（同 Stage 09d/09f 的单位）。"""
    out = []
    for col, lab in (("s_ext", "external"), ("s", "held_out")):
        v = df[col].values
        nd = df.status.values == "ND"
        mu = df[nd].groupby("glucose")[col].mean()
        sd = float((df[nd][col] - df[nd].glucose.map(mu)).std())
        z = (v - df.glucose.map(mu).values) / sd
        d = df.assign(z=z).groupby("donor").agg(score=("z", "mean"), n_cells=("z", "size"), status=("status", "first"),
                                                 site=("site", "first")).reset_index()
        out.append(d.assign(signature=lab))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    mc, seed = cfg["t2d_mechanism"], cfg["seed"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ext = ROOT / cfg["t2d_robust"]["external_file"]
    prep = prepare(cfg, "Beta", ext_path=ext)
    td = prep["td"]
    ok = np.where(np.isin(prep["status"], ["ND", "preT2D", "T2D"]))[0]
    gmt = read_gmt(ROOT / cfg["paths"]["raw"] / "genesets" / "GO_Biological_Process_2025.gmt")
    cols, gtab = panel_genes(gmt, mc["panel_terms"], mc["panel_extra"], td.symbols, td.xlog, ok,
                             mc["panel_min_detect"])
    zp = standardise(td.xlog, cols, ok)
    P = panel_pcs(zp, ok, mc["panel_pcs"], seed)
    sym = pd.Series(np.arange(len(td.symbols)), index=td.symbols)
    cand = standardise(td.xlog, np.array([int(sym[mc["candidate_gene"]])]), ok)[:, 0]
    print(f"prepared {time.time() - t0:.0f}s; panel {len(cols)} of {len(gtab)} genes", flush=True)
    files = []

    def save(df, name):
        df.to_csv(out / name, sep="\t", index=False)
        files.append(out / name)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    save(gtab.drop(columns="col"), "panel_genes.tsv")
    fr = prep["frames"]
    kb, nb = mc["panel_between_pcs"], mc["n_boot"]
    res = Parallel(n_jobs=len(KEYS))(delayed(boot_panel)(fr[k], P, cand, kb, nb, np.random.default_rng(seed + i))
                                     for i, k in enumerate(KEYS))
    rid = Parallel(n_jobs=len(KEYS))(delayed(ridge_share)(fr[k], zp, prep["folds"], cfg["t2d_conditioned"]["ridge_alphas"],
                                                          nb, np.random.default_rng(seed + 10 + i))
                                     for i, k in enumerate(KEYS))
    rows = []
    for k, a, b in zip(KEYS, res, rid):
        d = fr[k]
        st = d.groupby("donor").status.first()
        rows += tag(a + [r for r in b if r["quantity"] != "total"], endpoint=k[0], part=k[1], n_cells=len(d),
                    n_nd_donors=int((st == "ND").sum()), n_t2d_donors=int((st == "T2D").sum()))
    save(pd.DataFrame(rows), "panel_conditioning.tsv")

    cs = Parallel(n_jobs=len(KEYS))(delayed(candidate_stats)(fr[k], cand, nb, np.random.default_rng(seed + 20 + i))
                                    for i, k in enumerate(KEYS))
    save(pd.DataFrame([r | {"endpoint": k[0], "part": k[1], "gene": mc["candidate_gene"]}
                       for k, rr in zip(KEYS, cs) for r in rr]), "candidate_gene.tsv")

    key = ("exocytosis", "positive responses")
    carrier, top = panel_carrier(fr[key], zp, mc["sim_n_genes"])
    sims = Parallel(n_jobs=2)(delayed(simulate_panel)(fr[k], P, cand, kb, prep["pc"][:, :50], carrier, mc["sim_pi"],
                                                      mc["n_boot_sim"], np.random.default_rng(seed + 30 + i))
                              for i, k in enumerate(KEYS[:2]))
    genes = ";".join(gtab[gtab.kept].gene.values[top])
    save(pd.DataFrame([r | {"endpoint": k[0], "part": k[1], "carrier_genes": genes}
                       for k, rr in zip(KEYS[:2], sims) for r in rr]), "panel_simulation.tsv")

    ca_rows, rel = [], []
    for i, k in enumerate(KEYS):
        d = fr[k][np.isfinite(fr[k].ca_log) & np.isfinite(fr[k].ca_early_log)].reset_index(drop=True)
        r = boot_generic(ca_models, d, nb, np.random.default_rng(seed + 40 + i), lams=mc["ca_lambdas"], col="ca_log")
        ca_rows += tag(r, endpoint=k[0], part=k[1], n_cells=len(d), n_donors=d.donor.nunique(), ca_scale="log")
        rel.append({"endpoint": k[0], "part": k[1]} | ca_reliability(d, "ca_log", "ca_early_log"))
    save(pd.DataFrame(ca_rows), "calcium_log_adjustment.tsv")
    save(pd.DataFrame(rel), "calcium_log_reliability.tsv")

    el = early_late({"early": fr[("early_exocytosis", "positive responses")],
                     "late": fr[("late_exocytosis", "positive responses")]}, 2 * nb, np.random.default_rng(seed + 50))
    save(pd.DataFrame(el), "early_late.tsv")
    gl = [r | {"endpoint": k[0], "part": k[1]} for k in KEYS[:2]
          for r in glucose_strata(fr[k], mc["glucose_levels"], nb, np.random.default_rng(seed + 60))]
    save(pd.DataFrame(gl), "glucose_strata.tsv")

    eps = mc["nd_diag_endpoints"]
    preds = Parallel(n_jobs=len(eps))(delayed(nd_trained_variants)(td, cfg, ep, seed) for ep in eps)
    diag = []
    for p in preds:
        for st, g in p.groupby("status"):
            for v in ("pooled", "within", "between"):
                diag.append({"endpoint": p.endpoint.iloc[0], "status": st, "training_target": v}
                            | within_r_boot(g, f"pc_{v}", 2 * nb, np.random.default_rng(seed + 70)))
    save(pd.DataFrame(diag), "nd_trained_targets.tsv")

    mi_jobs = [(k, col) for k in KEYS[:2] for col in ("s_ext", "s")]
    mi = Parallel(n_jobs=len(mi_jobs))(delayed(boot_decompose)(fr[k], col, nb, np.random.default_rng(seed + 100 + i))
                                       for i, (k, col) in enumerate(mi_jobs))
    save(pd.DataFrame([r | {"endpoint": k[0], "part": k[1], "signature": "external" if col == "s_ext" else "internal"}
                       for (k, col), rr in zip(mi_jobs, mi) for r in rr]), "mediation_interaction.tsv")
    save(donor_scores(fr[("exocytosis", "all cells")]), "donor_scores.tsv")
    # 敏感性：保留只有 1 个细胞的供体 × 葡萄糖组（主分析要求 ≥ 2 个细胞）
    p1 = prepare(cfg, "Beta", ext_path=ext, min_group=1)
    keys1 = [("exocytosis", "positive responses"), ("late_exocytosis", "positive responses")]
    r1 = Parallel(n_jobs=2)(delayed(boot_shares)(p1["frames"][k], p1["pc"], [3, 5], 50, nb, np.random.default_rng(seed + 90 + i))
                            for i, k in enumerate(keys1))
    rows1 = []
    for k, r in zip(keys1, r1):
        d = p1["frames"][k]
        st = d.groupby("donor").status.first()
        rows1 += tag([x for x in r if x["quantity"] in ("total", "prop_full50", "prop_between3", "prop_between5",
                                                         "prop_external_signature", "prop_signature")],
                     endpoint=k[0], part=k[1], n_cells=len(d), n_nd_donors=int((st == "ND").sum()),
                     n_t2d_donors=int((st == "T2D").sum()))
    save(pd.DataFrame(rows1), "single_cell_groups.tsv")
    iso = pd.read_csv(ROOT / cfg["paths"]["raw"] / "humanislets" / "isolation.csv")
    donors = pd.read_csv(ROOT / "results/01_dataset/donors.tsv", sep="\t")
    save(pd.DataFrame(insulin_content(iso, donors, 2 * nb, 5000, np.random.default_rng(seed + 80))),
         "insulin_content.tsv")
    gmt_path = ROOT / cfg["paths"]["raw"] / "genesets" / "GO_Biological_Process_2025.gmt"
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv", gmt_path,
                             ROOT / cfg["paths"]["raw"] / "humanislets" / "isolation.csv", ext], files, seed=seed)
    print(f"done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
