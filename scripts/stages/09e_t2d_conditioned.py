"""Stage 09e：以细胞自身转录组为条件后，T2D 的功能差异是否仍然存在。

问题：T2D 的胞吐缺陷是否由转录状态携带。与 Stage 09c 不同，这里不要求模型能预测胞吐：
若 T2D 细胞的胞吐差异来自某种转录状态（包括只在 T2D 中出现的状态），把细胞转录组作为条件后，
T2D 系数应当缩小。T2D 供体的细胞参与主成分与交叉拟合，因此 T2D 特有的转录轴也能被吸收。
阳性对照：转录组编码的终点（Na+ 电流、细胞大小）的 T2D 差异应被吸收；
半合成检验：令同样大小的缺陷由留出供体推导的 T2D 转录特征携带，检验该方法能否识别。
输出 results/09e_t2d_conditioned/{effects.tsv, synthetic.tsv, recognisability.tsv, pc_variance.tsv}。
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from t2d_conditioned import (EXO, boot_df, crossfit, dml_theta, hvg_matrix, parts, pc_effects, pcs,  # noqa: E402
                             perm_df, summarise, synthetic_y)
from t2d_hidden import nd_sd  # noqa: E402
from t2d_signature import fold_scores  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09e_t2d_conditioned"
MAIN_SITES = ["ADI IsletCore", "HPAP"]


def prepare(cfg: dict, ct: str) -> dict:
    """一种细胞类型的表达表示、特征得分与各终点的细胞表。"""
    c = cfg["t2d_conditioned"]
    sc = cfg["t2d_signature"]
    td = load_type(cfg, ct, "all")
    status = td.cells.status.values
    site = td.cells.site.values
    site_main = np.where(np.isin(site, MAIN_SITES), site, "Other")
    ok = np.isin(status, ["ND", "preT2D", "T2D"])
    rows = np.where(ok)[0]
    z_ok = hvg_matrix(td.xlog[rows], c["n_hvg"])
    pc_ok, var = pcs(z_ok, max(c["n_pcs"]), cfg["seed"])
    z = np.zeros((len(status), z_ok.shape[1]))
    z[rows] = z_ok
    pc = np.zeros((len(status), pc_ok.shape[1]))
    pc[rows] = pc_ok
    dev = site != cfg["cv"]["external_site"]
    s, _ = fold_scores(td, status, site, dev, sc["n_genes"], sc["min_cpm"], cfg["cv"]["outer_folds"], cfg["seed"])
    ud = np.unique(td.donor[ok])
    fd = donor_folds(ud, c["folds"], cfg["seed"])
    folds = dict(zip(ud, fd))
    frames = {}
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        m = ok & ~np.isnan(y) & ~np.isnan(s)
        d = pd.DataFrame({"y": y, "s": s, "g": td.group, "status": status, "donor": td.donor, "site": site_main,
                          "glucose": td.glucose, "cell_row": np.arange(len(y))})[m]
        d = d[d.groupby("g").y.transform("size") >= 2].reset_index(drop=True)
        d["t2d"] = (d.status == "T2D").astype(float)
        d["pre"] = (d.status == "preT2D").astype(float)
        for part, sub in parts(d, ep).items():
            frames[(ep, part)] = sub
    return {"z": z, "pc": pc, "var": var, "folds": folds, "frames": frames}


def run_part(df: pd.DataFrame, pc, z, folds, c: dict, seed: int, synth: bool) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    ks, main = c["n_pcs"], c["main_pcs"]
    alphas = np.asarray(c["ridge_alphas"], float)
    point = pc_effects(df, pc, ks)
    ry, rt = crossfit(df, z, folds, alphas)
    df = df.assign(ry=ry, rt=rt)

    def dml(d, tot):
        th = dml_theta(d.ry.values, d.rt.values, d.donor.values)
        return {"direct_dml": th, "indirect_dml": tot - th, "prop_dml": (tot - th) / tot if tot != 0 else np.nan}

    point |= dml(df, point["total"])
    boots = []
    for _ in range(c["n_boot"]):
        b = boot_df(df, rng)
        e = pc_effects(b, pc, ks)
        boots.append(e | dml(b, e["total"]))
    tested = ("total", f"direct_pc{main}")
    perms = []
    for _ in range(c["n_perm"]):
        e = pc_effects(perm_df(df, rng), pc, [main])
        perms.append({q: e[q] for q in tested})
    rows = summarise(point, boots, perms, tested)
    syn = []
    if synth:
        for pi in c["synthetic_pi"]:
            ys = synthetic_y(df, pi)
            pt = pc_effects(df, pc, [main], y=ys)
            ry_s, _ = crossfit(df, z, folds, alphas, y=ys, rt=rt)
            th = dml_theta(ry_s, rt, df.donor.values)
            bs = []
            for _ in range(c["n_boot_synthetic"]):
                b = boot_df(df, rng)
                bs.append(pc_effects(b, pc, [main], y=synthetic_y(b, pi))[f"prop_pc{main}"])
            bs = np.asarray(bs)
            bs = bs[np.isfinite(bs)]
            syn.append({"pi": pi, "method": f"pc{main}", "estimate": pt[f"prop_pc{main}"],
                        "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "n_boot": len(bs),
                        "share_boot_above_half": float(np.mean(bs > 0.5))})
            syn.append({"pi": pi, "method": "dml", "estimate": (pt["total"] - th) / pt["total"], "lo": np.nan,
                        "hi": np.nan, "n_boot": 0, "share_boot_above_half": np.nan})
    return rows, syn


def recognisability(prep: dict, c: dict) -> list[dict]:
    """交叉拟合的 T2D 指示预测（转录组 + 协变量）在细胞与供体层面的 AUC（描述用）。"""
    df = prep["frames"][("size", "all cells")]
    df = df[df.status.isin(["ND", "T2D"])].reset_index(drop=True)
    _, rt = crossfit(df, prep["z"], prep["folds"], np.asarray(c["ridge_alphas"], float))
    pred = df.t2d.values - rt
    dm = pd.DataFrame({"donor": df.donor, "t": df.t2d, "p": pred}).groupby("donor").mean()
    return [{"level": "cell", "auc": roc_auc_score(df.t2d, pred), "n": len(df)},
            {"level": "donor", "auc": roc_auc_score(dm.t, dm.p), "n": len(dm)}]


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    c = cfg["t2d_conditioned"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    preps = {ct: prepare(cfg, ct) for ct in cfg["dataset"]["cell_types"]}
    print(f"prepared: {time.time() - t0:.0f}s", flush=True)
    tasks = [(ct, key) for ct, p in preps.items() for key in p["frames"]]
    res = Parallel(n_jobs=min(len(tasks), c["n_jobs"]))(
        delayed(run_part)(preps[ct]["frames"][key], preps[ct]["pc"], preps[ct]["z"], preps[ct]["folds"], c,
                          cfg["seed"] + i, key[0] in EXO) for i, (ct, key) in enumerate(tasks))
    print(f"effects: {time.time() - t0:.0f}s", flush=True)
    eff, syn = [], []
    for (ct, (ep, part)), (rows, sy) in zip(tasks, res):
        df = preps[ct]["frames"][(ep, part)]
        st = df.groupby("donor").status.first()
        meta = {"cell_type": ct, "endpoint": ep, "part": part, "nd_sd": nd_sd(df), "n_cells": len(df),
                "n_nd_donors": int((st == "ND").sum()), "n_t2d_donors": int((st == "T2D").sum())}
        eff += [meta | r for r in rows]
        syn += [meta | r for r in sy]
    eff = pd.DataFrame(eff)
    lin = eff.quantity.str.match(r"^(total|direct_|indirect_)")
    for col in ("estimate", "lo", "hi"):
        eff.loc[lin, col + "_sd"] = eff.loc[lin, col] / eff.loc[lin, "nd_sd"]
        pos = lin & (eff.part == "positive responses")
        eff.loc[pos, col + "_ratio"] = np.exp(eff.loc[pos, col])
    eff.to_csv(out / "effects.tsv", sep="\t", index=False)
    pd.DataFrame(syn).to_csv(out / "synthetic.tsv", sep="\t", index=False)
    rec = [{"cell_type": ct} | r for ct, p in preps.items() for r in recognisability(p, c)]
    pd.DataFrame(rec).to_csv(out / "recognisability.tsv", sep="\t", index=False)
    pv = [{"cell_type": ct, "pc": i + 1, "variance_explained": v} for ct, p in preps.items()
          for i, v in enumerate(p["var"])]
    pd.DataFrame(pv).to_csv(out / "pc_variance.tsv", sep="\t", index=False)
    main_q = ["total", f"direct_pc{c['main_pcs']}", f"prop_pc{c['main_pcs']}", "prop_dml"]
    with pd.option_context("display.width", 220):
        print(eff[eff.quantity.isin(main_q)][["cell_type", "endpoint", "part", "quantity", "estimate", "lo", "hi",
                                               "p_perm"]].round(3).to_string())
        print(pd.DataFrame(syn).round(3).to_string())
        print(pd.DataFrame(rec).round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv"],
                     [out / f for f in ("effects.tsv", "synthetic.tsv", "recognisability.tsv", "pc_variance.tsv")],
                     seed=cfg["seed"])
    print(f"done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
