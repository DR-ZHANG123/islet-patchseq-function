"""Stage 11：供体层面，Patch-seq 单细胞功能（含转录组读不出的偶联效率）与同一供体胰岛胰岛素分泌的关系。

问题：β 细胞单细胞胞吐（实测正值反应、Ca²⁺–分泌偶联效率、转录组预测值）是否反映同一供体胰岛的胰岛素分泌
（灌流：3 mM 基线、6/15 mM 葡萄糖、KCl、亮氨酸、油酸/棕榈酸 AUC 与葡萄糖刺激倍数；静态 GSIS：2.8 → 16.7 mM
刺激指数与分泌占含量百分比）。α 细胞同样汇总，作为对照。汇总与统计方法见 scripts/lib/islet_secretion.py。
另报告这些供体中胰岛分泌的 T2D − ND 差（背景）。
输出 results/11_islet_secretion/{donor_values.tsv, correlations.tsv, status_differences.tsv, fig_source.tsv}。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import islet_secretion as isl  # noqa: E402

STAGE = "11_islet_secretion"


def inputs(cfg: dict) -> dict[str, Path]:
    raw = ROOT / cfg["paths"]["raw"]
    out = {"perifusion": raw / "humanislets_omics" / "function_summary.csv",
           "gsis": raw / "humanislets" / "gsis.csv",
           "donors": ROOT / "results/01_dataset/donors.tsv", "cells": ROOT / "results/01_dataset/cells.tsv"}
    for ct in cfg["islet_secretion"]["cell_types"]:
        out[f"pred_{ct}"] = ROOT / cfg["paths"]["interim"] / "03_benchmark" / f"pred_{ct}_donors.parquet"
    return out


def donor_values(cfg: dict, sc: dict, paths: dict, status: pd.Series) -> pd.DataFrame:
    """长表：供体 × 细胞类型 × 特征（单细胞汇总）与供体 × 胰岛分泌终点。"""
    islet = isl.islet_secretion(pd.read_csv(paths["perifusion"]), pd.read_csv(paths["gsis"]), sc)
    frames = []
    for ct in sc["cell_types"]:
        td = load_type(cfg, ct, "all")
        pred = pd.read_parquet(paths[f"pred_{ct}"], filters=[("model", "==", sc["prediction_model"])])
        f = isl.cell_function(td, cfg["coupling"], pred, sc["min_cells"])[sc["features"]]
        frames.append(f.rename_axis("donor").reset_index().melt("donor", var_name="feature", value_name="x")
                      .assign(cell_type=ct))
    cell = pd.concat(frames, ignore_index=True).dropna(subset=["x"])
    isl_long = islet.rename_axis("donor").reset_index().melt("donor", var_name="outcome", value_name="y").dropna()
    df = cell.merge(isl_long, on="donor")
    df["status"] = df.donor.map(status)
    return df[df.status.isin(sc["statuses"])].reset_index(drop=True), islet


def correlations(df: pd.DataFrame, sc: dict, seed: int) -> pd.DataFrame:
    keys = [k for k, g in df.groupby(["cell_type", "feature", "outcome"], sort=False) if len(g) >= sc["min_donors"]]
    grp = df.groupby(["cell_type", "feature", "outcome"], sort=False)
    res = Parallel(n_jobs=sc["n_jobs"])(
        delayed(isl.infer_pair)(grp.get_group(k).x.values, grp.get_group(k).y.values,
                                grp.get_group(k).status.values, sc["n_boot"], sc["n_perm"], seed + i)
        for i, k in enumerate(keys))
    tab = pd.DataFrame([dict(zip(("cell_type", "feature", "outcome"), k)) | r for k, r in zip(keys, res)])
    pr = sc["primary"]
    tab["primary"] = ((tab.cell_type == pr["cell_type"]) & tab.feature.isin(pr["features"])
                      & tab.outcome.isin(pr["outcomes"]))
    for lab in ("rho", "rho_adj"):
        tab[f"{lab}_q_primary"] = np.nan
        m = tab.primary
        tab.loc[m, f"{lab}_q_primary"] = isl.bh(tab.loc[m, f"{lab}_p"].values)
        tab[f"{lab}_q_all"] = np.nan
        for ct in tab.cell_type.unique():
            m = tab.cell_type == ct
            tab.loc[m, f"{lab}_q_all"] = isl.bh(tab.loc[m, f"{lab}_p"].values)
    return tab


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    sc = cfg["islet_secretion"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    paths = inputs(cfg)
    status = pd.read_csv(paths["donors"], sep="\t").set_index("donor").status
    df, islet = donor_values(cfg, sc, paths, status)
    df.to_csv(out / "donor_values.tsv", sep="\t", index=False)
    print(f"donor values: {time.time() - t0:.0f}s, {df.donor.nunique()} donors", flush=True)

    tab = correlations(df, sc, cfg["seed"])
    tab.to_csv(out / "correlations.tsv", sep="\t", index=False)
    print(f"correlations: {time.time() - t0:.0f}s", flush=True)

    rows = []
    isl_st = islet.assign(status=islet.index.map(status))
    isl_st = isl_st[isl_st.status.isin(["ND", "T2D"])]
    for i, col in enumerate(islet.columns):
        d = isl_st[[col, "status"]].dropna()
        rows.append({"outcome": col} | isl.status_difference(d[col].values, d.status.values, sc["n_boot"],
                                                                sc["n_perm"], cfg["seed"] + 900 + i))
    sd = pd.DataFrame(rows)
    sd.to_csv(out / "status_differences.tsv", sep="\t", index=False)

    # 作图输入：主要检验的散点（每位供体一点）+ 对应统计量
    pr = sc["primary"]
    fig = df[(df.cell_type == pr["cell_type"]) & df.feature.isin(pr["features"]) & df.outcome.isin(pr["outcomes"])]
    fig = fig.merge(tab[["cell_type", "feature", "outcome", "n_donors", "n_t2d", "rho", "rho_lo", "rho_hi", "rho_p",
                         "rho_adj", "rho_adj_lo", "rho_adj_hi", "rho_adj_p", "rho_adj_q_primary"]],
                    on=["cell_type", "feature", "outcome"])
    fig.to_csv(out / "fig_source.tsv", sep="\t", index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        cols = ["cell_type", "feature", "outcome", "n_donors", "n_t2d", "rho", "rho_p", "rho_adj", "rho_adj_lo",
                "rho_adj_hi", "rho_adj_p", "rho_adj_q_primary", "rho_adj_q_all"]
        print(tab[tab.primary][cols].round(3).to_string())
        print(tab.sort_values("rho_adj_p").head(10)[cols].round(3).to_string())
        print(sd.round(3).to_string())
    write_provenance(STAGE, list(paths.values()),
                     [out / f for f in ("donor_values.tsv", "correlations.tsv", "status_differences.tsv",
                                        "fig_source.tsv")], seed=cfg["seed"])
    print(f"done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
