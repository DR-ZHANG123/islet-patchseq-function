"""Stage 09h：β 细胞 T2D 胞吐效应按记录时葡萄糖浓度分层（与 Stage 09 主分析相同的两部分模型与供体层面推断）。

各浓度（1、5、10 mM）单独拟合两部分模型（非零反应的随机截距 logistic + 正值对数的随机截距线性混合模型），
区间来自状态 × 中心分层的供体重抽样，P 值来自中心内供体状态置换；另列各浓度下各中心的供体数，
因为 HPAP 的记录全部在 5 mM，浓度与来源中心部分重合。
输出 results/09h_t2d_glucose/{glucose_two_part.tsv, glucose_donors.tsv}。
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from t2d_effects import load_cells  # noqa: E402
from t2d_resample import resample, summarize  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09h_t2d_glucose"
ENDPOINTS = ["exocytosis", "early_exocytosis", "late_exocytosis"]


def spec_for(tc: dict, df: pd.DataFrame) -> dict:
    """同 Stage 09 的 base_spec，分位数只取稳健性分析用的两个。"""
    return {"zero_part": True, "min_zero_cells": tc["min_zero_cells"], "gh_nodes": tc["gh_nodes"],
            "logit_ridge": tc["logit_ridge"], "min_sd_df": tc["min_sd_df"], "weighting": tc["weighting"],
            "floor": float(df.y[df.y > 0].min()) / 2, "qs_all": tc["robustness_quantiles"],
            "qs_pos": tc["robustness_quantiles"], "tail": tc["tail_quantiles"], "thresholds": [],
            "min_nd_threshold_cells": tc["min_nd_threshold_cells"], "covariates": ()}


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    tc = cfg["t2d"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    rows, don = [], []
    job = 900
    for ep in ENDPOINTS:
        df = load_cells("Beta", ep)
        spec = spec_for(tc, df)
        for g in cfg["t2d_mechanism"]["glucose_levels"]:
            t0 = time.time()
            sub = df[df.glucose == g].reset_index(drop=True)
            point, boots, perms = resample(sub, spec, tc["n_boot"], tc["n_perm"], cfg["seed"] + job, tc["n_jobs"])
            job += 1
            st = sub.groupby("donor").status.first()
            rows.append(summarize(point, boots, perms).assign(cell_type="Beta", endpoint=ep, glucose=g,
                                                              n_cells=len(sub), n_nd_donors=int((st == "ND").sum()),
                                                              n_t2d_donors=int((st == "T2D").sum())))
            if ep == ENDPOINTS[0]:
                tab = sub.groupby(["site", "status"]).donor.nunique().rename("n_donors").reset_index()
                don.append(tab.assign(glucose=g))
            print(f"{ep} {g} mM: {time.time() - t0:.0f}s", flush=True)
    files = []
    for name, df in (("glucose_two_part.tsv", pd.concat(rows, ignore_index=True)),
                     ("glucose_donors.tsv", pd.concat(don, ignore_index=True))):
        df.to_csv(out / name, sep="\t", index=False)
        files.append(out / name)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv"], files,
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
