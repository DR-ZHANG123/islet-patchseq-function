"""Stage 09：2 型糖尿病中的胰岛细胞功能（全部 Patch-seq 供体，ND / 糖尿病前期 / T2D，无 T1D）。

A. 两部分模型（每个终点、每种细胞类型）：
   (1) 非零反应概率：logit P(y>0) ~ T2D + 糖尿病前期 + 葡萄糖 + 来源中心 + 供体随机截距 → T2D 比值比；
   (2) 正值部分：log y ~ 同上协变量 + 供体随机截距（REML）→ exp(β) 为正值几何均值之比（按比例变化）。
   另给出 log1p 后葡萄糖内标准化尺度上的均值差（SD 单位），以及供体内 log 正值离散度之比。
B. 分布形状：按供体等权（细胞权重 1/n_d）的分位数回归，原始尺度含零值与正值部分各给 T2D/ND 分位数比；
   上尾增强检验 = q90 比值 / q50 比值（=1 表示等比例下降，<1 表示上尾额外降低）。
C. 高功能细胞比例：高于同葡萄糖、同中心 ND 细胞上分位数的 T2D 细胞比例，与"正值按比例缩放"所预测比例之差。
推断：置信区间来自 状态 × 中心 分层的供体重抽样；P 值来自中心内 ND/T2D 供体标签置换；
     同一效应在 终点 × 细胞类型 间做 Benjamini–Hochberg 校正。
D. 稳健性（β 细胞）：各中心单独、仅 5 mM、调整年龄/性别/BMI、仅临床诊断 T2D、IsletCore 加分离/培养变量、
   细胞等权；IsletCore 与 HPAP 的效应之比（中心 × T2D 交互）。
E. 转录组–功能对应：对称的留出供体交叉验证（只用 ND 供体训练，T2D/前期供体从不进入训练），
   T2D − ND 供体内精度差及等效性界。
输出 results/09_t2d_function/{t2d_effects.tsv, t2d_robustness.tsv, coupling_by_status.tsv, coupling_differences.tsv}。
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
from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
from t2d_coupling import coupling_inference, predict_symmetric  # noqa: E402
from t2d_effects import add_isolation, load_cells  # noqa: E402
from t2d_models import bh  # noqa: E402
from t2d_resample import resample, summarize  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09_t2d_function"
ROBUST_LABELS = {"isletcore": "ADI IsletCore donors", "hpap": "HPAP donors", "glucose5": "5 mM glucose only",
                 "age_sex_bmi": "adjusted for age, sex and BMI", "clinical_t2d": "clinically diagnosed T2D only",
                 "isletcore_isolation": "ADI IsletCore, adjusted for isolation and culture",
                 "cell_weighted": "cells weighted equally"}
OLD_OUTPUTS = ["quantile_effects.tsv", "high_function_fraction.tsv", "quantile_sensitivity.tsv"]


def base_spec(tc: dict, df: pd.DataFrame) -> dict:
    return {"zero_part": True, "min_zero_cells": tc["min_zero_cells"], "gh_nodes": tc["gh_nodes"],
            "logit_ridge": tc["logit_ridge"], "min_sd_df": tc["min_sd_df"], "weighting": tc["weighting"],
            "floor": float(df.y[df.y > 0].min()) / 2, "qs_all": tc["quantiles"], "qs_pos": tc["quantiles"],
            "tail": tc["tail_quantiles"], "thresholds": tc["high_thresholds"],
            "min_nd_threshold_cells": tc["min_nd_threshold_cells"], "covariates": ()}


def counts(df: pd.DataFrame) -> dict:
    st = df.groupby("donor").status.first()
    return {"n_cells": len(df), "n_donors": len(st), "n_nd_donors": int((st == "ND").sum()),
            "n_t2d_donors": int((st == "T2D").sum()), "n_pre_donors": int((st == "preT2D").sum()),
            "zero_fraction_nd": float((df[df.status == "ND"].y == 0).mean()),
            "zero_fraction_t2d": float((df[df.status == "T2D"].y == 0).mean())}


def variant(df: pd.DataFrame, name: str, spec: dict, tc: dict) -> tuple[pd.DataFrame, dict]:
    """稳健性分析的数据子集与模型设定。"""
    spec = dict(spec, qs_all=tc["robustness_quantiles"], qs_pos=tc["robustness_quantiles"], thresholds=[])
    if name == "isletcore":
        return df[df.site == "ADI IsletCore"], spec
    if name == "hpap":
        return df[df.site == "HPAP"], spec
    if name == "glucose5":
        return df[df.glucose == 5], spec
    if name == "age_sex_bmi":
        return df.dropna(subset=["age", "bmi", "sex_male"]), dict(spec, covariates=("age", "bmi", "sex_male"))
    if name == "clinical_t2d":
        return df[~((df.status == "T2D") & (df.clinical != "T2D"))], spec
    if name == "isletcore_isolation":
        cov = tc["isolation_covariates"]
        sub = add_isolation(df[df.site == "ADI IsletCore"], cov).dropna(subset=cov)
        return sub, dict(spec, covariates=tuple(cov))
    if name == "cell_weighted":
        return df, dict(spec, weighting="cell")
    raise KeyError(name)


def run_effects(cfg: dict, tc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows, rob_rows = [], []
    job = 0
    for ct in cfg["dataset"]["cell_types"]:
        for ep in tc["endpoints"]:
            t0 = time.time()
            df = load_cells(ct, ep)
            spec = base_spec(tc, df)
            point, boots, perms = resample(df, spec, tc["n_boot"], tc["n_perm"], cfg["seed"] + job, tc["n_jobs"])
            job += 1
            main_rows.append(summarize(point, boots, perms).assign(analysis="all donors", cell_type=ct, endpoint=ep,
                                                                   **counts(df)))
            print(f"[main] {ct} {ep}: {time.time() - t0:.0f}s", flush=True)
            if ct != "Beta" or ep not in tc["robustness_endpoints"]:
                continue
            kept = {}
            for name in tc["robustness"]:
                t0 = time.time()
                sub, sp = variant(df, name, spec, tc)
                sub = sub.reset_index(drop=True)
                point, boots, perms = resample(sub, sp, tc["n_boot"], tc["n_perm"], cfg["seed"] + job, tc["n_jobs"])
                job += 1
                kept[name] = (point, boots)
                rob_rows.append(summarize(point, boots, perms).assign(analysis=ROBUST_LABELS[name], cell_type=ct,
                                                                      endpoint=ep, **counts(sub)))
                print(f"[robust] {ct} {ep} {name}: {time.time() - t0:.0f}s", flush=True)
            rob_rows.append(centre_interaction(kept["isletcore"], kept["hpap"]).assign(cell_type=ct, endpoint=ep))
    main = pd.concat(main_rows, ignore_index=True)
    main["p_bh"] = np.nan
    for eff, idx in main.groupby("effect").groups.items():
        main.loc[idx, "p_bh"] = bh(main.loc[idx, "p_perm"].values)
    return main, pd.concat(rob_rows, ignore_index=True)


def centre_interaction(a: tuple, b: tuple) -> pd.DataFrame:
    """IsletCore 与 HPAP 的 T2D 效应之比（对数尺度差；两中心供体互不重叠，重抽样独立配对）。"""
    rows = []
    for k in a[0]:
        if not (k.endswith("_logratio") or k.endswith("_logor")) or k not in b[0]:
            continue
        d = a[1][k].values - b[1][k].values
        d = d[np.isfinite(d)]
        est = a[0][k] - b[0][k]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (1 + len(d)))
        rows.append({"effect": k.replace("_logratio", "_ratio").replace("_logor", "_odds_ratio"), "scale": "ratio",
                     "estimate": np.exp(est), "lo": np.exp(np.percentile(d, 2.5)), "hi": np.exp(np.percentile(d, 97.5)),
                     "p_boot": p, "n_boot": len(d), "analysis": "ADI IsletCore / HPAP (centre × T2D)"})
    return pd.DataFrame(rows)


def run_coupling(cfg: dict, tc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "all")
        jobs += [(ct, td, td.endpoints.index(ep)) for ep in tc["coupling_endpoints"][ct]]
    # 每个模型 4 个线程 × 8 个进程，总 CPU 不超过 32
    with parallel_config(backend="loky", inner_max_num_threads=tc["coupling_threads"]):
        preds = Parallel(n_jobs=tc["n_jobs"] // tc["coupling_threads"])(
            delayed(predict_symmetric)(td, cfg, k, cfg["seed"]) for _, td, k in jobs)
    inf = Parallel(n_jobs=min(len(jobs), tc["n_jobs"]))(delayed(coupling_inference)(p, tc["coupling_n_boot"], tc["equivalence_margin"],
                                                                 cfg["seed"]) for p in preds)
    diffs = pd.concat([r.assign(cell_type=ct, endpoint=p.endpoint.iloc[0])
                       for (ct, _, _), p, r in zip(jobs, preds, inf)], ignore_index=True)
    by = diffs[diffs.quantity.str.startswith("r_")].copy()
    by = by.assign(status=by.quantity.str[2:], within_pearson=by.estimate,
                   training="ND donors only; donor-held-out folds shared by all statuses")
    by = by[["endpoint", "status", "within_pearson", "lo", "hi", "n_cells", "n_donors", "training", "cell_type"]]
    return by, diffs[~diffs.quantity.str.startswith("r_")].drop(columns=["n_cells", "n_donors"])


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    tc = cfg["t2d"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        main_tab, rob = run_effects(cfg, tc)
    files = {"t2d_effects.tsv": main_tab, "t2d_robustness.tsv": rob}
    for fn, df in files.items():
        df.to_csv(out / fn, sep="\t", index=False)
    by, diffs = run_coupling(cfg, tc)
    files.update({"coupling_by_status.tsv": by, "coupling_differences.tsv": diffs})
    for fn in ("coupling_by_status.tsv", "coupling_differences.tsv"):
        files[fn].to_csv(out / fn, sep="\t", index=False)
    for fn in OLD_OUTPUTS:
        (out / fn).unlink(missing_ok=True)
    show = main_tab[main_tab.effect.isin(["nonzero_odds_ratio", "positive_ratio", "q50_all_ratio", "q90_all_ratio",
                                          "tail_pos_ratio", "mean_sd"])]
    print(show[["cell_type", "endpoint", "effect", "estimate", "lo", "hi", "p_perm", "p_bh"]].round(4).to_string())
    print(diffs.round(3).to_string())
    raw = ROOT / "data/raw/humanislets"
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv",
                             raw / "isolation.csv", raw / "donor.csv"], [out / f for f in files], seed=cfg["seed"])


if __name__ == "__main__":
    main()
