"""Stage 09b：β 细胞胞吐 T2D 效应的敏感性分析（全部 Patch-seq 供体，ND / 糖尿病前期 / T2D）。

A. 反应阈值：反应定义由 y > 0 改为 y > c（c = 0.5、1、2 fF/pF 及 ND 正值的第 10 百分位），重估反应比值比与正值比。
B. 有符号值：检查发布数据是否保留负值；按 总 = 早 + 晚 恢复部分有符号值，随机截距 Tobit（左删失）重估 T2D 效应。
C. 条件关联：在原有协变量（糖尿病前期、葡萄糖、中心、供体随机截距）之外加入 Ca²⁺ 内流、早/晚期 Ca²⁺ 电流与细胞大小，
   同一完整病例子集上比较调整前后的正值比与反应比值比。
D. 等效性界：反应比值比与反应细胞比例差的 90% 置信区间（双单侧检验）给出的最小对称等效界。
推断同 Stage 09：状态 × 中心分层的供体重抽样（置信区间）与中心内 ND/T2D 供体标签置换（P 值）。
输出 results/09b_t2d_sensitivity/{threshold_sensitivity, signed_values, signed_audit, conditional_adjustment,
equivalence_bounds}.tsv。
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
from joblib import parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from t2d_sens import equivalence, load_beta, resample_fn  # noqa: E402
from t2d_signed import signed_audit, signed_values  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09b_t2d_sensitivity"


def thresholds_for(df: pd.DataFrame, sc: dict) -> list[tuple[str, float]]:
    """阈值列表：0（主分析）、固定 fF/pF 阈值、ND 细胞正值的给定分位数（在全部数据上确定一次，重抽样中不变）。"""
    pos_nd = df.y[(df.status == "ND") & (df.y > 0)]
    q = float(np.quantile(pos_nd, sc["nd_positive_quantile"]))
    return [("0 (main)", 0.0)] + [(f"{c:g} fF/pF", float(c)) for c in sc["thresholds"]] + \
        [(f"ND positive P{int(round(sc['nd_positive_quantile'] * 100))}", q)]


def run_thresholds(base: pd.DataFrame, sc: dict, tc: dict, seed: int) -> pd.DataFrame:
    rows, job = [], 0
    for ep in sc["endpoints"]:
        df = base.assign(y=base[ep].values)
        for label, c in thresholds_for(df, sc):
            t0 = time.time()
            spec = {"effect": "two_part", "threshold": c, "min_zero_cells": tc["min_zero_cells"],
                    "gh_nodes": tc["gh_nodes"], "logit_ridge": tc["logit_ridge"]}
            out = resample_fn(df, spec, sc["n_boot"], sc["n_perm"], seed + 100 + job, sc["n_jobs"])
            job += 1
            nd, t2 = df.status == "ND", df.status == "T2D"
            rows.append(out.assign(endpoint=ep, threshold_label=label, threshold=c,
                                   responder_fraction_nd=float((df.y[nd] > c).mean()),
                                   responder_fraction_t2d=float((df.y[t2] > c).mean())))
            print(f"[threshold] {ep} {label}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def run_signed(base: pd.DataFrame, sc: dict, tc: dict, seed: int) -> pd.DataFrame:
    rows = []
    for k, ep in enumerate(sc["endpoints"]):
        ys, cens = signed_values(base, ep)
        df = base.assign(ys=ys, cens=cens)
        t0 = time.time()
        out = resample_fn(df, {"effect": "tobit", "gh_nodes": sc["tobit_gh_nodes"]}, sc["n_boot"], sc["n_perm"],
                          seed + 200 + k, sc["n_jobs"])
        rows.append(out.assign(endpoint=ep, n_negative_recovered=int(((ys < 0) & ~cens).sum()),
                               n_left_censored=int(cens.sum()), n_positive=int((ys > 0).sum())))
        print(f"[signed] {ep}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def run_conditional(base: pd.DataFrame, sc: dict, tc: dict, seed: int) -> pd.DataFrame:
    rows, job = [], 0
    for ep in sc["conditional_endpoints"]:
        for name, covs in sc["conditional_sets"].items():
            cols = [f"cov_{v}" for v in covs]
            df = base.assign(y=base[ep].values).dropna(subset=cols)
            if not np.isfinite(df[cols].values).all():
                df = df[np.isfinite(df[cols].values).all(1)]
            for label, use in (("base covariates (same cells)", ()), (f"+ {' + '.join(covs)}", tuple(cols))):
                t0 = time.time()
                spec = {"effect": "two_part", "threshold": 0.0, "min_zero_cells": tc["min_zero_cells"],
                        "gh_nodes": tc["gh_nodes"], "logit_ridge": tc["logit_ridge"], "covariates": use}
                out = resample_fn(df, spec, sc["n_boot"], sc["n_perm"], seed + 300 + job, sc["n_jobs"])
                job += 1
                rows.append(out.assign(endpoint=ep, covariate_set=name, model=label))
                print(f"[conditional] {ep} {name} {label}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def run_equivalence(thr: pd.DataFrame, sc: dict) -> pd.DataFrame:
    main = pd.read_csv(ROOT / "results/09_t2d_function/t2d_effects.tsv", sep="\t")
    main = main[(main.cell_type == sc["cell_type"]) & main.endpoint.isin(["exocytosis", "early_exocytosis",
                                                                           "late_exocytosis"])]
    a = equivalence(main.assign(analysis="Stage 09 main estimate (5,000 permutations)", threshold=0.0),
                    sc["equivalence_or_margins"], sc["equivalence_share_margins"])
    b = equivalence(thr.assign(analysis="responder threshold " + thr.threshold_label),
                    sc["equivalence_or_margins"], sc["equivalence_share_margins"])
    return pd.concat([a, b], ignore_index=True)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    sc, tc = cfg["t2d_sensitivity"], cfg["t2d"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw/humanislets/ephys_cell.csv"
    audit = pd.DataFrame([signed_audit(pd.read_csv(raw, low_memory=False), sc["signed_tolerance"])])
    base = load_beta(sc["cell_type"], sc["covariate_transform"])
    files = {"signed_audit.tsv": audit}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        files["threshold_sensitivity.tsv"] = run_thresholds(base, sc, tc, cfg["seed"])
        files["conditional_adjustment.tsv"] = run_conditional(base, sc, tc, cfg["seed"])
        files["signed_values.tsv"] = run_signed(base, sc, tc, cfg["seed"])
    files["equivalence_bounds.tsv"] = run_equivalence(files["threshold_sensitivity.tsv"], sc)
    for fn, df in files.items():
        df.to_csv(out / fn, sep="\t", index=False)
    keep = ["endpoint", "effect", "estimate", "lo", "hi", "p_perm"]
    print(audit.T.to_string())
    t = files["threshold_sensitivity.tsv"]
    print(t[["threshold_label"] + keep].round(4).to_string())
    c = files["conditional_adjustment.tsv"]
    print(c[["covariate_set", "model", "n_cells"] + keep].round(4).to_string())
    print(files["signed_values.tsv"][keep + ["n_negative_recovered", "n_left_censored"]].round(4).to_string())
    print(files["equivalence_bounds.tsv"].round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", raw,
                             ROOT / "results/09_t2d_function/t2d_effects.tsv"], [out / f for f in files],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
