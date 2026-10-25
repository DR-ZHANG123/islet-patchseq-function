"""Stage 04e：训练规模对照——把 α 细胞下采样到 β 细胞的数据规模后，α 细胞胞吐是否仍可在供体内预测。

每次下采样：从 α 开发供体中随机取与 β 开发供体相同数目的供体，按细胞数秩次与 β 供体的每供体细胞数配对
（不足时取全部），在每位供体内随机抽取细胞；随后在该子集上按 Stage 03 的流程做 10 折留出供体交叉验证
（每次下采样一个新的折划分），训练与评价规模都与 β 细胞相当。
输出 results/04e_training_size/{metrics_by_draw.tsv, summary.tsv, subsample_sizes.tsv}。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import TypeData, donor_folds, load_type  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402

STAGE = "04e_training_size"


def subsample(td: TypeData, target_counts: np.ndarray, frac: float, rng) -> np.ndarray:
    """返回入选细胞的行号：随机取 len(target_counts) 位供体，按细胞数秩次配对目标细胞数（×frac）。"""
    counts = pd.Series(td.donor).value_counts()
    pick = rng.choice(counts.index.values, min(len(target_counts), len(counts)), replace=False)
    pick = counts.loc[pick].sort_values(ascending=False).index.values
    tgt = np.sort(target_counts)[::-1][:len(pick)]
    rows = []
    for d, n in zip(pick, tgt):
        idx = np.where(td.donor == d)[0]
        rows.append(rng.choice(idx, min(len(idx), int(round(n * frac))), replace=False))
    return np.sort(np.concatenate(rows))


def subset(td: TypeData, idx: np.ndarray) -> TypeData:
    return TypeData(td.cells.iloc[idx].reset_index(drop=True), td.xlog[idx], td.symbols, td.y[idx], td.endpoints,
                    td.donor[idx], td.group[idx], td.glucose[idx], td.counts[idx])


def job(sub, cfg, frac, draw, fold, folds):
    robust_cache.install()
    seed = cfg["seed"] + 7000 + 100 * draw
    p, _ = rc.run_fold(sub, cfg, fold, folds, seed + 37 * fold, sub.endpoints, tag={"fraction": frac, "draw": draw})
    robust_cache.clear()
    p = p[p.model.isin(["context_model", "cell_branch", "ridge", "linear_cell_branch"])]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None, help="并行进程数（缺省取 robustness.jobs）")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    ts = cfg["robustness"]["training_size"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    td = load_type(cfg, ts["cell_type"], "dev")
    ref = load_type(cfg, ts["match_type"], "dev")
    target = pd.Series(ref.donor).value_counts().values
    subs, sizes, jobs = {}, [], []
    for frac in ts["fractions"]:
        for draw in range(ts["draws"]):
            rng = np.random.default_rng([cfg["seed"], 7000, draw, int(frac * 100)])
            sub = subset(td, subsample(td, target, frac, rng))
            folds = donor_folds(sub.donor, cfg["cv"]["outer_folds"], cfg["seed"] + 7000 + 100 * draw)
            subs[(frac, draw)] = (sub, folds)
            sizes.append({"fraction": frac, "draw": draw, "n_cells": len(sub.donor),
                          "n_donors": len(np.unique(sub.donor)), "median_cells_per_donor":
                          float(pd.Series(sub.donor).value_counts().median()),
                          "reference_cells": int(len(ref.donor)), "reference_donors": len(np.unique(ref.donor))})
            jobs += [(frac, draw, f) for f in range(cfg["cv"]["outer_folds"])]
    res = Parallel(n_jobs=args.jobs or cfg["robustness"]["jobs"], verbose=5)(
        delayed(rc.checkpointed)(ROOT / "data/interim" / STAGE / "jobs" / f"x{fr}_d{d}_f{f}.pkl", job,
                                 subs[(fr, d)][0], cfg, fr, d, f, subs[(fr, d)][1]) for fr, d, f in jobs)
    pred = re_.attach_meta(pd.concat(res, ignore_index=True), cells)
    met = re_.per_repeat_metrics(pred, ["fraction", "draw", "endpoint", "model"], cfg)
    met.insert(0, "cell_type", ts["cell_type"])
    # 每次下采样的供体内 r 的按供体重抽样置信区间
    rows = []
    for (frac, draw, ep, m), s in pred.groupby(["fraction", "draw", "endpoint", "model"]):
        if m not in ("context_model", "ridge"):
            continue
        mm, _ = re_.boot_compare(s.assign(model=m), [], 300, cfg["seed"] + draw, cfg["cv"]["min_group"], {})
        w = mm[mm.level == "within"].iloc[0]
        rows.append({"fraction": frac, "draw": draw, "endpoint": ep, "model": m, "within_lo": w.lo,
                     "within_hi": w.hi})
    ci = pd.DataFrame(rows)
    met = met.merge(ci, on=["fraction", "draw", "endpoint", "model"], how="left")
    full = pd.read_csv(ROOT / "results/03c_summary/metrics_cv.tsv", sep="\t")
    full = full[(full.split == "donors") & full.model.isin(["context_model", "cell_branch", "ridge"])]
    summ = []
    for (frac, ep, m), s in met.groupby(["fraction", "endpoint", "model"]):
        f_a = full[(full.cell_type == ts["cell_type"]) & (full.endpoint == ep) & (full.model == m)]
        f_b = full[(full.cell_type == ts["match_type"]) & (full.endpoint == ep) & (full.model == m)]
        summ.append({"fraction": frac, "endpoint": ep, "model": m, "draws": len(s),
                     "within_mean": s.within_pearson.mean(), "within_q025": s.within_pearson.quantile(0.025),
                     "within_q975": s.within_pearson.quantile(0.975), "within_min": s.within_pearson.min(),
                     "draws_ci_above_0": int((s.within_lo > 0).sum()) if s.within_lo.notna().any() else np.nan,
                     "donor_mean": s.donor_spearman.mean(), "cell_mean": s.cell_spearman.mean(),
                     f"within_full_{ts['cell_type']}": f_a.within_pearson.iloc[0] if len(f_a) else np.nan,
                     f"within_full_{ts['match_type']}": f_b.within_pearson.iloc[0] if len(f_b) else np.nan})
    tabs = {"metrics_by_draw.tsv": met, "summary.tsv": pd.DataFrame(summ), "subsample_sizes.tsv": pd.DataFrame(sizes)}
    outs = []
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/03c_summary/metrics_cv.tsv"],
                     outs, seed=cfg["seed"])
    print(tabs["subsample_sizes.tsv"].to_string())
    print(tabs["summary.tsv"].round(3).to_string())


if __name__ == "__main__":
    main()
