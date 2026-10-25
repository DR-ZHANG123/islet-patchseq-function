"""Stage 04b：供体层面结论的稳健性（开发供体，留出供体交叉验证）。

在与 Stage 03 相同的折划分上（重复 r 的种子 = seed + 1000·r）重新拟合双分支模型，比较：
- standard：原设定，共 robustness.repeats[细胞类型] 次重复（前 5 次与 Stage 03 完全一致，用于核对可复现性），
  稳定供体层面估计以及完整模型与其细胞分支的配对差值；
- extended_grid：岭回归惩罚网格向上扩展（robustness.alphas_extended），检验惩罚取到网格上限是否造成供体层面负偏；
- loo_reference：留一供体参照（参照均值与 C̄_d 均不含细胞自身）。
扩展网格与留一参照只在前 robustness.variant_repeats 次重复上运行，与 standard 的同一重复配对比较。
输出 results/04b_robustness/{metrics_by_repeat, metrics_boot, paired_differences, grid_max, stage03_check}.tsv，
逐细胞预测在 data/interim/04b_robustness/。
"""
from __future__ import annotations

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
from data import load_type  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402

STAGE = "04b_robustness"
PAIRS_STD = [("context_model", "cell_branch"), ("linear_context_model", "linear_cell_branch"),
             ("context_model", "linear_cell_branch"), ("context_model", "ridge")]
SHARED = ["context_model", "cell_branch", "ridge", "linear_cell_branch", "linear_context_model"]


def job(td, cfg, rep, fold):
    """一个重复的一个外层折：逐终点运行各变体，同一终点内共享 SVD 与梯度提升树。"""
    robust_cache.install()
    rb = cfg["robustness"]
    folds, seed = rc.repeat_folds(td, cfg, rep)
    fseed = seed + 37 * fold
    preds, pars = [], []
    for ep in td.endpoints:
        variants = {"standard": {}}
        if rep < rb["variant_repeats"]:
            variants |= {"extended_grid": {"alphas": rb["alphas_extended"]}, "loo_reference": {"loo": True}}
        for v, kw in variants.items():
            p, q = rc.run_fold(td, cfg, fold, folds, fseed, [ep], tag={"variant": v, "repeat": rep}, **kw)
            preds.append(p)
            pars.append(q)
        robust_cache.clear()
    return pd.concat(preds, ignore_index=True), pd.concat(pars, ignore_index=True)


def stage03_check(pred, ct):
    """standard 前 5 次重复与 Stage 03 逐细胞预测的最大绝对差。"""
    f = ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet"
    if not f.exists():
        return pd.DataFrame()
    old = pd.read_parquet(f)
    old = old[old.model.isin(SHARED)]
    new = pred[(pred.variant == "standard") & pred.model.isin(SHARED)]
    m = new.merge(old, on=["cell_id", "endpoint", "model", "repeat"], suffixes=("", "_s03"))
    return (m.assign(absdiff=(m.pred - m.pred_s03).abs()).groupby("model").absdiff.agg(["max", "size"])
            .reset_index().assign(cell_type=ct))


def grid_max(par: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    top = {"standard": max(cfg["model"]["alphas"]), "extended_grid": max(cfg["robustness"]["alphas_extended"])}
    for (v, ep), s in par[par.variant.isin(top)].groupby(["variant", "endpoint"]):
        s = s[s.repeat < cfg["robustness"]["variant_repeats"]]
        rows.append({"variant": v, "endpoint": ep, "n_fits": len(s),
                     "ridge_at_max": float((s.alpha_ridge >= top[v]).mean()),
                     "cell_at_max": float((s.alpha_cell >= top[v]).mean()),
                     "ridge_at_or_above_1e5": float((s.alpha_ridge >= 1e5).mean()),
                     "ridge_median_alpha": float(s.alpha_ridge.median())})
    return pd.DataFrame(rows)


def summaries(pred, cfg, ct):
    n_boot, mg = cfg["summary"]["n_boot"], cfg["cv"]["min_group"]
    vr = cfg["robustness"]["variant_repeats"]
    met, dif = [], []
    for ep, s in pred.groupby("endpoint"):
        std = re_.average_repeats(s[s.variant == "standard"], ["model"])
        m, d = re_.boot_compare(std, PAIRS_STD, n_boot, cfg["seed"], mg,
                                {"cell_type": ct, "endpoint": ep, "comparison": "standard_all_repeats"})
        met.append(m); dif.append(d)
        sub = s[s.repeat < vr].copy()
        sub["model"] = sub.model + "@" + sub.variant
        avg = re_.average_repeats(sub, ["model"])
        pairs = [(f"{m_}@extended_grid", f"{m_}@standard") for m_ in SHARED] + \
                [(f"{m_}@loo_reference", f"{m_}@standard") for m_ in SHARED] + \
                [("context_model@loo_reference", "cell_branch@standard"),
                 ("linear_context_model@loo_reference", "linear_cell_branch@standard")]
        m, d = re_.boot_compare(avg, pairs, n_boot, cfg["seed"], mg,
                                {"cell_type": ct, "endpoint": ep, "comparison": f"variants_first_{vr}_repeats"})
        met.append(m); dif.append(d)
    return pd.concat(met), pd.concat(dif)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rb = cfg["robustness"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    allrep, allmet, alldif, allgm, allchk, outputs = [], [], [], [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        jobs = [(r, f) for r in range(rb["repeats"][ct]) for f in range(cfg["cv"]["outer_folds"])]
        jobs.sort(key=lambda t: t[0] >= rb["variant_repeats"])        # 计算量大的变体重复先跑
        res = Parallel(n_jobs=rb["jobs"], verbose=5)(
            delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_r{r}_f{f}.pkl", job, td, cfg, r, f) for r, f in jobs)
        pred = re_.attach_meta(pd.concat([r[0] for r in res], ignore_index=True), cells)
        par = pd.concat([r[1] for r in res], ignore_index=True)
        pp = tmp / f"pred_{ct}.parquet"
        pred.to_parquet(pp)
        outputs.append(pp)
        rep = re_.per_repeat_metrics(pred, ["variant", "endpoint", "model", "repeat"], cfg)
        rep.insert(0, "cell_type", ct)
        allrep.append(rep)
        m, d = summaries(pred, cfg, ct)
        allmet.append(m); alldif.append(d)
        allgm.append(grid_max(par, cfg).assign(cell_type=ct))
        allchk.append(stage03_check(pred, ct))
        par.insert(0, "cell_type", ct)
        par.to_csv(out / f"hyperparameters_{ct}.tsv", sep="\t", index=False)
        outputs.append(out / f"hyperparameters_{ct}.tsv")
    tabs = {"metrics_by_repeat.tsv": pd.concat(allrep), "metrics_boot.tsv": pd.concat(allmet),
            "paired_differences.tsv": pd.concat(alldif), "grid_max.tsv": pd.concat(allgm),
            "stage03_check.tsv": pd.concat(allchk)}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outputs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"])
    d = tabs["paired_differences.tsv"]
    print(d[(d.level == "donor")].round(3).to_string())
    print(tabs["stage03_check.tsv"].to_string())


if __name__ == "__main__":
    main()
