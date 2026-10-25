"""Stage 04c：供体层面准确度的置换零分布（完整交叉验证流程，开发供体）。

零模型（每个细胞类型 × 终点 robustness.perm_null.n_perm 次）：
- donor：供体平均偏移（葡萄糖内去均值后）在供体间随机互换，供体内偏差保持——供体层面不再有表达—功能联系，
  而模型在供体内能学到的信号与实测数据相同；
- full（robustness.perm_null.full_null_types）：在 donor 的基础上再于比较组内打乱，三个层面均无联系。
每次置换按 Stage 03 的完整流程重跑留出供体交叉验证（高变基因、内层调参、梯度提升树、门控），默认惩罚网格与
扩展网格各一次。为复用不依赖终点的 SVD，置换 p 使用 Stage 03 第 (p mod partitions) 个重复的折划分。
零分布均值 = 交叉验证造成的期望偏倚；观测值（Stage 04b 各重复的均值）与零分布比较得到校准后效应与单侧置换 P。
输出 results/04c_donor_null/{null_metrics.tsv, calibration.tsv, calibration_differences.tsv}。
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
from data import load_type  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402

STAGE = "04c_donor_null"
DIFFS = [("context_model", "cell_branch"), ("linear_context_model", "linear_cell_branch"),
         ("context_model", "linear_cell_branch")]


def permuted_y(td, cfg, ct, k, perm, null_type):
    """置换后的终点矩阵（只替换第 k 列）；随机数只取决于 (seed, 细胞类型, 终点, 置换号, 零模型)，各折一致。"""
    tag = [cfg["seed"], cfg["dataset"]["cell_types"].index(ct), k, perm, int(null_type == "full")]
    rng = np.random.default_rng(tag)
    y = td.y.copy()
    y[:, k] = rc.permute_donor_offsets(td.y[:, k], td.donor, td.glucose, rng, within=null_type == "full",
                                       group=td.group)
    return y


def job(td, cfg, ct, part, fold, ep, perms, null_types):
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, part)
    k = td.endpoints.index(ep)
    grids = {"default": None, "extended": cfg["robustness"]["alphas_extended"]}
    out = []
    for perm in perms:
        for nt in null_types:
            ym = permuted_y(td, cfg, ct, k, perm, nt)
            for g, al in grids.items():
                p, _ = rc.run_fold(td, cfg, fold, folds, seed + 37 * fold, [ep], alphas=al, ymat=ym,
                                   tag={"perm": perm, "null": nt, "grid": g})
                out.append(p)
            robust_cache.clear(svd=False)                     # 梯度提升树只在同一置换内复用
    robust_cache.clear()
    o = pd.concat(out, ignore_index=True)
    o[["y", "pred"]] = o[["y", "pred"]].astype(np.float32)
    return o


def null_metrics(pred, cfg, ct):
    rows = []
    for (nt, g, ep, m, perm), s in pred.groupby(["null", "grid", "endpoint", "model", "perm"]):
        r = re_.per_repeat_metrics(s.assign(k=0), ["k"], cfg).iloc[0]
        rows.append({"cell_type": ct, "null": nt, "grid": g, "endpoint": ep, "model": m, "perm": perm,
                     "cell_spearman": r.cell_spearman, "donor_spearman": r.donor_spearman,
                     "within_pearson": r.within_pearson})
    return pd.DataFrame(rows)


def calibrate_all(nullm: pd.DataFrame, obs: pd.DataFrame) -> tuple:
    """obs：Stage 04b 逐重复指标。standard ↔ 默认网格零分布；extended_grid ↔ 扩展网格零分布。"""
    vmap = {"default": "standard", "extended": "extended_grid"}
    levels = {"donor": ["donor_spearman"], "full": ["cell_spearman", "donor_spearman", "within_pearson"]}
    cal, cald = [], []
    for (ct, nt, g, ep), s in nullm.groupby(["cell_type", "null", "grid", "endpoint"]):
        o = obs[(obs.cell_type == ct) & (obs.variant == vmap[g]) & (obs.endpoint == ep)]
        wide = s.pivot_table(index="perm", columns="model", values=levels[nt])
        for lev in levels[nt]:
            om = o.groupby("model")[lev].mean()
            for m in s.model.unique():
                if m not in om or (lev, m) not in wide:
                    continue
                cal.append({"cell_type": ct, "null": nt, "grid": g, "endpoint": ep, "model": m, "level": lev,
                            "n_obs_repeats": int(o[o.model == m].repeat.nunique())}
                           | re_.calibrate(float(om[m]), wide[(lev, m)].values))
            for a, b in DIFFS:
                if a in om and b in om and (lev, a) in wide and (lev, b) in wide:
                    nd = wide[(lev, a)].values - wide[(lev, b)].values
                    cald.append({"cell_type": ct, "null": nt, "grid": g, "endpoint": ep, "model_a": a,
                                 "model_b": b, "level": lev} | re_.calibrate(float(om[a] - om[b]), nd))
    return pd.DataFrame(cal), pd.DataFrame(cald)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None, help="并行进程数（缺省取 robustness.jobs）")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    nc = cfg["robustness"]["perm_null"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    obs_path = ROOT / "results/04b_robustness/metrics_by_repeat.tsv"
    obs = pd.read_csv(obs_path, sep="\t")
    allnull, outputs = [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        nts = ["donor"] + (["full"] if ct in nc["full_null_types"] else [])
        jobs = [(p, f, ep, list(range(p, nc["n_perm"], nc["partitions"])))
                for p in range(nc["partitions"]) for f in range(cfg["cv"]["outer_folds"]) for ep in td.endpoints]
        res = Parallel(n_jobs=args.jobs or cfg["robustness"]["jobs"], verbose=5)(
            delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_p{p}_f{f}_{ep}.pkl", job, td, cfg, ct, p, f, ep, pl, nts)
            for p, f, ep, pl in jobs)
        pred = re_.attach_meta(pd.concat(res, ignore_index=True), cells)
        pp = tmp / f"null_pred_{ct}.parquet"
        pred.to_parquet(pp)
        outputs.append(pp)
        groups = [(k, s) for k, s in pred.groupby(["null", "grid", "endpoint"])]
        parts = Parallel(n_jobs=cfg["robustness"]["jobs"])(delayed(null_metrics)(s, cfg, ct) for _, s in groups)
        allnull.append(pd.concat(parts, ignore_index=True))
    nullm = pd.concat(allnull, ignore_index=True)
    cal, cald = calibrate_all(nullm, obs)
    tabs = {"null_metrics.tsv": nullm, "calibration.tsv": cal, "calibration_differences.tsv": cald}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outputs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", obs_path], outputs, seed=cfg["seed"])
    show = cal[(cal.level == "donor_spearman") & (cal.null == "donor") & (cal.grid == "default")]
    print(show.pivot_table(index=["cell_type", "endpoint"], columns="model",
                           values=["observed", "null_mean", "p_perm"]).round(3).to_string())


if __name__ == "__main__":
    main()
