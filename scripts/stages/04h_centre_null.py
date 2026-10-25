"""Stage 04h：按来源中心 / GEO series 分层的供体层面置换零分布（开发供体，完整留出供体交叉验证）。

与 Stage 04c 的 donor 零模型相同（供体平均偏移在供体间互换、供体内偏差保持），但互换受分层限制
（config centre_null.variants）：
- site：只在同一来源中心内的供体之间互换——中心间的平均差异保留在零分布中；
- site_geo：同一中心且同一 GEO series 内互换（更严格；单供体分层的供体保持不动）；
- residual：供体偏移对 中心 + series（加性哑变量）回归，拟合值保留、残差在全部供体间互换。
因此零分布包含模型仅凭中心或 series 差异就能得到的供体层面相关，观测值超出它才说明模型识别了中心内的供体差异。
每次置换用 Stage 03 第 (p mod partitions) 个重复的折划分重跑全流程（训练细胞内选高变基因、内层调参、岭回归 +
梯度提升树），只拟合集成（cell_branch）与岭回归（与完整模型逐位一致，见 robust_light）。
观测值 = Stage 04b standard 各重复供体层面 Spearman 的均值（与 Stage 04c 相同），P = (1 + #{零 ≥ 观测}) / (B + 1)；
另给出每个 零模型 × 模型 的 18 个 细胞类型 × 终点 组合内的 BH q 值，并并列 Stage 04c 的不分层结果。
输出 results/04h_centre_null/{null_metrics.tsv, calibration.tsv, strata.tsv}。
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
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import load_type  # noqa: E402
import robust_cache  # noqa: E402
import robust_cv as rc  # noqa: E402
import robust_eval as re_  # noqa: E402
import robust_light as rl  # noqa: E402

STAGE = "04h_centre_null"


def strata_table(td) -> pd.DataFrame:
    return td.cells.groupby("donor")[["site", "geo"]].first()


def permuted_y(td, cfg, ct, k, perm, vname, strata):
    """随机数只取决于 (seed, 细胞类型, 终点, 置换号, 零模型)，各折一致。"""
    cn = cfg["centre_null"]
    mode, cols = cn["variants"][vname]
    vi = list(cn["variants"]).index(vname)
    rng = np.random.default_rng([cfg["seed"], 9200, cfg["dataset"]["cell_types"].index(ct), k, perm, vi])
    y = td.y.copy()
    y[:, k] = rl.permute_offsets_strata(td.y[:, k], td.donor, td.glucose, rng, strata[cols], mode)
    return y


def job(td, cfg, ct, part, fold, ep, plan):
    """一个 (划分, 外层折, 终点)：依次运行各零模型的全部置换；特征与岭回归 SVD 只算一次（与终点、置换无关）。
    plan：[(零模型, 置换号列表)]。"""
    robust_cache.install()
    folds, seed = rc.repeat_folds(td, cfg, part)
    inp = rl.fold_inputs(td, cfg, fold, folds)
    k = td.endpoints.index(ep)
    strata = strata_table(td)
    out = []
    for vname, perms in plan:
        for perm in perms:
            ym = permuted_y(td, cfg, ct, k, perm, vname, strata)
            p = rl.run_fold_light(td, cfg, fold, seed + 37 * fold, [ep], inp, ymat=ym,
                                  tag={"perm": perm, "null": vname})
            out.append(p[p.model.isin(cfg["centre_null"]["models"])])
            robust_cache.clear(svd=False)
    robust_cache.clear()
    o = pd.concat(out, ignore_index=True)
    o[["y", "pred"]] = o[["y", "pred"]].astype(np.float32)
    return o


def job_list(cfg, ct, endpoints):
    """每个 (终点, 划分 p, 折) 一个作业；置换 i 属于划分 i mod partitions。"""
    cn = cfg["centre_null"]
    jobs = []
    for ep in endpoints:
        n = cn["n_perm_priority"] if ep in cn["priority"].get(ct, []) else cn["n_perm"]
        for p in range(cn["partitions"]):
            plan = [(v, list(range(p, n, cn["partitions"]))) for v in cn["variants"]]
            jobs += [(ep, p, f, plan) for f in range(cfg["cv"]["outer_folds"])]
    return jobs


def null_metrics(pred, cfg, ct):
    rows = []
    for (nt, ep, m, perm), s in pred.groupby(["null", "endpoint", "model", "perm"]):
        r = re_.per_repeat_metrics(s.assign(k=0), ["k"], cfg).iloc[0]
        rows.append({"cell_type": ct, "null": nt, "endpoint": ep, "model": m, "perm": perm,
                     "donor_spearman": r.donor_spearman})
    return pd.DataFrame(rows)


def calibrate(nullm, obs, c04):
    rows = []
    for (ct, nt, ep, m), s in nullm.groupby(["cell_type", "null", "endpoint", "model"]):
        o = obs[(obs.cell_type == ct) & (obs.variant == "standard") & (obs.endpoint == ep) & (obs.model == m)]
        rows.append({"cell_type": ct, "null": nt, "endpoint": ep, "model": m, "n_obs_repeats": o.repeat.nunique()}
                    | re_.calibrate(float(o.donor_spearman.mean()), s.donor_spearman.values))
    cal = pd.DataFrame(rows)
    cal["q_bh"] = cal.groupby(["null", "model"]).p_perm.transform(lambda p: false_discovery_control(p, method="bh"))
    ref = c04[(c04.null == "donor") & (c04.grid == "default") & (c04.level == "donor_spearman")]
    ref = ref.rename(columns={"null_mean": "null_mean_unrestricted", "calibrated": "calibrated_unrestricted",
                              "p_perm": "p_perm_unrestricted", "n_perm": "n_perm_unrestricted"})
    ref["q_bh_unrestricted"] = ref.groupby("model").p_perm_unrestricted.transform(
        lambda p: false_discovery_control(p, method="bh"))
    keep = ["cell_type", "endpoint", "model", "null_mean_unrestricted", "calibrated_unrestricted",
            "p_perm_unrestricted", "q_bh_unrestricted", "n_perm_unrestricted"]
    return cal.merge(ref[keep], on=["cell_type", "endpoint", "model"], how="left")


def strata_summary(td, cfg, ct):
    st = strata_table(td)
    rows = []
    for vname, (mode, cols) in cfg["centre_null"]["variants"].items():
        key = st[cols].astype(str).agg("|".join, axis=1)
        vc = key.value_counts()
        rows.append({"cell_type": ct, "null": vname, "mode": mode, "strata_vars": "+".join(cols),
                     "n_donors": len(st), "n_strata": len(vc), "largest_stratum": int(vc.max()),
                     "donors_in_singleton_strata": int((vc == 1).sum()),
                     "strata_sizes": ";".join(f"{k}:{v}" for k, v in vc.items())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    cn = cfg["centre_null"]
    out = ROOT / "results" / STAGE
    tmp = ROOT / "data" / "interim" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    obs_path = ROOT / "results/04b_robustness/metrics_by_repeat.tsv"
    c04_path = ROOT / "results/04c_donor_null/calibration.tsv"
    allnull, outputs, strata = [], [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        strata += strata_summary(td, cfg, ct)
        jobs = job_list(cfg, ct, td.endpoints)
        res = Parallel(n_jobs=args.jobs or cn["jobs"], verbose=5)(
            delayed(rc.checkpointed)(tmp / "jobs" / f"{ct}_{ep}_p{p}_f{f}.pkl", job, td, cfg, ct, p, f, ep, plan)
            for ep, p, f, plan in jobs)
        pred = re_.attach_meta(pd.concat(res, ignore_index=True), cells)
        del res
        pp = tmp / f"null_pred_{ct}.parquet"
        pred.to_parquet(pp)
        outputs.append(pp)
        groups = [s for _, s in pred.groupby(["null", "endpoint", "model"])]
        parts = Parallel(n_jobs=args.jobs or cn["jobs"])(delayed(null_metrics)(s, cfg, ct) for s in groups)
        allnull.append(pd.concat(parts, ignore_index=True))
        del pred, groups
    nullm = pd.concat(allnull, ignore_index=True)
    cal = calibrate(nullm, pd.read_csv(obs_path, sep="\t"), pd.read_csv(c04_path, sep="\t"))
    tabs = {"null_metrics.tsv": nullm, "calibration.tsv": cal, "strata.tsv": pd.DataFrame(strata)}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
        outputs.append(out / fn)
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", obs_path, c04_path], outputs, seed=cfg["seed"])
    print(tabs["strata.tsv"].to_string())
    print(cal.pivot_table(index=["cell_type", "endpoint", "model"], columns="null",
                          values="p_perm").join(cal.groupby(["cell_type", "endpoint", "model"]).p_perm_unrestricted.first()
                                                ).round(3).to_string())


if __name__ == "__main__":
    main()
