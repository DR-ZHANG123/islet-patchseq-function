"""Stage 09f：T2D 胞吐差异不由转录组携带这一结论的边界。

1. 供体间 / 供体内分解：以供体均值主成分、供体内主成分及两者同时为条件的携带比例（β 与 α）。
2. 阳性对照模拟：缺陷由（a）特征的供体均值、（b）供体均值 + 部分供体内特征 + 非转录噪声、
   （c）10 + 10 基因的小特征、（d）细胞层面特征携带时，各估计量能否找回 0/50/100% 的携带比例。
3. Ca2+ 进入：同一组细胞上 Ca2+ 进入的 T2D 比，调整 Ca2+ 进入后的胞吐比，按供体内信度的回归校准。
4. 留出供体特征的中介区间：每次重抽样重新推导特征；各折基因集的一致性。
5. 安慰剂：把供体均值主成分在供体间置换，所得携带比例的分布（检验 α 细胞的携带比例是否来自吸收供体身份）。
6. ND 训练模型在 ND 供体中的负的供体内 r：区间、组内置换零分布与换种子重复。
7. 放宽细胞类型置信度过滤后的总效应与携带比例。
8. 外部发表的 β 细胞 T2D 特征（Bandesh 等 2026，Dataset EV7）：T2D 差（区间与置换 P）与以其为条件的携带比例。
9. 结局在供体 × 葡萄糖组内置换的零分布：保留供体层面结构时的携带比例基线。
供体内主成分（within）的携带比例按构造近似为零（与供体层面的 T2D 指示正交），只作说明，不作证据。
输出 results/09f_t2d_robust/*.tsv。
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
from t2d_coupling import predict_symmetric  # noqa: E402
from t2d_conditioned import perm_df  # noqa: E402
from t2d_robust import (boot_generic, boot_shares, ca_models, ca_reliability, carriers, placebo,  # noqa: E402
                        score_effect, simulate, within_perm_null)
from t2d_robust_data import jaccard_stability, prepare  # noqa: E402
from t2d_robust_sig import boot_rederived, nd_within_diag  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09f_t2d_robust"
EXO_KEYS = [("exocytosis", "all cells"), ("exocytosis", "positive responses"),
            ("early_exocytosis", "positive responses"), ("late_exocytosis", "all cells"),
            ("late_exocytosis", "positive responses")]
SIM_KEYS = [("exocytosis", "positive responses"), ("late_exocytosis", "positive responses"),
            ("exocytosis", "all cells")]
SIG_KEYS = [("exocytosis", "positive responses"), ("late_exocytosis", "positive responses"),
            ("exocytosis", "all cells")]


def tag(rows: list[dict], **kw) -> list[dict]:
    return [kw | r for r in rows]


def run_conditioning(preps: dict, rc: dict, seed: int, relaxed: dict | None = None) -> pd.DataFrame:
    jobs = [(ct, k) for ct in preps for k in EXO_KEYS if k in preps[ct]["frames"]]
    src = relaxed if relaxed is not None else preps
    res = Parallel(n_jobs=min(len(jobs), rc["n_jobs"]))(
        delayed(boot_shares)(src[ct]["frames"][k], src[ct]["pc"], rc["between_pcs"], rc["within_pcs"],
                             rc["n_boot"], np.random.default_rng(seed + i)) for i, (ct, k) in enumerate(jobs))
    rows = []
    for (ct, (ep, part)), r in zip(jobs, res):
        df = src[ct]["frames"][(ep, part)]
        st = df.groupby("donor").status.first()
        rows += tag(r, cell_type=ct, endpoint=ep, part=part, n_cells=len(df), n_nd_donors=int((st == "ND").sum()),
                    n_t2d_donors=int((st == "T2D").sum()))
    return pd.DataFrame(rows)


def run_simulations(prep: dict, rc: dict, seed: int) -> pd.DataFrame:
    jobs = []
    for key in SIM_KEYS:
        df = prep["frames"][key]
        for name, c in carriers(df, rc["sim_rhos"], seed).items():
            jobs.append((key, name, c))
    res = Parallel(n_jobs=min(len(jobs), rc["n_jobs"]))(
        delayed(simulate)(prep["frames"][key], prep["pc"], c, rc["sim_pi"], rc["sim_estimators"], rc["between_pcs"],
                          rc["within_pcs"], rc["n_boot_sim"], np.random.default_rng(seed + 100 + i))
        for i, (key, _, c) in enumerate(jobs))
    rows = []
    for (key, name, _), r in zip(jobs, res):
        rows += tag(r, cell_type="Beta", endpoint=key[0], part=key[1], carrier=name)
    return pd.DataFrame(rows)


def run_ca(prep: dict, rc: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [("exocytosis", "positive responses"), ("late_exocytosis", "positive responses"),
            ("exocytosis", "all cells"), ("early_exocytosis", "positive responses")]
    rows, rel = [], []
    i = 0
    for key in keys:
        for col, col_e, scale in (("ca", "ca_early", "log1p"), ("ca_log", "ca_early_log", "log")):
            df = prep["frames"][key]
            df = df[np.isfinite(df[col]) & np.isfinite(df[col_e])].reset_index(drop=True)
            r = boot_generic(ca_models, df, rc["n_boot"], np.random.default_rng(seed + 200 + i), lams=rc["ca_lambdas"],
                             col=col)
            i += 1
            rows += tag(r, cell_type="Beta", endpoint=key[0], part=key[1], ca_scale=scale, n_cells=len(df),
                        n_donors=df.donor.nunique())
            rel.append({"endpoint": key[0], "part": key[1], "ca_scale": scale} | ca_reliability(df, col, col_e))
    return pd.DataFrame(rows), pd.DataFrame(rel)


def run_rederived(prep: dict, cfg: dict, rc: dict) -> pd.DataFrame:
    sc = cfg["t2d_signature"]
    frames = {k: prep["frames"][k] for k in SIG_KEYS}
    n_chunks = rc["n_jobs"]
    per = int(np.ceil(rc["n_rep_signature"] / n_chunks))
    with parallel_config(backend="loky"):
        res = Parallel(n_jobs=n_chunks)(
            delayed(boot_rederived)(frames, prep["td"], prep["status"], prep["site"], prep["dev"], sc["n_genes"],
                                    sc["min_cpm"], cfg["cv"]["outer_folds"], per, cfg["seed"] + 1000 * (i + 1))
            for i in range(n_chunks))
    reps = pd.DataFrame([r for chunk in res for r in chunk])
    out = []
    for (ep, part), g in reps.groupby(["endpoint", "part"]):
        for q in ("total", "score_t2d", "prop_signature", "prop_signature_between"):
            v = g[q].replace([np.inf, -np.inf], np.nan).dropna().values
            out.append({"endpoint": ep, "part": part, "quantity": q, "median": np.median(v),
                        "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5), "n_rep": len(v)})
    return pd.DataFrame(out)


def run_placebo(preps: dict, rc: dict, seed: int) -> pd.DataFrame:
    jobs = [(ct, k) for ct in preps for k in EXO_KEYS if k in preps[ct]["frames"]]
    res = Parallel(n_jobs=min(len(jobs), rc["n_jobs"]))(
        delayed(placebo)(preps[ct]["frames"][k], preps[ct]["pc"], rc["within_pcs"], rc["n_placebo"],
                         np.random.default_rng(seed + 300 + i)) for i, (ct, k) in enumerate(jobs))
    return pd.DataFrame([{"cell_type": ct, "endpoint": k[0], "part": k[1]} | r for (ct, k), r in zip(jobs, res)])


def run_nd_diag(cfg: dict, rc: dict, td) -> pd.DataFrame:
    pred = pd.read_csv(ROOT / "results/09c_t2d_hidden/predictions.tsv.gz", sep="\t")
    rows = []
    eps = ["exocytosis", "late_exocytosis", "early_exocytosis", "size"]
    for ep in eps:
        p = pred[(pred.cell_type == "Beta") & (pred.endpoint == ep)]
        for st, d in nd_within_diag(p, rc["n_boot_diag"], cfg["seed"]).items():
            rows.append({"endpoint": ep, "seed_offset": 0, "status": st} | d)
    jobs = [(ep, off) for ep in eps[:3] for off in range(1, rc["nd_seeds"] + 1)]
    with parallel_config(backend="loky", inner_max_num_threads=4):
        res = Parallel(n_jobs=len(jobs))(delayed(predict_symmetric)(td, cfg, td.endpoints.index(ep), cfg["seed"] + off)
                                         for ep, off in jobs)
    for (ep, off), p in zip(jobs, res):
        for st, g in p.groupby("status"):
            rows.append({"endpoint": ep, "seed_offset": off, "status": st,
                         "r": float(np.corrcoef(g.yc, g.pc)[0, 1]), "n_cells": len(g), "n_donors": g.donor.nunique()})
    return pd.DataFrame(rows)


def run_within_null(preps: dict, rc: dict, seed: int) -> pd.DataFrame:
    jobs = [(ct, k) for ct in preps for k in EXO_KEYS if k in preps[ct]["frames"]]
    res = Parallel(n_jobs=min(len(jobs), rc["n_jobs"]))(
        delayed(within_perm_null)(preps[ct]["frames"][k], preps[ct]["pc"], rc["null_pcs"], rc["n_within_null"],
                                  np.random.default_rng(seed + 400 + i)) for i, (ct, k) in enumerate(jobs))
    return pd.DataFrame([{"cell_type": ct, "endpoint": k[0], "part": k[1]} | r
                         for (ct, k), rr in zip(jobs, res) for r in rr])


def run_external(prep: dict, cfg: dict, rc: dict, seed: int) -> pd.DataFrame:
    """特征得分的 T2D 差（供体重抽样区间）。外部特征固定，可用中心内标签置换 P；留出供体特征的置换须重新推导，
    见 Stage 09d，这里不给 P。"""
    rng = np.random.default_rng(seed + 500)
    df = prep["frames"][("exocytosis", "all cells")]
    st = df.groupby("donor").status.first()
    n_int = cfg["t2d_signature"]["n_genes"]
    rows = []
    for col, lab, n_up, n_down in (("s_ext", "external", *prep["n_ext"]),
                                   ("s", "internal_held_out_per_fold", n_int, n_int)):
        r = boot_generic(score_effect, df, rc["n_boot"], rng, col=col)
        p, n_perm = np.nan, 0
        if col == "s_ext":
            obs = score_effect(df, col)["score_t2d"]
            null = np.asarray([score_effect(perm_df(df, rng), col)["score_t2d"] for _ in range(rc["n_boot"])])
            p, n_perm = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))), len(null)
        rows += tag(r, signature=lab, p_perm=p, n_perm=n_perm, n_up=n_up, n_down=n_down,
                    n_nd_donors=int((st == "ND").sum()), n_t2d_donors=int((st == "T2D").sum()))
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rc = cfg["t2d_robust"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    t0 = time.time()
    ext = ROOT / rc["external_file"]
    ext = ext if ext.exists() else None
    preps = {ct: prepare(cfg, ct, ext_path=ext) for ct in cfg["dataset"]["cell_types"]}
    print(f"prepared: {time.time() - t0:.0f}s; external genes {preps['Beta']['n_ext']}", flush=True)
    files = []
    force = "--force" in sys.argv

    def step(name: str, fn) -> None:
        """已存在的输出不重算（--force 时重算）；返回 DataFrame 或 {文件名: DataFrame}。"""
        names = [name] if isinstance(name, str) else name
        files.extend(out / n for n in names)
        if not force and all((out / n).exists() for n in names):
            return
        res = fn()
        res = res if isinstance(res, tuple) else (res,)
        for n, df in zip(names, res):
            df.to_csv(out / n, sep="\t", index=False)
        print(f"{names}: {time.time() - t0:.0f}s", flush=True)

    def relaxed_step() -> pd.DataFrame:
        relaxed = {"Beta": prepare(cfg, "Beta", relaxed=True, ext_path=None)}
        rx = run_conditioning(relaxed, rc | {"n_boot": rc["n_boot_relaxed"]}, seed + 7, relaxed)
        rx["n_cells_type_all_status"] = relaxed["Beta"]["n_cells"]
        rx["n_cells_type_strict"] = preps["Beta"]["n_cells"]
        return rx

    step("conditioning.tsv", lambda: run_conditioning(preps, rc, seed))
    step("placebo.tsv", lambda: run_placebo(preps, rc, seed))
    step("within_permutation_null.tsv", lambda: run_within_null(preps, rc, seed))
    step(["calcium_adjustment.tsv", "calcium_reliability.tsv"], lambda: run_ca(preps["Beta"], rc, seed))
    step("simulations.tsv", lambda: run_simulations(preps["Beta"], rc, seed))
    step("signature_stability.tsv",
         lambda: pd.DataFrame([{"cell_type": "Beta"} | jaccard_stability(preps["Beta"]["sets"])]))
    step("signature_rederived.tsv", lambda: run_rederived(preps["Beta"], cfg, rc))
    step("external_signature.tsv", lambda: run_external(preps["Beta"], cfg, rc, seed))
    step("relaxed_filter.tsv", relaxed_step)
    step("nd_within_diagnosis.tsv", lambda: run_nd_diag(cfg, rc, preps["Beta"]["td"]))
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/09c_t2d_hidden/predictions.tsv.gz"]
                     + ([ext] if ext else []), files, seed=seed)
    print(f"done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
