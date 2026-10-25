"""Stage 09c：T2D 中的功能差异是否能从转录组预测值中看到。

预测：与 Stage 09 对应关系分析相同的对称留出供体设计（每一折只用折外 ND 供体训练岭–提升集成模型，
T2D 与糖尿病前期供体从不进入训练），对两种细胞类型的全部终点给出预测值。
比较：实测值、预测值、残差的 T2D − ND 差（供体等权线性模型，协变量同 Stage 09 主模型），
预测值捕获比例 β_p / β_y，以及各状态内的供体内与供体层面预测精度。
输出 results/09c_t2d_hidden/{predictions.tsv.gz, hidden_effects.tsv}。
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
from t2d_coupling import predict_symmetric  # noqa: E402
from t2d_hidden import infer, nd_sd, parts  # noqa: E402

warnings.filterwarnings("ignore")
STAGE = "09c_t2d_hidden"


def predict_all(cfg: dict, hc: dict, out: Path) -> pd.DataFrame:
    fn = out / "predictions.tsv.gz"
    if fn.exists():
        return pd.read_csv(fn, sep="\t")
    jobs, meta = [], []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "all")
        for k, ep in enumerate(td.endpoints):
            jobs.append((td, k))
            meta.append((ct, td.glucose))
    with parallel_config(backend="loky", inner_max_num_threads=hc["threads"]):
        preds = Parallel(n_jobs=hc["n_jobs"] // hc["threads"])(
            delayed(predict_symmetric)(td, cfg, k, cfg["seed"]) for td, k in jobs)
    frames = [p.assign(cell_type=ct, glucose=glc[p.index.values]) for p, (ct, glc) in zip(preds, meta)]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(fn, sep="\t", index=False)
    return df


def run_one(sub: pd.DataFrame, hc: dict, seed: int) -> pd.DataFrame:
    rows = []
    for part, d in parts(sub, sub.endpoint.iloc[0]).items():
        d = d[d.status.isin(["ND", "preT2D", "T2D"])].copy()
        d["r"] = d.y - d.p
        d["t2d"] = (d.status == "T2D").astype(float)
        d["pre"] = (d.status == "preT2D").astype(float)
        sd = nd_sd(d)
        res = infer(d, hc["n_boot"], hc["n_perm"], seed)
        st = d.groupby("donor").status.first()
        rows.append(res.assign(part=part, nd_sd=sd, n_cells=len(d), n_nd_donors=int((st == "ND").sum()),
                               n_t2d_donors=int((st == "T2D").sum())))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    hc = cfg["t2d_hidden"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pred = predict_all(cfg, hc, out)
    print(f"predictions: {time.time() - t0:.0f}s", flush=True)
    keys = list(pred.groupby(["cell_type", "endpoint"], sort=False).groups)
    res = Parallel(n_jobs=min(len(keys), hc["n_jobs"]))(
        delayed(run_one)(pred[(pred.cell_type == ct) & (pred.endpoint == ep)].reset_index(drop=True), hc,
                         cfg["seed"] + i) for i, (ct, ep) in enumerate(keys))
    tab = pd.concat([r.assign(cell_type=ct, endpoint=ep) for r, (ct, ep) in zip(res, keys)], ignore_index=True)
    is_beta = tab.quantity.str.startswith("beta_")
    for c in ("estimate", "lo", "hi"):
        tab.loc[is_beta, c + "_sd"] = tab.loc[is_beta, c] / tab.loc[is_beta, "nd_sd"]
    tab = tab[["cell_type", "endpoint", "part", "quantity", "estimate", "lo", "hi", "p_perm", "estimate_sd", "lo_sd",
               "hi_sd", "nd_sd", "n_boot", "n_cells", "n_nd_donors", "n_t2d_donors"]]
    tab.to_csv(out / "hidden_effects.tsv", sep="\t", index=False)
    show = tab[tab.quantity.isin(["beta_y", "beta_p", "beta_r", "captured_fraction", "within_r_T2D",
                                  "within_r_ND"])]
    print(show[["cell_type", "endpoint", "part", "quantity", "estimate", "lo", "hi", "p_perm"]].round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv"],
                     [out / "predictions.tsv.gz", out / "hidden_effects.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
