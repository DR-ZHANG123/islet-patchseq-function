"""Stage 06b：外部队列（HPAP 供体）内部可达到的预测精度。

在 HPAP 供体内部做留出供体交叉验证（folds 折 × repeats 次，训练与测试都只用 HPAP 供体），
得到该队列自身的转录组–功能对应强度，作为 Stage 06 中“开发供体训练 → HPAP 预测”精度的参照。
输出 results/06b_external_ceiling/metrics_internal.tsv。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from data import donor_folds, load_type  # noqa: E402
from features import context_mean, fit_scaler, glucose_onehot  # noqa: E402
from context_model import ContextModel  # noqa: E402
from metrics import three_level  # noqa: E402

STAGE = "06b_external_ceiling"


def run_fold(td, cfg, rep, fold, folds, seed):
    tr_all, te_all = np.where(folds != fold)[0], np.where(folds == fold)[0]
    sc = fit_scaler(td.xlog[tr_all], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(td.xlog)
    mu = context_mean(z, td.donor)
    g1 = glucose_onehot(td.glucose)
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    rows = []
    for k, ep in enumerate(td.endpoints):
        y = td.y[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        m = ContextModel(mcfg, True, cfg["model"]["n_boost_features"], seed + k).fit(z, mu, g1, y, td.donor,
                                                                                    td.group, tr, cells_of)
        p = m.predict(z, mu, g1, td.donor, te, cells_of)
        for name in ("context_model", "ridge"):
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": name,
                                      "repeat": rep, "y": y[te], "pred": p[name]}))
    return pd.concat(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    ec = cfg["external_ceiling"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    met = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "external")
        jobs = []
        for rep in range(ec["repeats"]):
            seed = cfg["seed"] + 1000 * rep
            folds = donor_folds(td.donor, ec["folds"], seed)
            jobs += [(rep, f, folds, seed + 37 * f) for f in range(ec["folds"])]
        res = Parallel(n_jobs=len(jobs))(delayed(run_fold)(td, cfg, r, f, fo, s) for r, f, fo, s in jobs)
        pred = pd.concat(res).join(td.cells.set_index("cell_id")[["donor", "glucose"]], on="cell_id")
        pred["group"] = pred.donor + "|" + pred.glucose.astype(str)
        for (ep, m, rep), s in pred.groupby(["endpoint", "model", "repeat"]):
            met.append({"cell_type": ct, "endpoint": ep, "model": m, "repeat": rep,
                        **three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                      cfg["cv"]["min_within"], s.donor.values, s.glucose.values)})
    met = pd.DataFrame(met)
    path = out / "metrics_internal.tsv"
    met.to_csv(path, sep="\t", index=False)
    print(met.groupby(["cell_type", "endpoint", "model"])[["cell_spearman", "donor_spearman", "within_pearson"]]
          .mean().round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], [path], seed=cfg["seed"])


if __name__ == "__main__":
    main()
