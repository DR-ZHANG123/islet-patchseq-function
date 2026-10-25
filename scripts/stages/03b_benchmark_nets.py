"""Stage 03b：神经网络对照的留出供体重复交叉验证（与 Stage 03 使用相同的供体分折与高变基因规则）。

变体：mlp_cell（单细胞输入）、mlp_concat（单细胞编码与供体参照编码拼接）、mlp_two（非线性双分支）、
mlp_two_within（非线性双分支 + 供体内差分损失）。每折 3 个随机种子取平均。
"""
from __future__ import annotations

import argparse
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
from features import fit_scaler, glucose_onehot  # noqa: E402
from metrics import three_level  # noqa: E402

STAGE = "03b_benchmark_nets"
VARIANTS = {"mlp_cell": ("cell", 0.0), "mlp_concat": ("concat", 0.0), "mlp_two": ("two", 0.0),
            "mlp_two_within": ("two", 1.0)}


def run_fold(td, cfg, rep, fold, folds, seed, device):
    import nets
    tr = np.where(folds != fold)[0]
    te = np.where(folds == fold)[0]
    sc = fit_scaler(td.xlog[tr], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(td.xlog).astype(np.float32)
    g1 = glucose_onehot(td.glucose).astype(np.float32)
    rows = []
    for name, (mode, lam) in VARIANTS.items():
        p = np.mean([nets.train_predict(z, g1, td.y, td.donor, td.group, tr, te, mode, lam, seed + s, device)
                     for s in range(cfg["nets"]["seeds"])], 0)
        for k, ep in enumerate(td.endpoints):
            ok = ~np.isnan(td.y[te, k])
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te][ok], "endpoint": ep, "model": name,
                                      "repeat": rep, "fold": fold, "y": td.y[te, k][ok], "pred": p[ok, k]}))
    return pd.concat(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out_dir = ROOT / "results" / STAGE
    int_dir = ROOT / "data" / "interim" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ct in cfg["dataset"]["cell_types"]:
        td = load_type(cfg, ct, "dev")
        jobs = []
        for rep in range(cfg["cv"]["repeats"]):
            seed = cfg["seed"] + 1000 * rep
            folds = donor_folds(td.donor, cfg["cv"]["outer_folds"], seed)
            jobs += [(rep, f, folds, seed + 37 * f) for f in range(cfg["cv"]["outer_folds"])]
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_fold)(td, cfg, r, f, fo, s, args.device) for r, f, fo, s in jobs)
        pred = pd.concat(res, ignore_index=True)
        pred_path = int_dir / f"pred_{ct}_donors.parquet"
        pred.to_parquet(pred_path)
        meta = td.cells.set_index("cell_id")
        pred = pred.join(meta[["donor", "glucose"]], on="cell_id")
        pred["group"] = pred.donor + "|" + pred.glucose.astype(str)
        met = []
        for (ep, m, rep), s in pred.groupby(["endpoint", "model", "repeat"]):
            met.append({"cell_type": ct, "endpoint": ep, "model": m, "repeat": rep,
                        **three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                      cfg["cv"]["min_within"])})
        met = pd.DataFrame(met)
        met_path = out_dir / f"metrics_{ct}_donors.tsv"
        met.to_csv(met_path, sep="\t", index=False)
        outputs += [pred_path, met_path]
        print(ct, met.groupby(["endpoint", "model"]).within_pearson.mean().unstack().round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv"], outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
