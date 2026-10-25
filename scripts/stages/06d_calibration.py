"""Stage 06d：外部预测的绝对尺度校准与跨平台共同供体比较（只读已存预测，不重新训练）。

1. 外部供体（Stage 06）与开发交叉验证（Stage 03，作参照）：每个细胞类型 × 终点 × 模型的
   Spearman、三种参照的 R²（测试均值 / 训练均值 / 同葡萄糖训练均值）、MAE 与常数基线 MAE（分析尺度与原始尺度）、
   校准斜率与截距、供体内斜率；按供体重抽样 95% 置信区间（external.n_boot 次）。
2. 误差分解：位置偏移（预测均值 − 实测均值，总体与各葡萄糖浓度）与尺度失配（Theil 分解）。
3. 供体层面跨平台比较：只用 Patch-seq 与液滴两种输入都有的供体，配对重抽样给出 Spearman 差值。
输出 results/06d_calibration/{external_calibration, development_calibration, location_shift,
crossplatform_common_donors}.tsv 与 PROVENANCE.json。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from calib_metrics import INVERSE, SCALE_NAME  # noqa: E402
from calib_data import development_units, endpoint_scales, external_units, load_cells  # noqa: E402
from calib_boot import common_donor_comparison, evaluate_unit  # noqa: E402

STAGE = "06d_calibration"


def _job(key, reps, how, levels, n_boot, seed, min_within, min_level):
    met, rows = evaluate_unit(reps, INVERSE[how], levels, n_boot, seed, min_within, min_level)
    return key, met, rows


def run_set(cfg, name, units_of, cells, scales):
    """name：external 或 development；返回 (指标表, 分解表)。"""
    c = cfg["calibration"]
    levels = cfg["dataset"]["glucose_levels"]
    n_boot = cfg["external"]["n_boot"]
    tasks = []
    for ct in cfg["dataset"]["cell_types"]:
        for (ep, m), reps in units_of(cfg, cells, ct, c["models"]).items():
            tasks.append(((ct, ep, m), reps, scales[ep][0]))
    res = Parallel(n_jobs=c["jobs"])(delayed(_job)(key, reps, how, levels, n_boot, cfg["seed"], c["min_within"],
                                                   (c["min_level_cells"], c["min_level_donors"]))
                                     for key, reps, how in tasks)
    met, sh = [], []
    for (ct, ep, m), mt, rows in res:
        how, unit = scales[ep]
        head = {"set": name, "cell_type": ct, "endpoint": ep, "model": m, "scale": SCALE_NAME[how], "unit": unit}
        met.append(head | mt)
        sh += [head | r for r in rows]
    return pd.DataFrame(met), pd.DataFrame(sh)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(cfg)
    scales = endpoint_scales(cfg)
    ext_m, ext_s = run_set(cfg, "external", external_units, cells, scales)
    dev_m, dev_s = run_set(cfg, "development", development_units, cells, scales)
    dl = ROOT / "results/06c_external_donor_level"
    pred = pd.read_csv(dl / "donor_level_predictions.tsv", sep="\t")
    summ = pd.read_csv(dl / "donor_level_summary.tsv", sep="\t")
    xp = common_donor_comparison(pred, summ, cfg["calibration"]["crossplatform_inputs"], cfg["external"]["n_boot"],
                                 cfg["seed"])
    paths = {"external_calibration": ext_m, "development_calibration": dev_m,
             "location_shift": pd.concat([ext_s, dev_s]), "crossplatform_common_donors": xp}
    outputs = []
    for fn, df in paths.items():
        p = out / f"{fn}.tsv"
        df.to_csv(p, sep="\t", index=False, float_format="%.6g")
        outputs.append(p)
    show = ["cell_type", "endpoint", "model", "spearman", "r2_test_mean", "r2_train_mean", "mae_ratio_train_mean",
            "calib_slope", "within_slope"]
    print(ext_m[ext_m.model.isin(["cell_branch", "context_model"])][show].round(3).to_string())
    print(xp[["cell_type", "endpoint", "n_common_donors", "rho_patchseq", "rho_droplet", "diff", "diff_lo",
              "diff_hi"]].round(3).to_string())
    inputs = [ROOT / "results/01_dataset/cells.tsv", dl / "donor_level_predictions.tsv", dl / "donor_level_summary.tsv"]
    inputs += [ROOT / "data/interim/06_external_hpap" / f"pred_{ct}.parquet" for ct in cfg["dataset"]["cell_types"]]
    inputs += [ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet" for ct in cfg["dataset"]["cell_types"]]
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
