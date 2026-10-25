"""Stage 03e 汇总：新增方法与 Stage 03/03b 方法在同一留出供体划分上的三个层面指标与配对差值。

- 逐重复指标：与 Stage 03 的 metrics_<type>_donors.tsv 列格式相同（three_level，按比较组）。
- 点估计与置信区间：与 Stage 03c 完全相同——每个细胞 5 次重复的留出预测取平均；按供体有放回重抽样
  （summary.n_boot 次，每个终点用同一随机种子，因此重抽样供体集合与 Stage 03c 相同）；
  所用函数 prepare / stats_from 直接取自 03c_summarize.py。
- 配对差值：同一组重抽样供体上 “方法 − 参照”（正值 = 方法更好），参照为 config sota.references；双侧重抽样 P 值。
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from metrics import three_level
from repro import ROOT, write_provenance

STAGE = "03e_sota"
INT = ROOT / "data" / "interim" / STAGE
OUT = ROOT / "results" / STAGE


def _load_03c():
    spec = importlib.util.spec_from_file_location("s03c", ROOT / "scripts/stages/03c_summarize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_pred_files(ct: str) -> list:
    return sorted(INT.glob(f"pred_{ct}_*.parquet"))


def reference_preds(ct: str) -> pd.DataFrame:
    parts = [ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet",
             ROOT / "data/interim/03b_benchmark_nets" / f"pred_{ct}_donors.parquet"]
    return pd.concat([pd.read_parquet(f) for f in parts if f.exists()], ignore_index=True)


def boot_endpoint(ct, ep, s, refs, n_boot, seed, min_group, label):
    """s：某终点全部模型的逐细胞预测（已平均或单次），列 cell_id/model/donor/glucose/y/pred。"""
    m03c = _load_03c()
    rng = np.random.default_rng(seed)
    avg = s.groupby(["cell_id", "model", "donor", "glucose"], as_index=False).agg(y=("y", "mean"), pred=("pred", "mean"))
    models = sorted(avg.model.unique())
    preps = {m: m03c.prepare(avg[avg.model == m], min_group) for m in models}
    donors = np.array(sorted(avg.donor.unique()))
    boots = {m: [] for m in models}
    for _ in range(n_boot):
        pick = rng.choice(donors, len(donors), replace=True)
        for m in models:
            boots[m].append(m03c.stats_from(preps[m], pick))
    rows, diffs = [], []
    point = {m: m03c.stats_from(preps[m], donors) for m in models}
    for m in models:
        b = np.array(boots[m])
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        rows.append({"cell_type": ct, "set": label, "endpoint": ep, "model": m,
                     "cell_spearman": point[m][0], "cell_lo": lo[0], "cell_hi": hi[0],
                     "donor_spearman": point[m][1], "donor_lo": lo[1], "donor_hi": hi[1],
                     "within_pearson": point[m][2], "within_lo": lo[2], "within_hi": hi[2],
                     "n_cells": int((avg.model == m).sum()), "n_donors": len(donors)})
    for ref in refs:
        if ref not in models:
            continue
        rb = np.array(boots[ref])
        for m in models:
            if m == ref:
                continue
            d = np.array(boots[m]) - rb
            pt = point[m] - point[ref]
            lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
            for j, lev in enumerate(("cell", "donor", "within")):
                col = d[:, j][~np.isnan(d[:, j])]
                pv = min(1.0, 2 * min((col <= 0).mean(), (col >= 0).mean())) if len(col) else np.nan
                diffs.append({"cell_type": ct, "set": label, "endpoint": ep, "reference": ref, "method": m,
                              "level": lev, "difference": pt[j], "lo": lo[j], "hi": hi[j], "p_boot": pv})
    return rows, diffs


def per_repeat(pred: pd.DataFrame, cells: pd.DataFrame, cfg: dict, ct: str) -> pd.DataFrame:
    """与 Stage 03 summarize() 相同：按 (终点, 模型, 重复) 计算 three_level（比较组口径）。"""
    p = pred.join(cells[["donor", "glucose"]], on="cell_id")
    p["group"] = p.donor + "|" + p.glucose.astype(str)
    out = []
    for (ep, m, rep), s in p.groupby(["endpoint", "model", "repeat"]):
        out.append({"cell_type": ct, "endpoint": ep, "model": m, "repeat": rep,
                    **three_level(s.y.values, s.pred.values, s.group.values, cfg["cv"]["min_group"],
                                  cfg["cv"]["min_within"])})
    return pd.DataFrame(out)


def run(cfg: dict) -> None:
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    refs = cfg["sota"]["references"]
    outputs, inputs, jobs = [], [ROOT / "results/01_dataset/cells.tsv"], []
    for ct in cfg["dataset"]["cell_types"]:
        files = new_pred_files(ct)
        new = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        inputs += files
        # 完整性：每个新方法 × 终点在 5 个重复中都覆盖与 Stage 03 相同的细胞
        ref = reference_preds(ct)
        n_ref = ref[ref.model == refs[0]].groupby("endpoint").size()
        n_new = new.groupby(["model", "endpoint"]).size().unstack("model")
        bad = n_new.columns[(n_new.reindex(n_ref.index).ne(n_ref, axis=0)).any()]
        if len(bad):
            raise RuntimeError(f"{ct}: 以下方法的留出预测与 Stage 03 覆盖不一致：{list(bad)}")
        met = per_repeat(new, cells, cfg, ct)
        f = OUT / f"metrics_{ct}_donors.tsv"
        met.to_csv(f, sep="\t", index=False)
        outputs.append(f)
        allp = pd.concat([ref, new], ignore_index=True).join(cells[["donor", "glucose"]], on="cell_id")
        jobs += [(ct, ep, s) for ep, s in allp.groupby("endpoint")]
    res = Parallel(n_jobs=min(cfg["sota"]["jobs"], len(jobs)), verbose=2)(
        delayed(boot_endpoint)(ct, ep, s, refs, cfg["summary"]["n_boot"], cfg["seed"], cfg["cv"]["min_group"], "development")
        for ct, ep, s in jobs)
    tables = {"metrics_cv.tsv": pd.DataFrame([r for rr, _ in res for r in rr]),
              "paired_differences.tsv": pd.DataFrame([d for _, dd in res for d in dd])}
    for fn, df in tables.items():
        df.to_csv(OUT / fn, sep="\t", index=False)
        outputs.append(OUT / fn)
    hp = sorted(OUT.glob("hyperparameters_*.tsv"))
    inputs += [ROOT / cfg["foundation"]["out_dir"] / f"{m}.npz" for m in cfg["sota"]["embeddings"]]
    inputs += sorted((INT / "scvi").glob("*_r*_f*.npz"))
    write_provenance(STAGE, inputs, outputs + hp, seed=cfg["seed"],
                     extra={"parts_summarized": sorted({f.stem.split("_", 2)[2] for f in INT.glob("pred_*.parquet")}),
                            "embeddings": [m for m in cfg["sota"]["embeddings"]
                                           if all((INT / f"pred_{c}_emb_{m}.parquet").exists()
                                                  for c in cfg["dataset"]["cell_types"])]})
    print(tables["metrics_cv.tsv"].pivot_table(index=["cell_type", "endpoint"], columns="model",
                                               values="within_pearson").round(3).T.to_string())
