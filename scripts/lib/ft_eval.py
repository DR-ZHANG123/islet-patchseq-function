"""Stage 03f 汇总：微调模型与 Stage 03（集成 cell_branch、context_model）及 Stage 03e（同一基础模型的冻结嵌入头）
在同一留出供体划分（相同重复编号）上的三个层面指标与配对差值；HPAP 外部供体的一次性评价。

- 开发供体：只取 config finetune.repeats 中的重复（各方法的逐细胞留出预测按重复编号取出），
  点估计、按供体重抽样置信区间与配对差值直接用 Stage 03e 的 sota_summary.boot_endpoint（即 Stage 03c 的函数，
  同一随机种子 → 同一组重抽样供体）。差值 = 微调模型 − 参照（正值 = 微调更好）。
- 外部供体：指标与置信区间用 Stage 06 的 bootstrap_ci；配对差值同上（参照为 Stage 06 与 Stage 03e 的外部预测）。
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from metrics import three_level
from repro import ROOT


def ft_name(m: str) -> str:
    return f"{m}_finetuned"


def frozen_refs(m: str) -> list:
    return [f"{m}_cell_branch", f"{m}_ridge"]


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/stages" / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def comparator_preds(cfg: dict, ct: str, models: list, repeats: list) -> pd.DataFrame:
    """Stage 03 参照模型与 Stage 03e 冻结嵌入头的逐细胞留出预测（指定重复）。"""
    base = pd.read_parquet(ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet")
    parts = [base[base.model.isin(cfg["finetune"]["references"])]]
    for m in models:
        e = pd.read_parquet(ROOT / "data/interim/03e_sota" / f"pred_{ct}_emb_{m}.parquet")
        parts.append(e[e.model.isin(frozen_refs(m))])
    p = pd.concat(parts, ignore_index=True)
    return p[p["repeat"].isin(repeats)]


def check_coverage(ft: pd.DataFrame, comp: pd.DataFrame, ct: str) -> None:
    """每个微调模型 × 终点 × 重复覆盖的细胞须与 Stage 03 cell_branch 完全相同。"""
    ref = comp[comp.model == "cell_branch"].groupby(["endpoint", "repeat"]).cell_id.apply(frozenset)
    for (m, ep, r), s in ft.groupby(["model", "endpoint", "repeat"]):
        if frozenset(s.cell_id) != ref.loc[(ep, r)]:
            raise RuntimeError(f"{ct} {m} {ep} 重复 {r}：留出细胞与 Stage 03 不一致")


def keep_pairs(diffs: list, models: list, refs: list) -> list:
    """只保留：微调模型 − Stage 03 参照，以及 微调模型 − 同一基础模型的冻结嵌入头。"""
    want = {(ft_name(m), r) for m in models for r in refs + frozen_refs(m)}
    return [d for d in diffs if (d["method"], d["reference"]) in want]


def development(cfg: dict, ft_preds: dict, models: list, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sota_summary
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    reps = cfg["finetune"]["repeats"]
    jobs = []
    for ct, ft in ft_preds.items():
        comp = comparator_preds(cfg, ct, models, reps)
        check_coverage(ft, comp, ct)
        allp = pd.concat([comp, ft], ignore_index=True).join(cells[["donor", "glucose"]], on="cell_id")
        jobs += [(ct, ep, s) for ep, s in allp.groupby("endpoint")]
    refs = cfg["finetune"]["references"] + [r for m in models for r in frozen_refs(m)]
    res = Parallel(n_jobs=min(cfg["sota"]["jobs"], len(jobs)), verbose=2)(
        delayed(sota_summary.boot_endpoint)(ct, ep, s, refs, cfg["summary"]["n_boot"], cfg["seed"],
                                            cfg["cv"]["min_group"], label) for ct, ep, s in jobs)
    met = pd.DataFrame([r for rr, _ in res for r in rr])
    dif = pd.DataFrame(keep_pairs([d for _, dd in res for d in dd], models, cfg["finetune"]["references"]))
    return met, dif


def _metric_job(cfg, ct, ep, m, s):
    s06 = _load("s06", "06_external_hpap.py")
    y, p, grp, don, glu = s.y.values, s.pred.values, s.group.values, s.donor.values, s.glucose.values
    r = three_level(y, p, grp, cfg["cv"]["min_group"], cfg["cv"]["min_within"], don, glu)
    lo, hi = s06.bootstrap_ci(y, p, grp, don, glu, cfg["external"]["n_boot"], cfg["seed"])
    return {"cell_type": ct, "endpoint": ep, "model": m, **r, "cell_lo": lo[0], "cell_hi": hi[0],
            "donor_lo": lo[1], "donor_hi": hi[1], "within_lo": lo[2], "within_hi": hi[2]}


def external(cfg: dict, ft_preds: dict, models: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sota_summary
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    met, diffs = [], []
    refs = cfg["finetune"]["references"] + [r for m in models for r in frozen_refs(m)]
    for ct, ft in ft_preds.items():
        r06 = pd.read_parquet(ROOT / "data/interim/06_external_hpap" / f"pred_{ct}.parquet")
        r3e = pd.read_parquet(ROOT / "data/interim/03e_sota" / f"pred_external_{ct}.parquet")
        comp = pd.concat([r06[r06.model.isin(cfg["finetune"]["references"])], r3e[r3e.model.isin(refs)]])
        allp = pd.concat([comp, ft[["cell_id", "endpoint", "model", "y", "pred"]]], ignore_index=True)
        allp = allp.join(cells[["donor", "glucose"]], on="cell_id")
        allp["group"] = allp.donor + "|" + allp.glucose.astype(str)
        met += Parallel(n_jobs=min(cfg["sota"]["jobs"], 64), verbose=2)(
            delayed(_metric_job)(cfg, ct, ep, m, s) for (ep, m), s in allp.groupby(["endpoint", "model"]))
        res = Parallel(n_jobs=allp.endpoint.nunique())(
            delayed(sota_summary.boot_endpoint)(ct, ep, s.drop(columns="group"), refs, cfg["external"]["n_boot"],
                                                cfg["seed"], cfg["cv"]["min_group"], "HPAP")
            for ep, s in allp.groupby("endpoint"))
        diffs += keep_pairs([d for _, dd in res for d in dd], models, cfg["finetune"]["references"])
    return pd.DataFrame(met), pd.DataFrame(diffs)
