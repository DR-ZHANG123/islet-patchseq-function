"""Stage 06d 的数据准备：读取已存的逐细胞预测，附加供体、葡萄糖与训练集均值（不重新训练任何模型）。

- 外部供体：Stage 06 预测（全部开发供体训练），训练均值 = 全部开发细胞的均值（及同葡萄糖均值）。
- 开发对照：Stage 03 留出供体交叉验证预测，训练均值按每个重复 × 折的训练细胞（同重复中其他折）计算。
返回的“单元”是若干个重复的数组字典列表（外部只有一个重复）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import transform_endpoint
from repro import ROOT


def endpoint_scales(cfg) -> dict:
    """终点短名 → (变换, 原始单位)。"""
    out = {}
    for col, (ep, how) in cfg["dataset"]["endpoint_panel"].items():
        tok = col.removesuffix("_cell").split("_")
        unit = f"{tok[-2]}/{tok[-1]}" if tok[-1] == "pF" and tok[-2] in ("fF", "pC", "pA") else tok[-1]
        out[ep] = (how, unit)
    return out


def load_cells(cfg) -> pd.DataFrame:
    c = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    c = c[c.qc_pass].copy()
    c["dev"] = c.site != cfg["cv"]["external_site"]
    return c


def dev_training_means(cfg, cells: pd.DataFrame, ct: str) -> pd.DataFrame:
    """外部预测的常数基线：开发细胞的终点均值（分析尺度），总体（glucose = 'all'）与按葡萄糖。"""
    d = cells[(cells.cell_type == ct) & cells.dev]
    rows = []
    for col, (ep, how) in cfg["dataset"]["endpoint_panel"].items():
        y = pd.Series(transform_endpoint(d[ep].values, how), index=d.index)
        ok = y.notna()
        rows.append({"endpoint": ep, "glucose": "all", "mean": y[ok].mean()})
        for gl, s in y[ok].groupby(d.glucose[ok]):
            rows.append({"endpoint": ep, "glucose": gl, "mean": s.mean()})
    return pd.DataFrame(rows)


def _arrays(s: pd.DataFrame, donor_code: dict) -> dict:
    g = pd.factorize(s.donor.astype(str) + "|" + s.glucose.astype(str))[0]
    return {"y": s.y.values.astype(float), "p": s.pred.values.astype(float), "base": s.base.values.astype(float),
            "base_g": s.base_g.values.astype(float), "donor": s.donor.map(donor_code).values.astype(int),
            "g": g.astype(int), "glucose": s.glucose.values}


def external_units(cfg, cells, ct, models) -> dict:
    """{(endpoint, model): [arrays]}，外部 HPAP 细胞。"""
    pred = pd.read_parquet(ROOT / "data/interim/06_external_hpap" / f"pred_{ct}.parquet")
    pred = pred[pred.model.isin(models)].join(cells.set_index("cell_id")[["donor", "glucose"]], on="cell_id")
    tm = dev_training_means(cfg, cells, ct)
    allm = tm[tm.glucose == "all"].set_index("endpoint")["mean"]
    glm = tm[tm.glucose != "all"].assign(glucose=lambda t: t.glucose.astype(int)).set_index(["endpoint", "glucose"])
    pred["base"] = pred.endpoint.map(allm)
    pred["base_g"] = glm.loc[list(zip(pred.endpoint, pred.glucose)), "mean"].values
    code = {d: i for i, d in enumerate(sorted(pred.donor.unique()))}
    return {(ep, m): [_arrays(s, code)] for (ep, m), s in pred.groupby(["endpoint", "model"])}


def _fold_means(ref: pd.DataFrame, keys: list) -> pd.DataFrame:
    """每个 (重复, 折[, 葡萄糖]) 的训练均值：同一重复内其他折细胞的均值。"""
    tot = ref.groupby(["repeat"] + keys[1:]).y.agg(["sum", "count"])
    fold = ref.groupby(["repeat", "fold"] + keys[1:]).y.agg(["sum", "count"])
    t = tot.reindex(fold.index.droplevel("fold")).values
    fold["mean"] = (t[:, 0] - fold["sum"].values) / (t[:, 1] - fold["count"].values)
    return fold[["mean"]].reset_index()


def development_units(cfg, cells, ct, models) -> dict:
    """{(endpoint, model): [每个重复的 arrays]}，Stage 03 留出供体交叉验证。"""
    pred = pd.read_parquet(ROOT / "data/interim/03_benchmark" / f"pred_{ct}_donors.parquet")
    pred = pred[pred.model.isin(models)].join(cells.set_index("cell_id")[["donor", "glucose"]], on="cell_id")
    code = {d: i for i, d in enumerate(sorted(pred.donor.unique()))}
    units = {}
    for ep, s_ep in pred.groupby("endpoint"):
        ref = s_ep[s_ep.model == models[0]]
        m_all = _fold_means(ref, ["fold"]).rename(columns={"mean": "base"})
        m_gl = _fold_means(ref, ["fold", "glucose"]).rename(columns={"mean": "base_g"})
        s_ep = s_ep.merge(m_all, on=["repeat", "fold"]).merge(m_gl, on=["repeat", "fold", "glucose"])
        for m, s in s_ep.groupby("model"):
            units[(ep, m)] = [_arrays(r, code) for _, r in s.groupby("repeat")]
    return units
