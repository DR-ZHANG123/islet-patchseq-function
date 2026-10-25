"""Stage 09f 用数据准备：表达表示、留出供体 T2D 特征（50 + 50 与 10 + 10 基因）、外部发表特征得分、
细胞层面 Ca2+ 协变量，以及放宽细胞类型置信度过滤的版本。

与 Stage 09e 的 prepare 相同：高变基因与主成分在 ND/糖尿病前期/T2D 全部细胞上无监督计算；
留出供体特征只用折外开发供体推导（Stage 09d）。外部特征（Bandesh 等 2026，EMBO J，Dataset EV7 β 细胞
T2D vs ND，FDR < 0.05）独立于本数据推导，以全部 ND 细胞为参照打分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from data import TypeData, donor_folds, load_type
from features import log_normalize, transform_endpoint
from repro import ROOT
from t2d_conditioned import hvg_matrix, parts, pcs
from t2d_signature import fold_scores, score_cells

MAIN_SITES = ["ADI IsletCore", "HPAP"]
ENDPOINTS = ("exocytosis", "early_exocytosis", "late_exocytosis", "calcium_entry", "sodium_current")


def load_type_relaxed(cfg: dict, cell_type: str) -> TypeData:
    """与 data.load_type 相同，但不要求细胞类型置信度 ≥ min_cell_score（其余质控不变）。"""
    res = ROOT / "results" / "01_dataset"
    cells = pd.read_csv(res / "cells.tsv", sep="\t")
    genes = pd.read_csv(res / "genes.tsv", sep="\t")
    counts = sp.load_npz(ROOT / cfg["paths"]["processed"] / "counts.npz")
    m = (cells.max_dev < 1e-2) & (cells.min_count == 1) & (cells.cell_type == cell_type) & cells.cell_score.notna()
    c = cells[m].reset_index(drop=True)
    keep = genes[[f"{cell_type}_{g}" for g in cfg["dataset"]["glucose_levels"]]].all(1).values
    sub = counts[c.row.values][:, keep].tocsr()
    xlog = log_normalize(sub, cfg["features"]["scale_factor"])
    panel = cfg["dataset"]["endpoint_panel"]
    eps = [v[0] for v in panel.values()]
    y = np.column_stack([transform_endpoint(c[v[0]].values, v[1]) for v in panel.values()])
    group = (c.donor + "|" + c.glucose.astype(str)).values
    return TypeData(c, xlog, genes.symbol.values[keep], y, eps, c.donor.values, group, c.glucose.values, sub)


def external_signature(path, symbols: np.ndarray, fdr: float) -> tuple[np.ndarray, np.ndarray]:
    """外部 β 细胞 T2D 差异表达表 → 本数据中存在的上调、下调基因索引。"""
    d = pd.read_excel(path, sheet_name="Beta", header=1)
    d["symbol"] = d.GeneID.astype(str).str.split(":").str[-1]
    d = d[d.FDR_edgeR < fdr]
    pos = pd.Series(np.arange(len(symbols)), index=symbols)
    pos = pos[~pos.index.duplicated()]
    up = pos.reindex(d[d.logFC_edgeR > 0].symbol).dropna().astype(int).values
    down = pos.reindex(d[d.logFC_edgeR < 0].symbol).dropna().astype(int).values
    return up, down


def prepare(cfg: dict, ct: str, relaxed: bool = False, ext_path=None, min_group: int = 2) -> dict:
    """一种细胞类型的表达表示、得分与各终点的细胞表（列：y、s、s_small、s_ext；Ca2+ 协变量 log1p 与 log 两种尺度）。"""
    c = cfg["t2d_conditioned"]
    sc = cfg["t2d_signature"]
    rc = cfg["t2d_robust"]
    td = load_type_relaxed(cfg, ct) if relaxed else load_type(cfg, ct, "all")
    status = td.cells.status.values
    site = td.cells.site.values
    site_main = np.where(np.isin(site, MAIN_SITES), site, "Other")
    ok = np.isin(status, ["ND", "preT2D", "T2D"])
    rows = np.where(ok)[0]
    z_ok = hvg_matrix(td.xlog[rows], c["n_hvg"])
    pc_ok, var = pcs(z_ok, max(c["n_pcs"]), cfg["seed"])  # 与 Stage 09e 相同：先算 100 个主成分，再取前 K 个
    z = np.zeros((len(status), z_ok.shape[1]))
    z[rows] = z_ok
    pc = np.zeros((len(status), pc_ok.shape[1]))
    pc[rows] = pc_ok
    dev = site != cfg["cv"]["external_site"]
    nf, seed = cfg["cv"]["outer_folds"], cfg["seed"]
    s, sets = fold_scores(td, status, site, dev, sc["n_genes"], sc["min_cpm"], nf, seed, keep_sets=True)
    s_small, _ = fold_scores(td, status, site, dev, rc["small_n_genes"], sc["min_cpm"], nf, seed)
    s_ext = np.full(len(status), np.nan)
    n_ext = (0, 0)
    if ext_path is not None and ct == "Beta":
        up, down = external_signature(ext_path, td.symbols, rc["external_fdr"])
        ref = np.where(ok & (status == "ND"))[0]
        s_ext = score_cells(td.xlog, ref, up, down)
        n_ext = (len(up), len(down))
    k_ca = td.endpoints.index("calcium_entry")
    k_cae = td.endpoints.index("early_ca_current")
    k_size = td.endpoints.index("size")
    raw_ca = td.cells.calcium_entry.values.astype(float)
    raw_cae = td.cells.early_ca_current.values.astype(float)
    ca_log = np.where(raw_ca > 0, np.log(np.where(raw_ca > 0, raw_ca, 1.0)), np.nan)
    cae_log = np.where(raw_cae > 0, np.log(np.where(raw_cae > 0, raw_cae, 1.0)), np.nan)
    ud = np.unique(td.donor[ok])
    folds = dict(zip(ud, donor_folds(ud, c["folds"], seed)))
    frames = {}
    for ep in ENDPOINTS:
        k = td.endpoints.index(ep)
        y = td.y[:, k]
        m = ok & ~np.isnan(y) & ~np.isnan(s)
        d = pd.DataFrame({"y": y, "s": s, "s_small": s_small, "s_ext": s_ext, "ca": td.y[:, k_ca],
                          "ca_early": td.y[:, k_cae], "ca_log": ca_log, "ca_early_log": cae_log, "size": td.y[:, k_size], "g": td.group, "status": status,
                          "donor": td.donor, "site": site_main, "glucose": td.glucose,
                          "cell_row": np.arange(len(y))})[m]
        d = d[d.groupby("g").y.transform("size") >= min_group].reset_index(drop=True)
        d["t2d"] = (d.status == "T2D").astype(float)
        d["pre"] = (d.status == "preT2D").astype(float)
        for part, sub in parts(d, ep, min_group).items():
            frames[(ep, part)] = sub
    return {"td": td, "z": z, "pc": pc, "var": var, "folds": folds, "frames": frames, "sets": sets,
            "status": status, "site": site, "dev": dev, "n_ext": n_ext, "n_cells": int(ok.sum())}


def jaccard_stability(sets: list) -> dict:
    """各折上调/下调基因集两两 Jaccard 的均值与范围。"""
    out = {}
    for lab, j in (("up", 1), ("down", 2)):
        gs = [set(x[j].tolist()) for x in sets]
        v = [len(a & b) / len(a | b) for i, a in enumerate(gs) for b in gs[i + 1:]]
        out[f"jaccard_{lab}_mean"] = float(np.mean(v))
        out[f"jaccard_{lab}_min"] = float(np.min(v))
        out[f"jaccard_{lab}_max"] = float(np.max(v))
    allg = [set(np.r_[x[1], x[2]].tolist()) for x in sets]
    core = set.intersection(*allg)
    out["genes_in_all_folds"] = len(core)
    out["genes_per_fold"] = len(allg[0])
    return out
