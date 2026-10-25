"""读取 Stage 01 产出的配对数据集，按细胞类型返回建模所需数组。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from features import log_normalize, transform_endpoint
from repro import ROOT


@dataclass
class TypeData:
    cells: pd.DataFrame          # 细胞表（已过质控）
    xlog: np.ndarray             # log 归一化表达（该类型三个葡萄糖矩阵共同保留的基因）
    symbols: np.ndarray
    y: np.ndarray                # (n, K) 变换后的终点，NaN 为缺测
    endpoints: list
    donor: np.ndarray
    group: np.ndarray            # 比较组：供体 × 葡萄糖
    glucose: np.ndarray
    counts: sp.csr_matrix = None  # 原始整数计数（与 xlog 相同的基因）


def load_type(cfg: dict, cell_type: str, sites: str = "all") -> TypeData:
    res = ROOT / "results" / "01_dataset"
    cells = pd.read_csv(res / "cells.tsv", sep="\t")
    genes = pd.read_csv(res / "genes.tsv", sep="\t")
    counts = sp.load_npz(ROOT / cfg["paths"]["processed"] / "counts.npz")
    c = cells[cells.qc_pass & (cells.cell_type == cell_type)]
    ext = cfg["cv"]["external_site"]
    if sites == "dev":
        c = c[c.site != ext]
    elif sites == "external":
        c = c[c.site == ext]
    c = c.reset_index(drop=True)
    keep = genes[[f"{cell_type}_{g}" for g in cfg["dataset"]["glucose_levels"]]].all(1).values
    sub = counts[c.row.values][:, keep].tocsr()
    xlog = log_normalize(sub, cfg["features"]["scale_factor"])
    panel = cfg["dataset"]["endpoint_panel"]
    eps = [v[0] for v in panel.values()]
    y = np.column_stack([transform_endpoint(c[v[0]].values, v[1]) for v in panel.values()])
    group = (c.donor + "|" + c.glucose.astype(str)).values
    return TypeData(c, xlog, genes.symbol.values[keep], y, eps, c.donor.values, group, c.glucose.values, sub)


def donor_folds(donor: np.ndarray, k: int, seed: int) -> np.ndarray:
    ud = np.unique(donor)
    rng = np.random.default_rng(seed)
    perm = dict(zip(ud, rng.permutation(len(ud))))
    return np.array([perm[d] % k for d in donor])
