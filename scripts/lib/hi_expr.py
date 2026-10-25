"""HumanIslets 单细胞 HDF5 读取：由 TMM-logCPM 还原整数计数。

HumanIslets 的每个矩阵（细胞类型 × 葡萄糖浓度）由 edgeR::cpm(log=TRUE, prior.count=1) 得到，零计数记为 NA。
对每个细胞，非零值满足 v = log2((y + p) / (L + 2p) * 1e6)，其中 p 为按文库大小缩放的先验计数。
取该细胞最小的两个不同取值（对应 y = 1 与 y = 2）即可解出 (L + 2p) 与 p，从而精确还原所有 y。
还原后逐细胞检查整数性；偏差超过阈值的细胞被标记。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


@dataclass
class Matrix:
    counts: sp.csr_matrix          # cells × genes（整数）
    cells: np.ndarray
    entrez: np.ndarray
    symbols: np.ndarray
    max_dev: np.ndarray            # 每个细胞还原值与最近整数的最大偏差
    min_count: np.ndarray          # 每个细胞还原的最小非零计数（应为 1）


def _decode(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def reconstruct_row(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    """v：一个细胞的 logCPM（NaN = 零计数）。返回还原计数、最大整数偏差、最小非零计数。

    1/D（一个计数对应的线性 CPM 增量）取最小几个取值之间的最小间隔；最小非零计数依次假设为 1、2、3，
    取使全部取值最接近整数的解。
    """
    y = np.zeros_like(v)
    ok = ~np.isnan(v)
    vals = v[ok]
    u = np.unique(np.round(vals, 9))
    if u.size < 2:
        return y, np.inf, np.nan
    lin = 2.0 ** u[:12]
    inv_d = np.diff(lin).min()
    d = 1.0 / inv_d
    best = (np.inf, None, np.nan)
    for y0 in (1.0, 2.0, 3.0):
        p = lin[0] * d - y0
        if p <= 0:
            continue
        rec = 2.0 ** vals * d - p
        dev = float(np.abs(rec - np.round(rec)).max())
        if dev < best[0]:
            best = (dev, rec, y0)
    dev, rec, y0 = best
    y[ok] = np.round(rec)
    return y, dev, float(np.round(rec).min())


def read_matrix(path: Path) -> Matrix:
    with h5py.File(path, "r") as h:
        x = h["data/norm_expression"][:]
        cells = _decode(h["meta/cells/cellid"][:])
        entrez = _decode(h["meta/genes/entrez"][:])
        symbols = _decode(h["meta/genes/symbol"][:])
    if x.shape[0] != cells.size:              # 防御：若存为 genes × cells 则转置
        x = x.T
    rows, devs, mins = [], [], []
    for i in range(x.shape[0]):
        y, dev, mn = reconstruct_row(x[i])
        rows.append(sp.csr_matrix(y.astype(np.float32)))
        devs.append(dev)
        mins.append(mn)
    return Matrix(sp.vstack(rows).tocsr(), cells, entrez, symbols, np.array(devs), np.array(mins))
