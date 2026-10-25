"""Stage 03h 的工具：限定基因集的 TypeData、折内集成模型，以及 GEO 原始计数表的读取与对齐。

- 限定基因集：只保留给定基因的计数列，再按这些基因重新做文库大小归一化（与基础模型只见到这些基因一致）。
- 集成模型：与 Stage 03 run_fold 中 ContextModel 部分逐行相同（同高变基因选择、同种子 seed + k）。
- GEO 原始计数：三个 series 的计数表按细胞 ID 匹配（Stage 01 的 geo 列即按列名匹配得到）；
  基因按符号对齐，GEO 表中的旧符号经 HGNC previous symbol（唯一）映射到现行符号，同名行求和。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from context_model import ContextModel
from data import TypeData
from features import context_mean, fit_scaler, glucose_onehot, log_normalize


def restrict(td: TypeData, rows: np.ndarray | None = None, cols: np.ndarray | None = None,
             counts: sp.csr_matrix | None = None, scale: float = 1e4) -> TypeData:
    """按行（细胞）/列（基因）取子集；counts 给定时替换计数矩阵（行、列须已与子集对齐）。重新归一化。"""
    rows = np.arange(len(td.donor)) if rows is None else rows
    cols = np.arange(len(td.symbols)) if cols is None else cols
    c = td.counts[rows][:, cols].tocsr() if counts is None else counts.tocsr()
    cells = td.cells.iloc[rows].reset_index(drop=True)
    return replace(td, cells=cells, xlog=log_normalize(c, scale), symbols=td.symbols[cols], y=td.y[rows],
                   donor=td.donor[rows], group=td.group[rows], glucose=td.glucose[rows], counts=c)


def ensemble_fold(td: TypeData, cfg: dict, rep: int, fold: int, folds: np.ndarray, seed: int,
                  suffix: str) -> pd.DataFrame:
    """Stage 03 的 ContextModel（集成细胞分支、线性分支、岭回归、供体参照模型）单折预测。"""
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
        if len(te) == 0:
            continue
        mdl = ContextModel(mcfg, True, cfg["model"]["n_boost_features"], seed + k)
        preds = mdl.fit(z, mu, g1, y, td.donor, td.group, tr, cells_of).predict(z, mu, g1, td.donor, te, cells_of)
        for m in ("cell_branch", "linear_cell_branch", "ridge", "context_model"):
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": f"{m}{suffix}",
                                      "repeat": rep, "fold": fold, "y": y[te], "pred": preds[m]}))
    return pd.concat(rows)


# ---------------------------------------------------------------- GEO 原始计数

def hgnc_previous(hgnc_file: Path) -> dict:
    h = pd.read_csv(hgnc_file, sep="\t", dtype=str, usecols=["symbol", "prev_symbol"])
    p = h.dropna(subset=["prev_symbol"]).assign(prev=lambda d: d.prev_symbol.str.split("|")).explode("prev")
    p = p[~p.prev.duplicated(keep=False) & ~p.prev.isin(h.symbol)]
    return dict(zip(p.prev, p.symbol))


def read_geo_table(path: Path, cell_ids: list) -> pd.DataFrame:
    sep = "," if path.name.endswith(".csv.gz") else "\t"
    first = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig").columns[0]
    g = pd.read_csv(path, sep=sep, usecols=[first] + list(cell_ids), index_col=0, encoding="utf-8-sig")
    return g[list(cell_ids)]


def geo_counts(cells: pd.DataFrame, symbols: np.ndarray, raw_dir: Path, geo_files: dict,
               hgnc_file: Path) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """cells：含 cell_id、geo 列。返回 (细胞 × symbols 的 GEO 计数, 基因对齐表)。
    GEO 表中找不到的基因记为 0 并在对齐表中标出（found=False）。"""
    prev = hgnc_previous(hgnc_file)
    target = pd.Index(symbols)
    mats, found = [], np.zeros(len(symbols), dtype=int)
    order = []
    for acc, rel in geo_files.items():
        ids = cells.cell_id[cells.geo == acc].tolist()
        if not ids:
            continue
        g = read_geo_table(raw_dir / Path(rel).name, ids)
        key = pd.Series(g.index, index=g.index)
        mapped = key.map(prev)
        mapped[mapped.isin(key[key.isin(target)])] = np.nan        # 现行符号已直接存在时不再用旧符号行
        key = key.where(key.isin(target), mapped)
        g = g[key.notna().values].groupby(key[key.notna()].values).sum()
        found += target.isin(g.index).astype(int)
        m = g.reindex(target).fillna(0).T
        mats.append(sp.csr_matrix(np.rint(m.values.astype(np.float64))))
        order += ids
    x = sp.vstack(mats).tocsr()
    pos = pd.Series(np.arange(len(order)), index=order)
    x = x[pos[cells.cell_id].values]
    n_series = len([a for a in geo_files if (cells.geo == a).any()])
    ali = pd.DataFrame({"symbol": symbols, "n_series_found": found, "found": found == n_series})
    return x.tocsr(), ali


def concordance(a: sp.csr_matrix, b: sp.csr_matrix, cell_ids, geo) -> pd.DataFrame:
    """逐细胞：还原计数与 GEO 原始计数在任一方检出基因上的 Spearman 相关，及总计数、检出基因数。"""
    from scipy.stats import spearmanr
    rows = []
    for i, (cid, acc) in enumerate(zip(cell_ids, geo)):
        x, y = a[i].toarray().ravel(), b[i].toarray().ravel()
        k = (x > 0) | (y > 0)
        rows.append({"cell_id": cid, "geo": acc, "n_genes_union": int(k.sum()),
                     "recovered_total": float(x.sum()), "raw_total": float(y.sum()),
                     "recovered_detected": int((x > 0).sum()), "raw_detected": int((y > 0).sum()),
                     "spearman": spearmanr(x[k], y[k]).statistic if k.sum() > 2 else np.nan})
    return pd.DataFrame(rows)
