"""技术协变量（Stage 04f）：细胞质控特征、批次特征、小特征集模型与控制质控后的供体内偏相关。

质控特征（每个细胞，均来自还原计数）：log 文库大小、log 检出基因数、线粒体转录本比例、核糖体蛋白转录本比例、
MALAT1 比例、本类型激素比例（β：INS；α：GCG）、他类激素 log1p(CPM)（β：GCG；α：INS，
即环境 RNA/混杂评分；多于一种时另加总和）、细胞类型注释置信度。
批次特征：GEO 系列与来源中心的哑变量。两类模型都加入葡萄糖哑变量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import glucose_onehot
from ridge import fit_path


def qc_features(td, cfg: dict, ct: str) -> pd.DataFrame:
    tc = cfg["robustness"]["technical"]
    x = td.counts.tocsr().astype(np.float64)
    lib = np.asarray(x.sum(1)).ravel()
    lib[lib == 0] = 1
    sym = pd.Series(td.symbols)

    def frac(mask):
        cols = np.where(mask.values)[0]
        return np.asarray(x[:, cols].sum(1)).ravel() / lib if len(cols) else np.zeros(len(lib))

    def cpm(gene):
        cols = np.where((sym == gene).values)[0]
        return np.log1p(np.asarray(x[:, cols].sum(1)).ravel() / lib * 1e6) if len(cols) else np.zeros(len(lib))

    q = pd.DataFrame({"log_n_counts": np.log(td.cells.n_counts.values), "log_n_genes": np.log(td.cells.n_genes.values),
                      "mito_fraction": frac(sym.str.startswith("MT-")),
                      "ribo_fraction": frac(sym.str.match(r"^RP[SL]\d")),
                      "malat1_fraction": frac(sym == "MALAT1"),
                      "own_hormone_fraction": frac(sym == tc["own_hormone"][ct]),
                      "cell_score": td.cells.cell_score.values})
    other = []
    for h in tc["hormones"][ct]:
        q[f"ambient_{h}"] = cpm(h)
        other.append(q[f"ambient_{h}"].values)
    if len(other) > 1:
        q["ambient_hormone_score"] = np.sum(other, axis=0)
    return q


def batch_features(td, cols) -> np.ndarray:
    return pd.get_dummies(td.cells[cols].astype(str), drop_first=False).values.astype(float)


def _std(f, tr):
    m, s = f[tr].mean(0), f[tr].std(0)
    s[s < 1e-9] = 1.0
    return (f - m) / s


def ridge_small(f, y, tr, te, donor, alphas, k_inner, seed):
    """小特征集岭回归：特征按训练细胞标准化，惩罚由训练供体内按供体分组的内层交叉验证选择。"""
    fs = _std(f, tr)
    ud = np.unique(donor[tr])
    perm = dict(zip(ud, np.random.default_rng(seed).permutation(len(ud))))
    fold = np.array([perm[d] % k_inner for d in donor[tr]])
    err = np.zeros(len(alphas))
    for k in range(k_inner):
        a, b = tr[fold != k], tr[fold == k]
        err += ((fit_path(fs[a], y[a]).predict_path(fs[b], alphas) - y[b][:, None]) ** 2).sum(0)
    al = float(np.asarray(alphas)[int(np.argmin(err))])
    return fit_path(fs[tr], y[tr]).predict(fs[te], al)


def boosting_small(f, y, tr, te, seed):
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                      random_state=seed)
    return m.fit(f[tr], y[tr]).predict(f[te])


def feature_sets(td, q: pd.DataFrame, cfg: dict) -> dict:
    g1 = glucose_onehot(td.glucose)
    b = batch_features(td, cfg["robustness"]["technical"]["batch_columns"])
    return {"qc_only": np.c_[q.values, g1], "batch_only": np.c_[b, g1], "qc_batch": np.c_[q.values, b, g1]}


def _centre(a, group):
    return a - pd.DataFrame(a).groupby(group).transform("mean").values


def partial_within(y, p, qmat, group, donor, n_boot, rng):
    """比较组内去均值后，y 与 p 分别对质控特征做线性回归取残差，残差的合并 Pearson；按供体重抽样置信区间。"""
    yc = _centre(y[:, None], group)[:, 0]
    pc = _centre(p[:, None], group)[:, 0]
    qc = _centre(qmat, group)
    qc = (qc - qc.mean(0)) / (qc.std(0) + 1e-12)

    def stat(idx):
        qq = np.c_[np.ones(len(idx)), qc[idx]]
        ry = yc[idx] - qq @ np.linalg.lstsq(qq, yc[idx], rcond=None)[0]
        rp = pc[idx] - qq @ np.linalg.lstsq(qq, pc[idx], rcond=None)[0]
        raw = np.corrcoef(yc[idx], pc[idx])[0, 1]
        return np.corrcoef(ry, rp)[0, 1], raw

    allidx = np.arange(len(y))
    point = stat(allidx)
    cells_of = {d: np.where(donor == d)[0] for d in np.unique(donor)}
    ud = np.array(list(cells_of))
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ud, len(ud), replace=True)
        bs.append(stat(np.concatenate([cells_of[d] for d in pick])))
    bs = np.array(bs)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    d = bs[:, 0] - bs[:, 1]
    return {"within_raw": point[1], "within_partial_qc": point[0], "partial_lo": lo[0], "partial_hi": hi[0],
            "raw_lo": lo[1], "raw_hi": hi[1], "change": point[0] - point[1],
            "change_lo": np.nanpercentile(d, 2.5), "change_hi": np.nanpercentile(d, 97.5)}


def qc_endpoint_within(y, q: pd.DataFrame, group) -> dict:
    """每个质控特征与终点的供体内（比较组内去均值）Pearson。"""
    ok = ~np.isnan(y)
    yc = _centre(y[ok][:, None], group[ok])[:, 0]
    qc = _centre(q.values[ok], group[ok])
    out = {}
    for j, c in enumerate(q.columns):
        out[c] = float(np.corrcoef(yc, qc[:, j])[0, 1]) if qc[:, j].std() > 1e-12 else np.nan
    return out
