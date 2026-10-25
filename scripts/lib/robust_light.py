"""只拟合细胞分支（岭回归–梯度提升集成）与岭回归的轻量留出供体交叉验证（Stage 04g、04h）。

CellOnlyModel 与 ContextModel 的第 1 步（内层交叉验证选 α、λ）与第 3 步（全部训练细胞重拟合）完全相同，
只省去供体分支与合成权重 g 的选择（它们不影响 cell_branch、ridge、linear_cell_branch 的预测），因此这三个
模型的预测与 robust_cv.run_fold 逐位一致（check_equivalence 核对），而每折的梯度提升树从 6 次减为 1 次。
另含供体层面的分层置换零模型：
- within：供体平均偏移只在同一分层（来源中心，或 来源中心 × GEO series）内的供体之间互换；
- residual：供体平均偏移对分层哑变量（中心 + series，加性）回归，拟合值保留，残差在全部供体间互换。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import context_model as cm
from features import group_center
from robust_cv import fold_features

LIGHT_MODELS = ["cell_branch", "ridge", "linear_cell_branch"]


class CellOnlyModel(cm.ContextModel):
    def fit(self, z, g1, y, donor, group, tr):
        cfg = self.cfg
        alphas = np.asarray(cfg["alphas"], float)
        folds = cm._inner_folds(donor[tr], cfg["inner_folds"], self.seed)
        ytr, gtr = y[tr], group[tr]
        ycen = group_center(ytr, gtr)
        pred = {lam: np.zeros((len(tr), len(alphas))) for lam in cfg["within_lambdas"]}
        for f in np.unique(folds):
            itr, iva = tr[folds != f], tr[folds == f]
            fc = np.c_[z[itr], g1[itr]]
            for lam in cfg["within_lambdas"]:
                path = cm.fit_augmented(fc, y[itr], group_center(fc, group[itr]), group_center(y[itr], group[itr]),
                                        lam)
                pred[lam][folds == f] = path.predict_path(np.c_[z[iva], g1[iva]], alphas)
        _, self.lam, jc = min(((cm._mse(ytr, pred[l][:, j]) + cm._mse(ycen, group_center(pred[l][:, j], gtr)), l, j)
                               for l in cfg["within_lambdas"] for j in range(len(alphas))))
        self.alpha_c = alphas[jc]
        self.path_c, self.hgb = self._fit_cell(z, g1, y, group, tr, self.lam)
        self.alpha_p = alphas[int(np.argmin([cm._mse(ytr, pred[0][:, j]) for j in range(len(alphas))]))]
        self.path_p = self.path_c if self.lam == 0 else cm.fit_path(np.c_[z[tr], g1[tr]], ytr)
        return self

    def _fit_cell(self, z, g1, y, group, idx, lam):
        fc = np.c_[z[idx], g1[idx]]
        path = cm.fit_augmented(fc, y[idx], group_center(fc, group[idx]), group_center(y[idx], group[idx]), lam)
        return path, self._fit_boost(z, g1, y, idx)

    def predict(self, z, g1, idx) -> dict:
        return {"cell_branch": self._cell_pred(self.path_c, self.alpha_c, self.hgb, z[idx], g1[idx]),
                "linear_cell_branch": self._cell_pred(self.path_c, self.alpha_c, None, z[idx], g1[idx]),
                "ridge": self.path_p.predict(np.c_[z[idx], g1[idx]], self.alpha_p)}


def fold_inputs(td, cfg, fold, folds, train_mask=None):
    """外层折的训练/测试细胞与特征（高变基因、标准化只用训练细胞）。"""
    tr_all = np.where((folds != fold) & (train_mask if train_mask is not None else True))[0]
    te_all = np.where(folds == fold)[0]
    z, _, g1 = fold_features(td, tr_all, cfg)
    return tr_all, te_all, z, g1


def run_fold_light(td, cfg, fold, seed, endpoints, inputs, ymat=None, tag=None):
    """inputs = fold_inputs(...)；种子约定与 robust_cv.run_fold 相同（折种子 + 终点序号）。"""
    tr_all, te_all, z, g1 = inputs
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    rows = []
    for ep in endpoints:
        k = td.endpoints.index(ep)
        y = (ymat if ymat is not None else td.y)[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        if len(te) == 0:
            continue
        mdl = CellOnlyModel(mcfg, True, cfg["model"]["n_boost_features"], seed + k).fit(z, g1, y, td.donor,
                                                                                        td.group, tr)
        for m, p in mdl.predict(z, g1, te).items():
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": m,
                                      "fold": fold, "y": y[te], "pred": p}))
    out = pd.concat(rows, ignore_index=True)
    for key, v in (tag or {}).items():
        out[key] = v
    return out


def check_equivalence(td, cfg, fold, folds, seed, endpoint) -> float:
    """与 robust_cv.run_fold 的最大绝对差（应为 0）。"""
    import robust_cv as rc
    full, _ = rc.run_fold(td, cfg, fold, folds, seed, [endpoint])
    light = run_fold_light(td, cfg, fold, seed, [endpoint], fold_inputs(td, cfg, fold, folds))
    a = full[full.model.isin(LIGHT_MODELS)].sort_values(["model", "cell_id"]).pred.values
    b = light.sort_values(["model", "cell_id"]).pred.values
    return float(np.max(np.abs(a - b)))


# ---------- 分层置换零模型 ----------
def donor_offsets(y, donor, glucose):
    """供体平均偏移（葡萄糖内去均值后取供体均值），定义与 robust_cv.permute_donor_offsets 相同。"""
    ok = ~np.isnan(y)
    r = y.copy()
    for g in np.unique(glucose):
        m = ok & (glucose == g)
        r[m] = y[m] - y[m].mean()
    return pd.Series(r[ok]).groupby(donor[ok]).mean(), ok


def stratum_design(strata: pd.DataFrame) -> np.ndarray:
    """加性哑变量设计矩阵（截距 + 每个分层变量去掉一个参照水平）。"""
    cols = [np.ones(len(strata))]
    for c in strata.columns:
        d = pd.get_dummies(strata[c].astype(str), drop_first=True, dtype=float)
        cols += [d[k].values for k in d.columns]
    return np.column_stack(cols)


def permute_offsets_strata(y, donor, glucose, rng, strata: pd.DataFrame, mode: str):
    """strata：以供体为索引、每列一个分层变量。mode = within（分层内互换）或 residual（回归残差互换）。"""
    off, ok = donor_offsets(y, donor, glucose)
    st = strata.loc[off.index]
    new = off.copy()
    if mode == "within":
        key = st.astype(str).agg("|".join, axis=1)
        for _, idx in off.groupby(key.values).groups.items():
            new.loc[idx] = rng.permutation(off.loc[idx].values)
    elif mode == "residual":
        x = stratum_design(st)
        beta, *_ = np.linalg.lstsq(x, off.values, rcond=None)
        fit = x @ beta
        new[:] = fit + rng.permutation(off.values - fit)
    else:
        raise ValueError(mode)
    y2 = y.copy()
    dd = donor[ok]
    y2[ok] = y[ok] - off.loc[dd].values + new.loc[dd].values
    return y2
