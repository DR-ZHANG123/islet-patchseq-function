"""预测头：内层交叉验证调参的梯度提升树，以及任意细胞表示（scVI 潜变量、基础模型冻结嵌入）上的
岭回归头、梯度提升树头与供体参照双分支模型（± 供体参照）。

梯度提升树调参：max_depth × learning_rate 网格（config sota.gbm_grid），迭代次数由内层验证误差曲线
（staged_predict）选择，即在训练供体内部按供体分组的内层交叉验证上做早停；选定后在全部训练细胞上重拟合。
表示头：表示在训练细胞上按维标准化并旋转到训练细胞的主成分轴（岭回归、供体分支对旋转不变）；梯度提升树
（独立的头与细胞分支中的部分）只用前 sota.boost_pcs 个主成分；供体参照 = 同供体同类型细胞标准化表示的均值（只用表达，不用功能标签）。
    <rep>_ridge        岭回归（α 按内层 MSE 选择，同 Stage 03 的 ridge）
    <rep>_gbm          调参梯度提升树（前 boost_pcs 个主成分 + 葡萄糖）
    <rep>_cell_branch  双分支模型的细胞分支（带供体内差分项的岭回归 + 梯度提升树的平均），不含供体参照
    <rep>_context      供体参照双分支模型（细胞分支 + 门控供体分支），与 Stage 03 的 context_model 同一实现
    <rep>_donor_branch 只用供体参照的岭回归
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from context_model import ContextModel
from features import context_mean, glucose_onehot


def tuned_gbm(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, inner: list, gcfg: dict,
              seed: int) -> tuple[np.ndarray, dict]:
    """inner：相对 tr 的内层折。返回测试细胞预测与所选 (max_depth, learning_rate, max_iter)。"""
    combos = list(itertools.product(gcfg["max_depth"], gcfg["learning_rate"]))
    curves = np.zeros((len(combos), gcfg["max_iter"]))
    for itr, iva in inner:
        a, b = tr[itr], tr[iva]
        for c, (depth, lr) in enumerate(combos):
            m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=gcfg["max_iter"],
                                              l2_regularization=gcfg["l2_regularization"], early_stopping=False,
                                              random_state=seed).fit(f[a], y[a])
            for it, p in enumerate(m.staged_predict(f[b])):
                curves[c, it] += ((p - y[b]) ** 2).sum()
            n_done = m.n_iter_
            if n_done < gcfg["max_iter"]:               # 理论上不会提前停止；如发生则沿用最后一步
                curves[c, n_done:] += ((p - y[b]) ** 2).sum()
    c, it = np.unravel_index(int(np.argmin(curves)), curves.shape)
    depth, lr = combos[c]
    m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=int(it) + 1,
                                      l2_regularization=gcfg["l2_regularization"], early_stopping=False,
                                      random_state=seed).fit(f[tr], y[tr])
    return m.predict(f[te]), {"gbm_max_depth": int(depth), "gbm_learning_rate": float(lr), "gbm_max_iter": int(it) + 1}


def standardize_rep(rep: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """按维标准化后旋转到训练细胞的主成分轴上（保留全部成分，不缩放）。
    岭回归与供体分支对正交旋转不变；梯度提升树只用前 boost_pcs 个主成分。"""
    mu = rep[tr].mean(0)
    sd = rep[tr].std(0) + 1e-6
    z = (rep - mu) / sd
    _, _, vt = np.linalg.svd(z[tr], full_matrices=False)
    return z @ vt.T


def rep_heads(rep: np.ndarray, y_all: np.ndarray, endpoints: list, donor, group, glucose, tr_all, te_all,
              cfg: dict, seed: int, prefix: str, inner_fn) -> tuple[dict, list]:
    """对每个终点拟合全部表示头。inner_fn(tr, seed) 返回相对 tr 的内层折。
    返回 {endpoint: (te, {model: pred})} 与超参数记录列表。"""
    z = standardize_rep(rep.astype(np.float64), tr_all)
    nb = min(cfg["sota"]["boost_pcs"], z.shape[1])
    mu = context_mean(z, donor)
    g1 = glucose_onehot(glucose)
    f = np.c_[z[:, :nb], g1]                      # 梯度提升树头：前 nb 个主成分 + 葡萄糖
    cells_of = {d: np.where(donor == d)[0] for d in np.unique(donor)}
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    out, params = {}, []
    for k, ep in enumerate(endpoints):
        y = y_all[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        if len(te) == 0:
            continue
        mdl = ContextModel(mcfg, True, nb, seed + k).fit(z, mu, g1, y, donor, group, tr, cells_of)
        p = mdl.predict(z, mu, g1, donor, te, cells_of)
        gb, gpar = tuned_gbm(f, y, tr, te, inner_fn(tr, seed + k), cfg["sota"]["gbm_grid"], seed + k)
        out[ep] = (te, {f"{prefix}_ridge": p["ridge"], f"{prefix}_gbm": gb, f"{prefix}_cell_branch": p["cell_branch"],
                        f"{prefix}_context": p["context_model"], f"{prefix}_donor_branch": p["donor_branch"]})
        params.append({"endpoint": ep, "representation": prefix, "alpha_cell": mdl.alpha_c, "within_lambda": mdl.lam,
                       "alpha_donor": mdl.alpha_d, "gate": mdl.gate, "alpha_ridge": mdl.alpha_p, **gpar})
    return out, params
