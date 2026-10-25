"""稳健性分析共用的留出供体交叉验证组件（Stage 04b–04f）。

- fold_features：外层折内只用训练细胞选高变基因与标准化（与 Stage 03 相同），可选：
  QC 协变量回归残差化（只在训练细胞上拟合）、留一供体参照（参照均值不含细胞自身）。
- LOOContextModel：供体参照与细胞分支参照均值 C̄_d 都去掉细胞自身；只有一个细胞的供体不做供体校正。
- permute_donor_offsets：供体层面置换零模型——供体平均偏移在供体间随机互换，供体内偏差保持不变；
  可选再在比较组（供体 × 葡萄糖）内打乱，得到各层面都没有表达—功能联系的完全零模型。
- run_cv_repeat：一次完整的 10 折留出供体交叉验证，返回逐细胞预测与超参数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from context_model import ContextModel
from data import donor_folds
from features import fit_scaler, glucose_onehot

MODELS = ["context_model", "cell_branch", "ridge", "linear_cell_branch", "linear_context_model", "donor_branch"]


def loo_mean(z: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """每个细胞所在供体其余细胞的平均表达；单细胞供体返回 0（训练细胞标准化后的总体均值）。"""
    df = pd.DataFrame(z)
    s = df.groupby(donor).transform("sum").values
    n = pd.Series(donor).map(pd.Series(donor).value_counts()).values[:, None].astype(float)
    out = np.zeros_like(z)
    m = n[:, 0] > 1
    out[m] = (s[m] - z[m]) / (n[m] - 1)
    return out


def residualize(z: np.ndarray, q: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """表达特征对 QC 协变量做线性回归取残差；系数只在训练细胞上估计，再重新标准化。"""
    qm, qs = q[tr].mean(0), q[tr].std(0) + 1e-9
    qq = np.c_[np.ones(len(q)), (q - qm) / qs]
    beta, *_ = np.linalg.lstsq(qq[tr], z[tr], rcond=None)
    r = z - qq @ beta
    return (r - r[tr].mean(0)) / (r[tr].std(0) + 1e-6)


def fold_features(td, tr_all, cfg, loo=False, qc=None):
    sc = fit_scaler(td.xlog[tr_all], cfg["features"]["n_hvg"], cfg["features"]["min_mean"])
    z = sc.transform(td.xlog)
    if qc is not None:
        z = residualize(z, qc, tr_all)
    mu = loo_mean(z, td.donor) if loo else pd.DataFrame(z).groupby(td.donor).transform("mean").values
    return z, mu, glucose_onehot(td.glucose)


class LOOContextModel(ContextModel):
    """留一供体参照：C̄_d 对每个目标细胞去掉其自身；单细胞供体的供体校正为 0。"""

    def _context_cell_mean(self, path, alpha, hgb, z, donor, cells_of, targets, g1_t):
        levels = g1_t.shape[1]
        lvl = g1_t.argmax(1)
        out = np.zeros(len(targets))
        for d in np.unique(donor[targets]):
            ctx = cells_of[d]
            preds = np.column_stack([self._cell_pred(path, alpha, hgb, z[ctx], np.tile(np.eye(levels)[k], (len(ctx), 1)))
                                     for k in range(levels)])           # (n_ctx, levels)
            tot = preds.sum(0)
            pos = {c: i for i, c in enumerate(ctx)}
            for j in np.where(donor[targets] == d)[0]:
                i = pos.get(targets[j])
                if i is None:
                    out[j] = tot[lvl[j]] / len(ctx)
                elif len(ctx) > 1:
                    out[j] = (tot[lvl[j]] - preds[i, lvl[j]]) / (len(ctx) - 1)
                else:
                    out[j] = preds[i, lvl[j]]                             # 单细胞供体：无其他参照细胞（预测时不做校正）
        return out

    def predict(self, z, mu, g1, donor, idx, cells_of) -> dict:
        out = super().predict(z, mu, g1, donor, idx, cells_of)
        single = np.array([len(cells_of[d]) == 1 for d in donor[idx]])
        for k in ("context_model", "linear_context_model", "context_model_full_gate"):
            if k in out:
                base = out["cell_branch" if k != "linear_context_model" else "linear_cell_branch"]
                out[k] = np.where(single, base, out[k])
        return out


def permute_donor_offsets(y, donor, glucose, rng, within=False, group=None):
    """供体层面置换：y' = y − m_d + m_π(d)，m_d 为供体在葡萄糖内去均值后的平均偏移；within=True 时再在组内打乱。"""
    y2 = y.copy()
    ok = ~np.isnan(y)
    r = y.copy()
    for g in np.unique(glucose):
        m = ok & (glucose == g)
        r[m] = y[m] - y[m].mean()
    off = pd.Series(r[ok]).groupby(donor[ok]).mean()
    perm = pd.Series(rng.permutation(off.values), index=off.index)
    dd = donor[ok]
    y2[ok] = y[ok] - off.loc[dd].values + perm.loc[dd].values
    if within:
        for g in np.unique(group[ok]):
            idx = np.where(ok & (group == g))[0]
            y2[idx] = y2[rng.permutation(idx)]
    return y2


def fit_predict(td, z, mu, g1, y, tr, te, cells_of, mcfg, nb, seed, loo=False):
    cls = LOOContextModel if loo else ContextModel
    mdl = cls(mcfg, True, nb, seed).fit(z, mu, g1, y, td.donor, td.group, tr, cells_of)
    preds = mdl.predict(z, mu, g1, td.donor, te, cells_of)
    params = {"alpha_cell": mdl.alpha_c, "within_lambda": mdl.lam, "alpha_donor": mdl.alpha_d,
              "gate": mdl.gate, "gate_linear": mdl.gate_lin, "alpha_ridge": mdl.alpha_p}
    return preds, params


def run_fold(td, cfg, fold, folds, seed, endpoints, alphas=None, loo=False, qc=None, ymat=None, train_mask=None,
             tag=None):
    """单个外层折：对 endpoints 中每个终点拟合双分支模型。ymat 给定时用它代替实测终点（置换零模型）；
    train_mask 给定时训练细胞只取其中为 True 者（训练规模下采样）。"""
    tr_all = np.where((folds != fold) & (train_mask if train_mask is not None else True))[0]
    te_all = np.where(folds == fold)[0]
    z, mu, g1 = fold_features(td, tr_all, cfg, loo=loo, qc=qc)
    cells_of = {d: np.where(td.donor == d)[0] for d in np.unique(td.donor)}
    mcfg = cfg["model"] | {"inner_folds": cfg["cv"]["inner_folds"]}
    if alphas is not None:
        mcfg = mcfg | {"alphas": alphas}
    nb = cfg["model"]["n_boost_features"]
    rows, params = [], []
    for ep in endpoints:
        k = td.endpoints.index(ep)
        y = (ymat if ymat is not None else td.y)[:, k]
        ok = ~np.isnan(y)
        tr, te = tr_all[ok[tr_all]], te_all[ok[te_all]]
        if len(te) == 0:
            continue
        preds, par = fit_predict(td, z, mu, g1, y, tr, te, cells_of, mcfg, nb, seed + k, loo=loo)
        for m in MODELS:
            rows.append(pd.DataFrame({"cell_id": td.cells.cell_id.values[te], "endpoint": ep, "model": m,
                                      "fold": fold, "y": y[te], "pred": preds[m]}))
        params.append({"endpoint": ep, "fold": fold, **par})
    out = pd.concat(rows, ignore_index=True)
    par = pd.DataFrame(params)
    if tag:
        for k, v in tag.items():
            out[k] = v
            par[k] = v
    return out, par


def repeat_folds(td, cfg, rep):
    """与 Stage 03 相同的折划分（重复 rep 的种子 = seed + 1000·rep），折内种子 = 重复种子 + 37·折号。"""
    seed = cfg["seed"] + 1000 * rep
    return donor_folds(td.donor, cfg["cv"]["outer_folds"], seed), seed


def checkpointed(path, fn, *args, **kwargs):
    """作业级断点：结果已存在则直接读取，否则计算后原子写入（中断后重跑只补未完成的作业）。"""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return pd.read_pickle(path)
    res = fn(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.to_pickle(res, tmp)
    tmp.replace(path)
    return res
