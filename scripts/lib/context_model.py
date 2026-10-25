"""供体参照双分支模型（细胞分支可为岭回归与梯度提升树的平均）。

细胞分支 C(x, c)：单细胞表达 x 与葡萄糖条件 c 的函数，取以下学习器预测的平均：
    (1) 带供体内差分监督的岭回归：||y − Xb||² + λ·||y_c − X_c b||² + α||b||²；
    (2) 梯度提升树（可选）。
供体分支 D(μ_d, c)：以同供体、同类型细胞平均表达 μ_d 为输入的岭回归。
合成：ŷ_i = C(x_i, c_i) + g·[D(μ_d, c_i) − C̄_d(c_i)]，C̄_d(c) 为该供体全部参照细胞在条件 c 下细胞分支预测的均值，
    即供体层面的水平按权重 g 由供体分支校正，细胞相对供体的差异始终由细胞分支给出。
α、λ、α_D、g 均在训练供体内部按供体分组的内层交叉验证中选择。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from features import group_center
from ridge import fit_augmented, fit_path


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _inner_folds(donors, k, seed):
    ud = np.unique(donors)
    perm = dict(zip(ud, np.random.default_rng(seed).permutation(len(ud))))
    return np.array([perm[d] % k for d in donors])


class ContextModel:
    def __init__(self, cfg: dict, use_boosting: bool, n_boost_features: int, seed: int):
        self.cfg, self.use_boosting, self.nb, self.seed = cfg, use_boosting, n_boost_features, seed

    # ---------- 细胞分支 ----------
    def _fit_boost(self, z, g1, y, idx):
        if not self.use_boosting:
            return None
        hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                            random_state=self.seed)
        return hgb.fit(np.c_[z[idx, :self.nb], g1[idx]], y[idx])

    def _fit_cell(self, z, g1, y, group, idx, lam):
        fc = np.c_[z[idx], g1[idx]]
        path = fit_augmented(fc, y[idx], group_center(fc, group[idx]), group_center(y[idx], group[idx]), lam)
        return path, self._fit_boost(z, g1, y, idx)

    def _cell_pred(self, path, alpha, hgb, z, g1):
        p = path.predict(np.c_[z, g1], alpha)
        if hgb is not None:
            p = 0.5 * p + 0.5 * hgb.predict(np.c_[z[:, :self.nb], g1])
        return p

    def _context_cell_mean(self, path, alpha, hgb, z, donor, cells_of, targets, g1_t):
        """对 targets 中每个细胞：其供体全部参照细胞在该细胞葡萄糖条件下的细胞分支预测均值。"""
        levels = g1_t.shape[1]
        out = np.zeros(len(targets))
        lvl = g1_t.argmax(1)
        for d in np.unique(donor[targets]):
            ctx = cells_of[d]
            means = [self._cell_pred(path, alpha, hgb, z[ctx], np.tile(np.eye(levels)[k], (len(ctx), 1))).mean()
                     for k in range(levels)]
            m = donor[targets] == d
            out[m] = np.asarray(means)[lvl[m]]
        return out

    # ---------- 拟合 ----------
    def fit(self, z, mu, g1, y, donor, group, tr, cells_of):
        cfg = self.cfg
        alphas = np.asarray(cfg["alphas"], float)
        folds = _inner_folds(donor[tr], cfg["inner_folds"], self.seed)
        n = len(tr)
        ytr, gtr = y[tr], group[tr]
        ycen = group_center(ytr, gtr)
        # 1) 岭回归部分的 α、λ（绝对误差 + 供体内误差）
        cache, pred = {}, {lam: np.zeros((n, len(alphas))) for lam in cfg["within_lambdas"]}
        pred_d = np.zeros((n, len(alphas)))
        for f in np.unique(folds):
            itr, iva = tr[folds != f], tr[folds == f]
            fc = np.c_[z[itr], g1[itr]]
            for lam in cfg["within_lambdas"]:
                path = fit_augmented(fc, y[itr], group_center(fc, group[itr]), group_center(y[itr], group[itr]), lam)
                cache[(lam, f)] = path
                pred[lam][folds == f] = path.predict_path(np.c_[z[iva], g1[iva]], alphas)
            pred_d[folds == f] = fit_path(np.c_[mu[itr], g1[itr]], y[itr]).predict_path(np.c_[mu[iva], g1[iva]], alphas)
        best = min(((_mse(ytr, pred[l][:, j]) + _mse(ycen, group_center(pred[l][:, j], gtr)), l, j)
                    for l in cfg["within_lambdas"] for j in range(len(alphas))))
        _, self.lam, jc = best
        self.alpha_c = alphas[jc]
        jd = int(np.argmin([_mse(ytr, pred_d[:, j]) for j in range(len(alphas))]))
        self.alpha_d = alphas[jd]
        # 2) 合成权重 g：线性版（细胞分支只用岭回归）与集成版（岭回归 + 梯度提升树）分别在内层验证上选择
        d_in = pred_d[:, jd]
        c_lin, cbar_lin = np.zeros(n), np.zeros(n)
        c_ens, cbar_ens = np.zeros(n), np.zeros(n)
        for f in np.unique(folds):
            itr, iva = tr[folds != f], tr[folds == f]
            path = cache[(self.lam, f)]
            m = folds == f
            c_lin[m] = self._cell_pred(path, self.alpha_c, None, z[iva], g1[iva])
            cbar_lin[m] = self._context_cell_mean(path, self.alpha_c, None, z, donor, cells_of, iva, g1[iva])
            if self.use_boosting:
                hgb = self._fit_boost(z, g1, y, itr)
                c_ens[m] = self._cell_pred(path, self.alpha_c, hgb, z[iva], g1[iva])
                cbar_ens[m] = self._context_cell_mean(path, self.alpha_c, hgb, z, donor, cells_of, iva, g1[iva])
        pick = lambda c, cb: cfg["gates"][int(np.argmin([_mse(ytr, c + g * (d_in - cb)) for g in cfg["gates"]]))]  # noqa: E731
        self.gate_lin = pick(c_lin, cbar_lin)
        self.gate = pick(c_ens, cbar_ens) if self.use_boosting else self.gate_lin
        # 3) 全部训练细胞上重拟合
        self.path_c, self.hgb = self._fit_cell(z, g1, y, group, tr, self.lam)
        self.path_d = fit_path(np.c_[mu[tr], g1[tr]], ytr)
        self.alpha_p = alphas[int(np.argmin([_mse(ytr, pred[0][:, j]) for j in range(len(alphas))]))]
        self.path_p = self.path_c if self.lam == 0 else fit_path(np.c_[z[tr], g1[tr]], ytr)
        return self

    def predict(self, z, mu, g1, donor, idx, cells_of) -> dict:
        d = self.path_d.predict(np.c_[mu[idx], g1[idx]], self.alpha_d)
        cl = self._cell_pred(self.path_c, self.alpha_c, None, z[idx], g1[idx])
        cbl = self._context_cell_mean(self.path_c, self.alpha_c, None, z, donor, cells_of, idx, g1[idx])
        out = {"donor_branch": d, "linear_cell_branch": cl, "linear_context_model": cl + self.gate_lin * (d - cbl),
               "ridge": self.path_p.predict(np.c_[z[idx], g1[idx]], self.alpha_p)}
        if self.use_boosting:
            c = self._cell_pred(self.path_c, self.alpha_c, self.hgb, z[idx], g1[idx])
            cb = self._context_cell_mean(self.path_c, self.alpha_c, self.hgb, z, donor, cells_of, idx, g1[idx])
            out |= {"cell_branch": c, "context_model": c + self.gate * (d - cb), "context_model_full_gate": c + d - cb}
        return out
