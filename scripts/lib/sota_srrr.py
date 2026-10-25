"""稀疏降秩回归（sparse reduced-rank regression, Kobak 等 2021, JRSS-C）：Patch-seq 中转录组 → 电生理的多输出线性标准方法。

模型：Ŷ = X W Vᵀ，V 为 q×r 正交矩阵（VᵀV = I），W 为 p×r 系数矩阵；目标函数
    (1/2n)·||Y − X W Vᵀ||²_F + λ·[ a·Σ_j ||W_j·||₂ + (1−a)/2·||W||²_F ]
其中行组套索（group lasso）使整行 W_j· 为零，即整基因被剔除；a 为 l1 比例（弹性网）。
交替求解：给定 V，因 V 正交，||Y − XWVᵀ||² = ||YV − XW||² + 常数，W 步即多任务弹性网（sklearn
MultiTaskElasticNet 的目标与上式一致）；给定 W，V 步为正交 Procrustes：YᵀXW = U S Qᵀ，V = U Qᵀ。
终点缺测：在交替过程中用当前拟合值 X W Vᵀ 填补缺测项（EM 式），只用观测项计算误差。
葡萄糖条件作为不受惩罚的固定效应：拟合前 X、Y 都在葡萄糖水平内去均值（Frisch–Waugh），预测时加回该水平均值。
秩 r、l1 比例 a、惩罚 λ 在训练供体内部按供体分组的内层交叉验证选择（标准化终点上观测项的均方误差）。
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _level_means(a: np.ndarray, lvl: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """每个葡萄糖水平内的列均值（忽略 NaN），返回 (len(levels), ncol)。"""
    return np.vstack([np.nanmean(a[lvl == g], 0) if (lvl == g).any() else np.zeros(a.shape[1]) for g in levels])


class _Prep:
    """在拟合细胞上估计葡萄糖水平均值与终点标准差；transform 后 Y 的缺测仍为 NaN。"""

    def __init__(self, x, y, lvl, levels):
        self.levels = levels
        self.mx = _level_means(x, lvl, levels)
        self.my = _level_means(y, lvl, levels)
        self.my = np.nan_to_num(self.my)
        self.sy = np.nanstd(y - self.my[np.searchsorted(levels, lvl)], 0) + 1e-8

    def x(self, x, lvl):
        return x - self.mx[np.searchsorted(self.levels, lvl)]

    def y(self, y, lvl):
        return (y - self.my[np.searchsorted(self.levels, lvl)]) / self.sy

    def y_back(self, ys, lvl):
        return ys * self.sy + self.my[np.searchsorted(self.levels, lvl)]


def lambda_grid(xc: np.ndarray, ys: np.ndarray, l1: float, n_lambda: int, min_ratio: float) -> np.ndarray:
    """λ 网格：λ_max 为 W = 0 时满足 KKT 条件的最小 λ（用 ||X_jᵀY||₂ 作为 ||X_jᵀYV||₂ 的上界），对数等距递减。"""
    y0 = np.nan_to_num(ys)
    lmax = np.max(np.linalg.norm(xc.T @ y0, axis=1)) / (len(xc) * l1)
    return lmax * np.logspace(0, np.log10(min_ratio), n_lambda + 1)[1:]


def _procrustes(y_fill: np.ndarray, xw: np.ndarray, v_old: np.ndarray) -> np.ndarray:
    m = y_fill.T @ xw                                   # (q, r)
    if not np.any(np.abs(m) > 1e-12):
        return v_old
    u, _, qt = np.linalg.svd(m, full_matrices=False)
    return u @ qt


def fit_path(xc: np.ndarray, ys: np.ndarray, rank: int, l1: float, lambdas: np.ndarray,
             max_alt: int, tol: float) -> list:
    """沿递减的 λ 路径热启动拟合，返回每个 λ 的 (W, V)。xc 已去均值，ys 已标准化（NaN 为缺测）。"""
    miss = np.isnan(ys)
    y_fill = np.where(miss, 0.0, ys)
    q = ys.shape[1]
    r = min(rank, q)
    _, _, vt = np.linalg.svd(y_fill, full_matrices=False)
    v = vt[:r].T                                        # 初值：Y 的前 r 个右奇异向量
    enet = MultiTaskElasticNet(alpha=lambdas[0], l1_ratio=l1, fit_intercept=False, warm_start=True,
                               max_iter=2000, tol=1e-4)
    out, fit = [], np.zeros_like(y_fill)
    for lam in lambdas:
        enet.set_params(alpha=lam)
        for _ in range(max_alt):
            enet.fit(xc, y_fill @ v)
            w = enet.coef_.T                            # (p, r)
            xw = xc @ w
            v = _procrustes(y_fill, xw, v)
            new = xw @ v.T
            y_fill[miss] = new[miss]                    # 缺测项以当前拟合值填补
            # 收敛：拟合矩阵的相对变化小于 tol
            done = np.linalg.norm(new - fit) <= tol * max(np.linalg.norm(new), 1e-12)
            fit = new
            if done:
                break
        out.append((w.copy(), v.copy()))
    return out


def _nanmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~np.isnan(a)
    return float(np.mean((a[ok] - b[ok]) ** 2))


def select_and_fit(x: np.ndarray, y: np.ndarray, lvl: np.ndarray, tr: np.ndarray, te: np.ndarray, inner: list,
                   scfg: dict) -> tuple[np.ndarray, dict]:
    """x：(n, p) 特征（训练折内标准化的高变基因）；y：(n, q) 终点（NaN 缺测）；lvl：葡萄糖水平；
    inner：训练细胞内（相对 tr 的下标）按供体分组的内层折。返回测试细胞原尺度预测 (len(te), q) 与所选参数。"""
    levels = np.unique(lvl)
    prep = _Prep(x[tr], y[tr], lvl[tr], levels)
    xc, ys = prep.x(x[tr], lvl[tr]), prep.y(y[tr], lvl[tr])
    grids = {a: lambda_grid(xc, ys, a, scfg["n_lambda"], scfg["lambda_min_ratio"]) for a in scfg["l1_ratios"]}
    err = {}
    for itr, iva in inner:
        # 内层折：葡萄糖均值在内层训练细胞上重新估计；终点仍按外层训练细胞的标准差缩放
        ip = _Prep(x[tr][itr], y[tr][itr], lvl[tr][itr], levels)
        ixc = ip.x(x[tr][itr], lvl[tr][itr])
        iys = (y[tr][itr] - ip.my[np.searchsorted(levels, lvl[tr][itr])]) / prep.sy
        vxc = ip.x(x[tr][iva], lvl[tr][iva])
        vys = (y[tr][iva] - ip.my[np.searchsorted(levels, lvl[tr][iva])]) / prep.sy
        for a, lams in grids.items():
            for rk in scfg["ranks"]:
                for j, (w, v) in enumerate(fit_path(ixc, iys, rk, a, lams, scfg["max_alt"], scfg["tol"])):
                    key = (rk, a, j)
                    err[key] = err.get(key, 0.0) + _nanmse(vys, vxc @ w @ v.T) * len(iva)
    rk, a, j = min(err, key=err.get)
    lams = grids[a]
    w, v = fit_path(xc, ys, rk, a, lams[:j + 1], scfg["max_alt"], scfg["tol"])[-1]
    pred = prep.y_back(prep.x(x[te], lvl[te]) @ w @ v.T, lvl[te])
    n_genes = int((np.linalg.norm(w, axis=1) > 0).sum())
    return pred, {"srrr_rank": rk, "srrr_l1_ratio": a, "srrr_lambda": float(lams[j]), "srrr_n_genes": n_genes}
