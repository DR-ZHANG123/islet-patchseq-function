"""表达特征与终点变换。所有依赖数据分布的步骤（高变基因、标准化）只在训练细胞上拟合。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


def log_normalize(counts: sp.csr_matrix, scale: float = 1e4) -> np.ndarray:
    x = counts.toarray().astype(np.float64)
    lib = x.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(x / lib * scale)


def transform_endpoint(v: np.ndarray, how: str) -> np.ndarray:
    v = np.asarray(v, float)
    if how == "log":
        return np.log(v)
    if how == "log1p":
        return np.log1p(np.clip(v, 0, None))
    return v


def glucose_onehot(glucose: np.ndarray, levels=(1, 5, 10)) -> np.ndarray:
    idx = pd.Series(glucose).map({g: i for i, g in enumerate(levels)}).values
    return np.eye(len(levels))[idx]


@dataclass
class Scaler:
    genes: np.ndarray       # 选中基因在输入矩阵中的列号
    mean: np.ndarray
    sd: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x[:, self.genes] - self.mean) / self.sd


def fit_scaler(x_train: np.ndarray, n_hvg: int, min_mean: float) -> Scaler:
    """在训练细胞上按离散度（方差/均值）选高变基因并拟合标准化参数。"""
    mu = x_train.mean(0)
    var = x_train.var(0)
    disp = np.where(mu > min_mean, var / (mu + 1e-3), -np.inf)
    genes = np.argsort(-disp)[:n_hvg]
    sub = x_train[:, genes]
    return Scaler(genes, sub.mean(0), sub.std(0) + 1e-6)


def context_mean(z: np.ndarray, context: np.ndarray) -> np.ndarray:
    """每个细胞所在供体参照组（同供体、同细胞类型）的平均表达，按细胞返回。"""
    df = pd.DataFrame(z)
    return df.groupby(context).transform("mean").values


def group_center(a: np.ndarray, group: np.ndarray) -> np.ndarray:
    """按比较组（同供体、同葡萄糖）去均值；a 可为向量或矩阵。"""
    if a.ndim == 1:
        return a - pd.Series(a).groupby(group).transform("mean").values
    return a - pd.DataFrame(a).groupby(group).transform("mean").values
