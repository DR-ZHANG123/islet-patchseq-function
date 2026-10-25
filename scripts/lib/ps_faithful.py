"""Camunas-Soler 等 2020（Cell Metab）原始“预测基因集”（PS）+ kNN 方法的折内忠实重建。

原作者代码（github.com/jcamunas/patchseq）对应关系：
- 基因–电生理相关：每个参数在训练细胞上做 Spearman（含并列校正）相关，按细胞有放回重抽样 1000 次，
  z = 重抽样相关均值 / 重抽样标准差（其结果表的 zscore_raw_mean = corr_raw_mean / std_raw_mean）。
  相关前：表达只保留训练细胞平均 log2(CPM+1) > 1 的基因；每个参数去掉最高、最低 3% 的细胞（含分位点），
  电流取正值方向，负值截为 0（helper_functions.filter_dataset_ephys / filter_quantile_values）。
- PS 基因：filter_correlation_table(z_th=2, zscore_raw=True, n_feat=1, n_blocks=2, signed='both')，
  即 |z|>2 的参数分属至少两个功能组；再保留训练细胞 log2(CPM+1) 中位数 > 0 的基因（Figure 3 笔记本）。
  一个基因集用于全部参数。
- 预测：predict_ephys(k=5, method='spearman', weight_method='flat', use_top_cell=True)：测试细胞与训练细胞在
  PS 基因上的 Spearman 相关（逐细胞对基因排序）取前 k 个近邻，预测为近邻实测值的等权平均（缺测值屏蔽）。
  训练细胞的目标值先截到其第 90 百分位、负值截为 0（Figure 3 笔记本对 ds_train 的处理）。
本研究的度量名与原文不同；功能组映射与其余不可避免的差异见 config ps_faithful 的注释。
"""
from __future__ import annotations

import numpy as np
import torch

from features import transform_endpoint


# ---------------------------------------------------------------- 表达

def log2cpm(counts) -> np.ndarray:
    """原始计数 → log2(CPM + 1)（原作者 singlet: counts.normalize() 后 log2，伪计数 1）。"""
    x = counts.toarray().astype(np.float64) if hasattr(counts, "toarray") else np.asarray(counts, float)
    lib = x.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log2(x / lib * 1e6 + 1.0)


def prefilter_genes(raw_tr: np.ndarray, min_cells: int, min_counts: int) -> np.ndarray:
    """filter_genes_pp：至少 min_cells 个细胞检出且总计数 ≥ min_counts（只用训练细胞）。"""
    return np.where(((raw_tr > 0).sum(0) >= min_cells) & (raw_tr.sum(0) >= min_counts))[0]


# ---------------------------------------------------------------- 电生理预处理

def corr_targets(raw: dict, tr: np.ndarray, q: tuple) -> dict:
    """相关分析用的电生理值：在训练细胞上按分位数去掉两端（含分位点保留），符号统一为正方向，负值截为 0。"""
    out = {}
    for name, v in raw.items():
        v = np.asarray(v, float)[tr].copy()
        lo, hi = np.nanquantile(v, q[0]), np.nanquantile(v, q[1])
        v[(v < lo) | (v > hi)] = np.nan
        if np.nanmean(v) < 0:
            v = -v
        out[name] = np.clip(v, 0, None)
    return out


def knn_targets(raw: np.ndarray, tr: np.ndarray, q_clip: float) -> tuple[np.ndarray, float]:
    """kNN 参照细胞的目标值：符号统一为正、负值截为 0、截到训练细胞第 q_clip 分位。返回 (值, 符号)。"""
    v = np.asarray(raw, float)[tr].copy()
    sign = -1.0 if np.nanmean(v) < 0 else 1.0
    v = np.clip(sign * v, 0, None)
    return np.minimum(v, np.nanquantile(v, q_clip)), sign


# ---------------------------------------------------------------- 重抽样 Spearman（GPU）

class BootSpearman:
    """按细胞有放回重抽样的 Spearman 相关。重抽样样本 = 每个细胞的出现次数 w；
    重抽样数据的平均秩（并列取平均）可由按原值排序后的累计权重直接得到，再计算加权 Pearson，
    与对重抽样数据显式排序后计算的结果完全相同，但无需每次重新排序。"""

    def __init__(self, x: np.ndarray, device: str):
        self.dev = torch.device(device)
        xt = torch.as_tensor(x, dtype=torch.float64, device=self.dev)        # 细胞 × 基因
        self.order = torch.argsort(xt, dim=0, stable=True)
        xs = torch.gather(xt, 0, self.order)
        n = xs.shape[0]
        new = torch.ones_like(xs, dtype=torch.bool)
        new[1:] = xs[1:] != xs[:-1]                                           # 并列组起点
        pos = torch.arange(n, device=self.dev).unsqueeze(1).expand_as(xs)
        self.start = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), 0).values
        last = torch.ones_like(new)
        last[:-1] = new[1:]
        rev = torch.flip(torch.where(last, pos, torch.full_like(pos, n)), [0])
        self.end = torch.flip(torch.cummin(rev, 0).values, [0])

    @staticmethod
    def _ranks_1d(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        o = torch.argsort(v, stable=True)
        vs, ws = v[o], w[o]
        c = torch.cumsum(ws, 0)
        n = len(v)
        new = torch.ones(n, dtype=torch.bool, device=v.device)
        new[1:] = vs[1:] != vs[:-1]
        pos = torch.arange(n, device=v.device)
        st = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), 0).values
        last = torch.ones_like(new)
        last[:-1] = new[1:]
        en = torch.flip(torch.cummin(torch.flip(torch.where(last, pos, torch.full_like(pos, n)), [0]), 0).values, [0])
        prev = torch.where(st > 0, c[(st - 1).clamp(min=0)], torch.zeros_like(c))
        r = torch.empty_like(c)
        r[o] = (prev + c[en]) / 2 + 0.5
        return r

    def zscores(self, y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
        """y：该参数的训练细胞值（NaN = 缺测或被分位数过滤）。返回每个基因的 z = 均值/标准差。"""
        ok = torch.as_tensor(~np.isnan(y), device=self.dev)
        yt = torch.as_tensor(np.nan_to_num(y), dtype=torch.float64, device=self.dev)
        n = len(y)
        g = torch.Generator(device=self.dev).manual_seed(int(seed))
        rs = []
        idx_ok = torch.where(ok)[0]
        for _ in range(n_boot):
            pick = idx_ok[torch.randint(len(idx_ok), (len(idx_ok),), generator=g, device=self.dev)]
            w = torch.bincount(pick, minlength=n).to(torch.float64)
            ry = self._ranks_1d(yt, w)
            ws = w[self.order]                                                # 按每个基因的排序取权重
            c = torch.cumsum(ws, 0)
            prev = torch.where(self.start > 0, torch.gather(c, 0, (self.start - 1).clamp(min=0)), torch.zeros_like(c))
            rx = (prev + torch.gather(c, 0, self.end)) / 2 + 0.5
            ryo = ry[self.order]
            sw = w.sum()
            mx = (ws * rx).sum(0) / sw
            my = (w * ry).sum() / sw
            dx, dy = rx - mx, ryo - my
            cov = (ws * dx * dy).sum(0)
            r = cov / torch.sqrt((ws * dx * dx).sum(0) * (w * (ry - my) ** 2).sum())
            rs.append(r)
        r = torch.stack(rs)                                                   # n_boot × 基因
        mu = torch.nanmean(r, 0)
        sd = torch.sqrt(torch.nanmean((r - mu) ** 2, 0))                      # numpy 默认 ddof=0
        return (mu / sd).cpu().numpy()


# ---------------------------------------------------------------- PS 基因集

def select_ps(z: dict, groups: dict, z_th: float, n_blocks: int) -> np.ndarray:
    """z：参数名 → 每基因 z 值。返回在至少 n_blocks 个功能组中有 |z|>z_th 参数的基因（布尔）。"""
    nblk = 0
    for pars in groups.values():
        hit = np.zeros_like(next(iter(z.values())), dtype=bool)
        for p in pars:
            hit |= np.nan_to_num(np.abs(z[p])) > z_th
        nblk = nblk + hit.astype(int)
    return nblk >= n_blocks


# ---------------------------------------------------------------- kNN 预测

def _rank_rows(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    r = rankdata(a, axis=1)
    r = r - r.mean(1, keepdims=True)
    return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)


def knn_neighbours(e_tr: np.ndarray, e_te: np.ndarray, k: int) -> np.ndarray:
    """测试细胞与参照细胞在 PS 基因上的 Spearman 相关，取相关最高的 k 个（距离 1 − r 最小）。"""
    sim = _rank_rows(e_te) @ _rank_rows(e_tr).T
    return np.argsort(-sim, axis=1, kind="stable")


def knn_predict(order: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """前 k 个近邻实测值的等权平均（缺测屏蔽）；k 个近邻全部缺测时顺延到最近的 k 个有测值的参照细胞。"""
    top = target[order[:, :k]]
    cnt = (~np.isnan(top)).sum(1)
    p = np.where(cnt > 0, np.nansum(top, 1) / np.maximum(cnt, 1), np.nan)
    for i in np.where(cnt == 0)[0]:
        vals = target[order[i]]
        p[i] = vals[~np.isnan(vals)][:k].mean()
    return p


def to_model_scale(p_raw: np.ndarray, sign: float, how: str) -> np.ndarray:
    """把原始单位的预测（正方向）还原符号后按本研究的终点变换，以便与其他模型在同一尺度评价。"""
    return transform_endpoint(sign * p_raw, how)
