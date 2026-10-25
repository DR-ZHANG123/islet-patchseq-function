"""神经网络对照的超参数调优版本：网络结构与损失同 nets.py（Net、多终点联合训练、可选供体内差分损失），
宽度、权重衰减、dropout 在训练供体内部按供体分组的内层交叉验证中从 config sota.mlp_grid 选择。

选择标准：内层验证细胞上各终点（按外层训练细胞标准化）观测项均方误差的平均（多输出网络，所有终点共用一组超参数）。
内层每个候选只用 1 个随机种子；选定后在全部训练细胞上按 nets.seeds 个种子训练并取平均（与 Stage 03b 一致）。
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from nets import Net, _codes


def train_predict_hp(z, g1, ymat, donor, group, tr, te, mode: str, lam: float, seed: int, device: str,
                     hidden: int, weight_decay: float, dropout: float, max_epochs: int = 300) -> np.ndarray:
    """与 nets.train_predict 相同的训练流程（训练供体中留出约 1/8 供体做早停），只是宽度、权重衰减、dropout 可调。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = np.nanmean(ymat[tr], 0)
    sd = np.nanstd(ymat[tr], 0) + 1e-8
    ys = (ymat - mean) / sd
    dons = np.unique(donor[tr])
    val_d = rng.choice(dons, max(3, len(dons) // 8), replace=False)
    itr = tr[~np.isin(donor[tr], val_d)]
    iva = tr[np.isin(donor[tr], val_d)]
    dev = torch.device(device)
    xt = torch.tensor(z, dtype=torch.float32, device=dev)
    gt = torch.tensor(g1, dtype=torch.float32, device=dev)
    yt = torch.tensor(ys, dtype=torch.float32, device=dev)
    net = Net(z.shape[1], ymat.shape[1], mode, hidden=hidden, drop_in=dropout, drop=dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=weight_decay)

    def loss_on(idx):
        ctx = torch.tensor(_codes(donor[idx]), device=dev, dtype=torch.long)
        out = net(xt[idx], ctx, gt[idx])
        y = yt[idx]
        msk = ~torch.isnan(y)
        e = (out - torch.nan_to_num(y)) * msk
        loss = (e ** 2).sum() / msk.sum()
        if lam > 0:
            gid = torch.tensor(_codes(group[idx]), device=dev, dtype=torch.long)
            ng = int(gid.max()) + 1
            se = torch.zeros(ng, e.shape[1], device=dev).index_add_(0, gid, e)
            cn = torch.zeros(ng, e.shape[1], device=dev).index_add_(0, gid, msk.float())
            ec = (e - (se / cn.clamp(min=1))[gid]) * msk
            loss = loss + lam * (ec ** 2).sum() / msk.sum()
        return loss

    best, state, patience = np.inf, None, 0
    tr_dons = np.unique(donor[itr])
    for epoch in range(max_epochs):
        rng.shuffle(tr_dons)
        for chunk in np.array_split(tr_dons, max(1, len(tr_dons) // 8)):
            idx = itr[np.isin(donor[itr], chunk)]
            opt.zero_grad()
            loss_on(idx).backward()
            opt.step()
        if epoch % 5 == 4:
            net.eval()
            with torch.no_grad():
                v = loss_on(iva).item()
            net.train()
            if v < best - 1e-4:
                best, patience = v, 0
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        ctx = torch.tensor(_codes(donor[te]), device=dev, dtype=torch.long)
        p = net(xt[te], ctx, gt[te]).cpu().numpy()
    return p * sd + mean


def grid(mcfg: dict) -> list:
    return [dict(hidden=h, weight_decay=w, dropout=d)
            for h, w, d in itertools.product(mcfg["hidden"], mcfg["weight_decay"], mcfg["dropout"])]


def tuned_predict(z, g1, ymat, donor, group, tr, te, mode, lam, inner, mcfg, n_seeds, seed, device):
    """inner：相对 tr 的内层（供体分组）折。返回 (测试细胞预测 (len(te), K), 所选超参数)。"""
    sd = np.nanstd(ymat[tr], 0) + 1e-8
    scores = []
    for hp in grid(mcfg):
        err = []
        for itr, iva in inner:
            p = train_predict_hp(z, g1, ymat, donor, group, tr[itr], tr[iva], mode, lam, seed, device, **hp)
            r = (p - ymat[tr[iva]]) / sd
            err.append(np.nanmean(r ** 2, 0))          # 每终点观测项 MSE
        scores.append(float(np.nanmean(np.mean(err, 0))))
    hp = grid(mcfg)[int(np.argmin(scores))]
    pred = np.mean([train_predict_hp(z, g1, ymat, donor, group, tr, te, mode, lam, seed + s, device, **hp)
                    for s in range(n_seeds)], 0)
    return pred, hp
