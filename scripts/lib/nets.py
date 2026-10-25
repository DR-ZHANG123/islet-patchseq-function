"""神经网络对照（多终点联合训练）：单细胞 MLP、拼接供体参照的 MLP、非线性双分支网络。

双分支网络：编码 h_i = E(x_i)；供体参照 m_d = 同供体细胞编码的均值（集合均值汇聚）；
ŷ_i = B(m_d, c_i) + R(h_i − m_d, c_i)；可加供体内差分损失（比较组内去均值残差的平方）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, p: int, k: int, mode: str, hidden: int = 128, z: int = 64, drop_in: float = 0.2,
                 drop: float = 0.3):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(nn.Dropout(drop_in), nn.Linear(p, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, z))
        extra = z if mode == "concat" else 0
        self.r = nn.Sequential(nn.Linear(z + extra + 3, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, k))
        if mode == "two":
            self.b = nn.Linear(z + 3, k)

    def forward(self, x, ctx, glu):
        h = self.enc(x)
        if self.mode == "cell":
            return self.r(torch.cat([h, glu], 1))
        n = int(ctx.max()) + 1
        s = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, ctx, h)
        cnt = torch.zeros(n, device=h.device).index_add_(0, ctx, torch.ones(len(ctx), device=h.device))
        m = (s / cnt[:, None])[ctx]
        if self.mode == "concat":
            return self.r(torch.cat([h, m, glu], 1))
        return self.b(torch.cat([m, glu], 1)) + self.r(torch.cat([h - m, glu], 1))


def _codes(a) -> np.ndarray:
    return pd.Series(a).astype("category").cat.codes.values


def train_predict(z, g1, ymat, donor, group, tr, te, mode: str, lam: float, seed: int, device: str,
                  max_epochs: int = 300) -> np.ndarray:
    """ymat：(n, K)，NaN 表示缺测；在训练细胞上逐终点标准化，返回测试细胞原尺度预测。"""
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
    net = Net(z.shape[1], ymat.shape[1], mode).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

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
