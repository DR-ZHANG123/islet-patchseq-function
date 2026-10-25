"""Stage 03f 的训练与预测：遮蔽 MSE（9 个终点在训练细胞上标准化，缺测项不计入），早停只用训练供体。

单折流程（开发供体交叉验证与外部检验相同）：
  1) 选择轮数：训练供体按 donor 分组的内层划分，取第一个内层折的供体作为验证供体；
     在其余训练供体上微调，每个 epoch 末计算验证供体的遮蔽 MSE，连续 patience 个 epoch 无改善即停止，
     记录验证误差最低的 epoch 数。
  2) 重拟合：从预训练权重重新开始，在全部训练供体上按同一学习率日程（以 max_epochs 定义的线性预热 + 线性衰减）
     训练至所选 epoch 数，预测测试细胞。
学习率、批量、轮数上限等全部来自 config finetune；测试细胞不参与任何选择。
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ft_data import make_loader
from ft_models import build, n_params, param_groups


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    ok = ~torch.isnan(y)
    n = int(ok.sum())
    if n == 0:
        return pred.sum() * 0.0, 0
    return ((pred.float()[ok] - y[ok]) ** 2).sum() / n, n


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def standardizer(y: np.ndarray, idx: np.ndarray):
    mu = np.nanmean(y[idx], 0)
    sd = np.nanstd(y[idx], 0) + 1e-6
    return mu, sd


@torch.no_grad()
def predict(model, mi, rows, g1, n_out, mcfg, fcfg, seed, device) -> np.ndarray:
    """返回标准化尺度的预测；scGPT 抽样输入时对 eval_draws 次独立抽样的预测取平均。"""
    model.eval()
    draws = mcfg.get("eval_draws", 1)
    out = np.zeros((len(rows), n_out))
    dummy = np.full((len(rows), n_out), np.nan, dtype=np.float32)
    for d in range(draws):
        np.random.seed(seed + 7919 * d)
        dl = make_loader(mi, rows, dummy, g1, mcfg["eval_batch"], False, seed + d, 0, mcfg.get("max_length"))
        for b in dl:
            b = _to(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b)
            out[b["pos"].cpu().numpy()] += p.float().cpu().numpy() / draws
    return out


def train_phase(name, cfg, mi, rows_tr, ys_tr, g1_tr, n_epochs, seed, device, val=None, log=None):
    """微调 n_epochs 个 epoch（学习率日程按 max_epochs 定义）。val=(rows, ys, g1) 时逐 epoch 计算验证误差并早停。
    返回 (model, 每 epoch 记录, 最佳 epoch)。"""
    fcfg = cfg["finetune"]
    mcfg = fcfg["models"][name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(name, cfg, mi, ys_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(param_groups(model, mcfg["lr_backbone"], fcfg["lr_head"], fcfg["weight_decay"]))
    micro, accum = mcfg["micro_batch"], mcfg["batch_size"] // mcfg["micro_batch"]
    n_micro = math.ceil(len(rows_tr) / micro)
    steps_per_epoch = math.ceil(n_micro / accum)
    total = fcfg["max_epochs"] * steps_per_epoch
    warm = max(1, int(fcfg["warmup_fraction"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    params = [p for p in model.parameters() if p.requires_grad]
    rec, best, best_ep, bad = [], np.inf, 0, 0
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        model.frozen_eval()
        dl = make_loader(mi, rows_tr, ys_tr, g1_tr, micro, True, seed * 100 + ep, fcfg["workers"],
                         mcfg.get("max_length"))
        tot_loss, tot_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl):
            b = _to(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b)
            loss, n = masked_mse(p, b["y"])
            (loss / accum).backward()
            tot_loss += float(loss) * n
            tot_n += n
            if (i + 1) % accum == 0 or i + 1 == n_micro:
                torch.nn.utils.clip_grad_norm_(params, fcfg["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        r = {"epoch": ep, "train_loss": tot_loss / max(tot_n, 1), "lr_backbone": opt.param_groups[0]["lr"]}
        if val is not None:
            pv = predict(model, mi, val[0], val[2], val[1].shape[1], mcfg, fcfg, seed, device)
            ok = ~np.isnan(val[1])
            r["val_loss"] = float(((pv[ok] - val[1][ok]) ** 2).mean())
            if r["val_loss"] < best - fcfg["min_delta"]:
                best, best_ep, bad = r["val_loss"], ep, 0
            else:
                bad += 1
        r["seconds"] = time.time() - t0
        rec.append(r)
        if log is not None:
            log(r)
        if val is not None and bad >= fcfg["patience"]:
            break
    return model, rec, (best_ep if val is not None else n_epochs)


def fit_predict(name, cfg, mi, rows, y, g1, donor, tr, te, val_idx, seed, device, log=None):
    """rows/y/g1/donor：按细胞（与 TypeData 同序）；tr/te：训练/测试细胞位置；val_idx：tr 中用作验证的位置子集。
    返回 (原始尺度预测[len(te), K], 训练记录, 信息字典)。"""
    fcfg = cfg["finetune"]
    mcfg = fcfg["models"][name]
    t0 = time.time()
    va = np.asarray(val_idx)
    tin = np.setdiff1d(tr, va)
    mu, sd = standardizer(y, tin)
    ys = (y - mu) / sd
    lg = (lambda r: log({"phase": "select", **r})) if log else None
    m, rec_s, best_ep = train_phase(name, cfg, mi, rows[tin], ys[tin], g1[tin], fcfg["max_epochs"], seed, device,
                                    val=(rows[va], ys[va], g1[va]), log=lg)
    del m
    torch.cuda.empty_cache()
    best_ep = max(best_ep, 1)
    mu, sd = standardizer(y, tr)
    ys = (y - mu) / sd
    lg = (lambda r: log({"phase": "refit", **r})) if log else None
    model, rec_r, _ = train_phase(name, cfg, mi, rows[tr], ys[tr], g1[tr], best_ep, seed, device, log=lg)
    p = predict(model, mi, rows[te], g1[te], y.shape[1], mcfg, fcfg, seed, device)
    tot, trn = n_params(model)
    del model
    torch.cuda.empty_cache()
    info = {"selected_epochs": int(best_ep), "n_train_cells": int(len(tr)), "n_inner_train": int(len(tin)),
            "n_val_cells": int(len(va)), "n_val_donors": int(len(np.unique(donor[va]))),
            "params_total": int(tot), "params_trainable": int(trn), "gpu_seconds": time.time() - t0}
    return p * sd + mu, [{"phase": "select", **r} for r in rec_s] + [{"phase": "refit", **r} for r in rec_r], info
