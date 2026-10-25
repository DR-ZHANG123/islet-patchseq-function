"""scVI 潜空间（scvi-tools）：只在训练折的训练细胞原始计数上训练，测试细胞用训练好的编码器嵌入。

- 基因：训练折内按 features.fit_scaler 同一规则选出的高变基因（与其他方法一致，只用训练细胞）。
- 批次：GEO series（cells.tsv 的 geo 列），绝不用供体——供体层面的差异正是要预测的信号。
- scVI 默认编码器不输入批次（encode_covariates=False），潜变量只依赖细胞自身计数；测试细胞若属于训练细胞中
  未出现的 GEO series，仅为满足注册要求映射到训练中的第一个类别，不影响潜变量。
- 潜变量取后验均值（get_latent_representation 默认 give_mean=True）。
本模块只在 islet_patchseq_scvi 环境中导入（environment_scvi.yml）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _quiet():
    for name in ("scvi", "lightning", "lightning.pytorch", "pytorch_lightning", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)


def to_int_counts(counts: sp.csr_matrix) -> sp.csr_matrix:
    """计数矩阵按 float32 存储；四舍五入为整数（NB 似然要求非负整数）。"""
    c = counts.copy().astype(np.float32)
    c.data = np.rint(c.data)
    return c


def fit_latent(counts: sp.csr_matrix, genes: np.ndarray, batch: np.ndarray, tr: np.ndarray, scfg: dict,
               seed: int, device: int) -> tuple[np.ndarray, dict]:
    """在 tr 细胞上训练 scVI，返回全部细胞的潜变量 (n, n_latent) 与训练摘要。"""
    import anndata as ad
    import scvi
    import torch
    _quiet()
    scvi.settings.seed = seed
    torch.set_num_threads(2)
    x = to_int_counts(counts[:, genes])
    cats = sorted(pd.unique(batch[tr]))
    a_tr = ad.AnnData(x[tr].tocsr())
    a_tr.obs["batch"] = pd.Categorical(batch[tr], categories=cats)
    scvi.model.SCVI.setup_anndata(a_tr, batch_key="batch")
    model = scvi.model.SCVI(a_tr, n_latent=scfg["n_latent"], n_hidden=scfg["n_hidden"], n_layers=scfg["n_layers"],
                            dropout_rate=scfg["dropout_rate"], gene_likelihood=scfg["gene_likelihood"])
    model.train(max_epochs=scfg["max_epochs"], accelerator="gpu", devices=[device], batch_size=scfg["batch_size"],
                train_size=scfg["train_size"], early_stopping=True, early_stopping_patience=scfg["patience"],
                plan_kwargs={"lr": scfg["lr"]}, datasplitter_kwargs={"drop_last": True},
                enable_progress_bar=False, enable_model_summary=False,
                enable_checkpointing=False)
    b_all = np.where(np.isin(batch, cats), batch, cats[0])
    a_all = ad.AnnData(x.tocsr())
    a_all.obs["batch"] = pd.Categorical(b_all, categories=cats)
    lat = model.get_latent_representation(adata=a_all, batch_size=512)
    hist = model.history
    n_ep = int(len(hist["elbo_train"])) if "elbo_train" in hist else -1
    val = float(hist["elbo_validation"].iloc[-1, 0]) if "elbo_validation" in hist else float("nan")
    return lat.astype(np.float32), {"epochs": n_ep, "elbo_validation": val, "batches": "|".join(map(str, cats))}
