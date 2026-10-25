"""嵌入的基本检查（不用任何电生理终点）：
- 细胞类型可分性：标准化后 L2 逻辑回归区分 α/β，按供体分组交叉验证的平衡准确率；
- 供体混合度：kNN（余弦）邻居中来自同一供体的比例，与随机期望（同类型内按供体细胞数）之比；
- 采集中心混合度：kNN 邻居中来自同一中心（site）的比例。
另给出 log 归一化表达的前 pca_dims 个主成分作为参照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def celltype_accuracy(emb: np.ndarray, cells: pd.DataFrame, folds: int) -> float:
    y = (cells.cell_type.values == "Beta").astype(int)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(emb, y, cells.donor.values):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(emb[tr], y[tr])
        pred[te] = clf.predict(emb[te])
    return float(balanced_accuracy_score(y, pred))


def knn_mixing(emb: np.ndarray, cells: pd.DataFrame, k: int) -> dict:
    """在每个细胞类型内计算：同供体邻居比例、其随机期望、同中心邻居比例。"""
    out = {"same_donor_frac": [], "same_donor_expected": [], "same_site_frac": [], "n": []}
    for ct in sorted(cells.cell_type.unique()):
        m = (cells.cell_type == ct).values
        e = emb[m] / np.linalg.norm(emb[m], axis=1, keepdims=True).clip(1e-12)
        donor, site = cells.donor.values[m], cells.site.values[m]
        nn = NearestNeighbors(n_neighbors=k + 1).fit(e)
        idx = nn.kneighbors(e, return_distance=False)[:, 1:]
        sizes = pd.Series(donor).map(pd.Series(donor).value_counts()).values
        out["same_donor_frac"].append((donor[idx] == donor[:, None]).mean())
        out["same_donor_expected"].append(((sizes - 1) / (len(donor) - 1)).mean())
        out["same_site_frac"].append((site[idx] == site[:, None]).mean())
        out["n"].append(m.sum())
    w = np.array(out["n"], float) / sum(out["n"])
    sd = float(np.dot(w, out["same_donor_frac"]))
    ex = float(np.dot(w, out["same_donor_expected"]))
    return {"knn_same_donor_frac": sd, "knn_same_donor_expected": ex,
            "knn_donor_enrichment": sd / ex, "knn_same_site_frac": float(np.dot(w, out["same_site_frac"]))}


def sanity(emb: np.ndarray, cells: pd.DataFrame, scfg: dict) -> dict:
    rec = {"celltype_balanced_acc": celltype_accuracy(emb, cells, scfg["cv_folds"])}
    rec.update(knn_mixing(emb, cells, scfg["knn_k"]))
    return rec


def pca_reference(counts: sp.csr_matrix, dims: int, seed: int) -> np.ndarray:
    x = counts.toarray().astype(np.float64)
    x = np.log1p(x / x.sum(1, keepdims=True).clip(1) * 1e4)
    return PCA(dims, random_state=seed).fit_transform(x).astype(np.float32)
