"""Stage 03d 的数据准备：选取质控通过的 α/β 细胞、取共有基因、基因 ID 映射（HGNC）与权重下载。

基础模型各自的预处理（归一化、排序、分箱、抽样）在 fm_models.py 中按官方实现完成；
这里只提供原始整数计数与基因注释，不做高变基因选择，也不读取任何电生理终点。
"""
from __future__ import annotations

import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from repro import ROOT


@dataclass
class CellSet:
    cells: pd.DataFrame        # cell_id, donor, cell_type, site, glucose, matrix
    counts: sp.csr_matrix      # 原始整数计数（细胞 × 基因）
    genes: pd.DataFrame        # symbol, entrez, hgnc_symbol, ensembl


# ---------------------------------------------------------------- 下载工具

def fetch_url(url: str, dest: Path) -> Path:
    """下载到临时文件再改名，避免留下不完整文件。"""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  下载 {url} → {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def fetch_hf_files(repo: str, revision: str, files: list[str], dest: Path) -> None:
    for f in files:
        fetch_url(f"https://huggingface.co/{repo}/resolve/{revision}/{f}", dest / f)


def fetch_gdrive_folder(url: str, dest: Path, expect: list[str]) -> None:
    if all((dest / f).exists() for f in expect):
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gdown", "--folder", url, "-O", str(dest)], check=True)


def fetch_git(repo: str, commit: str, dest: Path) -> None:
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "-q", repo, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)


def ensure_uce_files(ucfg: dict, uce_dir: Path, variants: list[str]) -> None:
    fetch_git(ucfg["repo"], ucfg["repo_commit"], uce_dir)
    mf = uce_dir / "model_files"
    needed = {"species_chrom.csv", "species_offsets.pkl", "all_tokens.torch", "protein_embeddings.tar.gz"}
    needed |= {ucfg["variants"][v]["file"] for v in variants}
    for name in needed:
        fetch_url(ucfg["figshare"][name], mf / name)
    if not (mf / "protein_embeddings").exists():
        with tarfile.open(mf / "protein_embeddings.tar.gz") as t:
            t.extractall(mf)


# ---------------------------------------------------------------- 细胞与基因

def load_cellset(cfg: dict) -> CellSet:
    """质控通过的 α/β 细胞（开发与外部供体全部纳入；嵌入为无监督，一次计算）。"""
    res = ROOT / "results" / "01_dataset"
    cells = pd.read_csv(res / "cells.tsv", sep="\t")
    genes = pd.read_csv(res / "genes.tsv", sep="\t")
    counts = sp.load_npz(ROOT / cfg["paths"]["processed"] / "counts.npz").tocsr()
    c = cells[cells.qc_pass & cells.cell_type.isin(cfg["dataset"]["cell_types"])].reset_index(drop=True)
    mats = [f"{t}_{g}" for t in cfg["dataset"]["cell_types"] for g in cfg["dataset"]["glucose_levels"]]
    if cfg["foundation"]["gene_set"] == "shared":
        keep = genes[mats].all(1).values & genes.symbol.notna().values
    else:
        keep = genes.symbol.notna().values
    x = counts[c.row.values][:, keep].tocsr()
    x.data = np.rint(x.data)
    g = genes.loc[keep, ["entrez", "symbol"]].reset_index(drop=True)
    g = annotate_hgnc(g, ROOT / cfg["foundation"]["hgnc_file"])
    meta = c[["cell_id", "donor", "cell_type", "site", "glucose", "matrix"]].copy()
    return CellSet(meta, x.astype(np.float32), g)


def annotate_hgnc(g: pd.DataFrame, hgnc_file: Path) -> pd.DataFrame:
    """Entrez → HGNC 现行符号与 Ensembl ID；Entrez 未命中时依次用现行符号、旧符号（唯一）匹配。"""
    h = pd.read_csv(hgnc_file, sep="\t", dtype=str, usecols=["symbol", "prev_symbol", "entrez_id", "ensembl_gene_id"])
    by_entrez = h.dropna(subset=["entrez_id"]).drop_duplicates("entrez_id").set_index("entrez_id")
    by_symbol = h.drop_duplicates("symbol").set_index("symbol")
    prev = h.dropna(subset=["prev_symbol"]).assign(prev=lambda d: d.prev_symbol.str.split("|")).explode("prev")
    prev = prev[~prev.prev.duplicated(keep=False)].set_index("prev")
    out_sym, out_ens, how = [], [], []
    for e, s in zip(g.entrez, g.symbol):
        key = str(int(float(e))) if pd.notna(e) else None
        if key is not None and key in by_entrez.index:
            r, m = by_entrez.loc[key], "entrez"
        elif s in by_symbol.index:
            r, m = by_symbol.loc[s], "symbol"
        elif s in prev.index:
            r, m = prev.loc[s], "previous_symbol"
        else:
            out_sym.append(s); out_ens.append(None); how.append("unmapped")
            continue
        out_sym.append(r["symbol"]); out_ens.append(r["ensembl_gene_id"] if pd.notna(r["ensembl_gene_id"]) else None)
        how.append(m)
    return g.assign(hgnc_symbol=out_sym, ensembl=out_ens, hgnc_match=how)


def collapse_to_tokens(counts: sp.csr_matrix, names: list) -> tuple[sp.csr_matrix, np.ndarray]:
    """把计数列按目标词表名合并（同名求和，None 丢弃）；返回新矩阵与列名。"""
    names = np.asarray(names, dtype=object)
    ok = np.array([n is not None for n in names])
    uniq, inv = np.unique(names[ok].astype(str), return_inverse=True)
    agg = sp.csr_matrix((np.ones(ok.sum()), (np.where(ok)[0], inv)), shape=(len(names), len(uniq)))
    return (counts @ agg).tocsr().astype(np.float32), uniq


def match_symbols(genes: pd.DataFrame, vocab: set) -> list:
    """优先用原符号匹配词表，否则用 HGNC 现行符号；都不在词表中返回 None。"""
    out = []
    for s, h in zip(genes.symbol, genes.hgnc_symbol):
        out.append(s if s in vocab else (h if h in vocab else None))
    return out
