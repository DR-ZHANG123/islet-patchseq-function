"""Stage 03d：单细胞基础模型的冻结细胞嵌入（scGPT 全人类、Geneformer V2、UCE）。

- 对全部质控通过的 α/β 细胞（开发供体与 HPAP 外部供体）从原始整数计数一次性计算；无监督，不读取任何电生理终点。
- 每个模型按其官方预处理（不用本研究的高变基因）；基因为六个计数矩阵共有的基因，经 HGNC 映射到各模型词表。
- 每个模型完成后立即写 data/interim/03d_embed/<model>.npz（emb, cell_id；先写临时文件再改名），
  并写同名 .json 记录；全部模型都有记录后汇总 results/03d_embed/embedding_summary.tsv 与 PROVENANCE.json。
环境：islet_patchseq_fm（environment_fm.yml）。用法：
  python scripts/stages/03d_embed.py [--models scgpt_human,uce_4layer] [--device cuda:0] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, md5, set_global_seed, write_provenance  # noqa: E402
from fm_data import (ensure_uce_files, fetch_gdrive_folder, fetch_hf_files, fetch_url,  # noqa: E402
                     load_cellset)
from fm_sanity import pca_reference, sanity  # noqa: E402

STAGE = "03d_embed"
GF_FILES = ["geneformer/token_dictionary_gc104M.pkl", "geneformer/gene_median_dictionary_gc104M.pkl",
            "geneformer/gene_name_id_dict_gc104M.pkl", "geneformer/ensembl_mapping_dict_gc104M.pkl"]


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name("." + path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def checkpoint_files(name: str, fcfg: dict, mdir: Path) -> list[Path]:
    """确保权重就位（官方来源下载），返回需记录 MD5 的权重文件。"""
    if name == "scgpt_human":
        d = mdir / "scgpt_human"
        fetch_gdrive_folder(fcfg["scgpt"]["gdrive_folder"], d, ["args.json", "best_model.pt", "vocab.json"])
        return [d / "best_model.pt", d / "vocab.json", d / "args.json"]
    if name.startswith("geneformer"):
        g = fcfg["geneformer"]
        var = g["variants"][name]
        fetch_hf_files(g["hf_repo"], g["hf_revision"], GF_FILES + [f"{var}/config.json", f"{var}/model.safetensors"],
                       mdir / "geneformer")
        return [mdir / "geneformer" / var / "model.safetensors", mdir / "geneformer" / GF_FILES[0]]
    if name.startswith("uce"):
        ensure_uce_files(fcfg["uce"], mdir / "UCE", [name])
        return [mdir / "UCE/model_files" / fcfg["uce"]["variants"][name]["file"],
                mdir / "UCE/model_files/all_tokens.torch"]
    raise ValueError(name)


def run_model(name: str, cs, cfg: dict, device: str, out_dir: Path) -> dict:
    from fm_models import embed_geneformer, embed_scgpt, embed_uce
    fcfg, mdir = cfg["foundation"], ROOT / cfg["foundation"]["model_dir"]
    ckpts = checkpoint_files(name, fcfg, mdir)
    set_global_seed(cfg["seed"])
    t0 = time.time()
    if name == "scgpt_human":
        emb, info = embed_scgpt(cs, mdir / "scgpt_human", fcfg["scgpt"], device)
    elif name.startswith("geneformer"):
        emb, info = embed_geneformer(cs, mdir / "geneformer", fcfg["geneformer"]["variants"][name],
                                     fcfg["geneformer"], out_dir / "_work" / name, cfg["seed"])
    else:
        import shutil
        shutil.rmtree(out_dir / "_work" / name, ignore_errors=True)
        emb, info = embed_uce(cs, mdir / "UCE", fcfg["uce"], name, out_dir / "_work" / name)
    runtime = time.time() - t0
    if emb.shape[0] != len(cs.cells) or not np.isfinite(emb).all():
        raise RuntimeError(f"{name}: 嵌入形状 {emb.shape} 或含非有限值")
    atomic_npz(out_dir / f"{name}.npz", emb=emb, cell_id=cs.cells.cell_id.values.astype(str))
    rec = {"model": name, "status": "ok", "checkpoint": str(ckpts[0].relative_to(ROOT / "data")),
           "checkpoint_md5": md5(ckpts[0]), "n_cells": int(emb.shape[0]), "dim": int(emb.shape[1]),
           "n_genes_input": int(cs.counts.shape[1]), "runtime_s": round(runtime, 1), **info}
    rec.update(sanity(emb, cs.cells, cfg["foundation"]["sanity"]))
    return rec


def assemble(cfg: dict, cs, out_dir: Path) -> None:
    """全部模型都有记录（成功或失败）后写汇总表与 PROVENANCE。"""
    names = cfg["foundation"]["models"]
    recs = [out_dir / f"{n}.json" for n in names]
    if not all(p.exists() for p in recs):
        print("  尚有模型未运行，暂不汇总")
        return
    rows = [json.loads(p.read_text()) for p in recs]
    set_global_seed(cfg["seed"])
    pca = pca_reference(cs.counts, cfg["foundation"]["sanity"]["pca_dims"], cfg["seed"])
    ref = {"model": "reference_pca_lognorm", "status": "reference", "checkpoint": "none",
           "n_cells": len(cs.cells), "dim": pca.shape[1], "n_genes_input": int(cs.counts.shape[1]),
           "n_genes_vocab": int(cs.counts.shape[1]), "setting": "PCA of log-normalised counts (not exported)"}
    ref.update(sanity(pca, cs.cells, cfg["foundation"]["sanity"]))
    tab = pd.DataFrame(rows + [ref])
    res = ROOT / "results" / STAGE
    res.mkdir(parents=True, exist_ok=True)
    tab.to_csv(res / "embedding_summary.tsv", sep="\t", index=False)
    print(tab.drop(columns=["setting", "error"], errors="ignore").to_string())
    ok = [n for n, r in zip(names, rows) if r["status"] == "ok"]
    write_provenance(STAGE,
                     inputs=[ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/genes.tsv",
                             ROOT / cfg["paths"]["processed"] / "counts.npz", ROOT / cfg["foundation"]["hgnc_file"]],
                     outputs=[res / "embedding_summary.tsv"] + [out_dir / f"{n}.npz" for n in ok],
                     seed=cfg["seed"], extra={"models_ok": ok,
                                              "models_failed": [n for n in names if n not in ok]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="逗号分隔；缺省为 config 中全部模型")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    fcfg = cfg["foundation"]
    fetch_url(fcfg["hgnc_url"], ROOT / fcfg["hgnc_file"])
    out_dir = ROOT / fcfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = load_cellset(cfg)
    print(f"细胞 {cs.counts.shape[0]}，共有基因 {cs.counts.shape[1]}")
    cs.genes.to_csv(out_dir / "gene_mapping.tsv", sep="\t", index=False)
    names = a.models.split(",") if a.models else fcfg["models"]
    for name in names:
        side = out_dir / f"{name}.json"
        if not a.force and (out_dir / f"{name}.npz").exists() and side.exists():
            print(f"[done] {name}")
            continue
        print(f"[run ] {name}")
        try:
            rec = run_model(name, cs, cfg, a.device, out_dir)
        except Exception as e:  # 记录失败原因，继续其余模型
            traceback.print_exc()
            rec = {"model": name, "status": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
        side.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(json.dumps(rec, ensure_ascii=False))
    assemble(cfg, cs, out_dir)


if __name__ == "__main__":
    main()
