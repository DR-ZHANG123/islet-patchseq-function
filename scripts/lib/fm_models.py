"""三类单细胞基础模型的冻结嵌入：scGPT（全人类预训练）、Geneformer V2、UCE。

每个函数输入原始整数计数（CellSet），按模型官方实现完成预处理与前向计算，
返回 (emb[n_cells, d], info)。info 记录词表匹配基因数与关键设置。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from fm_data import CellSet, collapse_to_tokens, match_symbols


@contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ---------------------------------------------------------------- scGPT

def embed_scgpt(cs: CellSet, model_dir: Path, scfg: dict, device: str) -> tuple[np.ndarray, dict]:
    """官方 scgpt.tasks.cell_emb 流程：词表匹配 → 原始计数逐细胞分箱(51) → 随机抽取 ≤max_length 个非零基因 → <cls> 嵌入（L2 归一化）。"""
    import torchtext
    torchtext.disable_torchtext_deprecation_warning()
    from scgpt.model import TransformerModel
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    names = match_symbols(cs.genes, set(vocab.get_stoi().keys()))
    x, genes = collapse_to_tokens(cs.counts, names)
    cfg = json.loads((model_dir / "args.json").read_text())
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(list(genes)), dtype=int)
    model = TransformerModel(
        ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"], d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
        dropout=cfg["dropout"], pad_token=cfg["pad_token"], pad_value=cfg["pad_value"],
        do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
        explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False)
    # 预训练权重用 flash-attn 的 Wqkv 命名，改为 PyTorch 原生注意力的 in_proj_ 命名后严格核对
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    state = {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    # 预训练未使用细胞分类头（args.json: no_cls），其参数不参与 _encode；其余参数必须全部来自权重文件
    missing = sorted(k for k in set(own) - set(loaded) if not k.startswith("cls_decoder."))
    if missing:
        raise RuntimeError(f"scGPT 权重未覆盖的参数: {missing[:10]} ...（共 {len(missing)} 个）")
    model.load_state_dict(loaded, strict=False)
    model.to(device).eval()
    a = ad.AnnData(X=x.toarray(), var=pd.DataFrame(index=genes))
    emb = get_batch_cell_embeddings(a, cell_embedding_mode="cls", model=model, vocab=vocab,
                                    max_length=scfg["max_length"], batch_size=scfg["batch_size"],
                                    model_configs=cfg, gene_ids=gene_ids, use_batch_labels=False)
    info = {"n_genes_vocab": int(len(genes)), "n_params_loaded": len(loaded),
            "n_ckpt_params_unused": len(set(state) - set(loaded)),
            "setting": f"cls token; max_length={scfg['max_length']}; n_bins={cfg['n_bins']}"}
    return emb.astype(np.float32), info


# ---------------------------------------------------------------- Geneformer

def embed_geneformer(cs: CellSet, gf_root: Path, variant_dir: str, gcfg: dict, work: Path,
                     seed: int) -> tuple[np.ndarray, dict]:
    """官方 TranscriptomeTokenizer（V2：中位数缩放后的秩编码，<cls>/<eos>，4096）+ EmbExtractor（mean-pool, 倒数第二层）。"""
    from geneformer import EmbExtractor, TranscriptomeTokenizer
    from geneformer import ENSEMBL_MAPPING_FILE, TOKEN_DICTIONARY_FILE

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)
    with open(ENSEMBL_MAPPING_FILE, "rb") as f:
        mapping = pickle.load(f)
    ens = cs.genes.ensembl.values
    ok = np.array([e is not None and e in mapping and mapping[e] in token_dict for e in ens])
    n_counts = np.asarray(cs.counts.sum(1)).ravel()
    h5_dir, tok_dir, emb_dir = work / "h5ad", work / "tokens", work / "emb"
    for d in (h5_dir, tok_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)
    a = ad.AnnData(X=cs.counts[:, ok].tocsr(),
                   obs=pd.DataFrame({"n_counts": n_counts, "cell_id": cs.cells.cell_id.values},
                                    index=cs.cells.cell_id.values),
                   var=pd.DataFrame({"ensembl_id": ens[ok]}, index=ens[ok]))
    a.var_names_make_unique()
    a.write_h5ad(h5_dir / "cells.h5ad")
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id"}, nproc=gcfg["nproc"],
                                model_version="V2", collapse_gene_ids=True)
    tk.tokenize_data(str(h5_dir), str(tok_dir), "cells", file_format="h5ad")
    ex = EmbExtractor(model_type="Pretrained", emb_mode="cell", emb_layer=gcfg["emb_layer"],
                      emb_label=["cell_id"], forward_batch_size=gcfg["batch_size"], max_ncells=None,
                      nproc=gcfg["nproc"], model_version="V2")
    df = ex.extract_embs(str(gf_root / variant_dir), str(tok_dir / "cells.dataset"), str(emb_dir), variant_dir)
    df = df.set_index("cell_id").reindex(cs.cells.cell_id.values)
    if df.isna().any().any():
        raise RuntimeError("Geneformer 输出缺少部分细胞")
    info = {"n_genes_vocab": int(len(set(mapping[e] for e in ens[ok]))),
            "setting": f"mean of non-special gene tokens; hidden layer {gcfg['emb_layer']} (penultimate); input 4096"}
    return df.values.astype(np.float32), info


# ---------------------------------------------------------------- UCE

def uce_vocab(uce_dir: Path) -> set:
    pe = torch.load(uce_dir / "model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
    chrom = pd.read_csv(uce_dir / "model_files/species_chrom.csv")
    chrom = set(chrom.loc[chrom.species == "human", "gene_symbol"].str.upper())
    return {k for k in pe.keys() if k.upper() in chrom}


def embed_uce(cs: CellSet, uce_dir: Path, ucfg: dict, variant: str, work: Path) -> tuple[np.ndarray, dict]:
    """官方 eval_single_anndata 流程（AnndataProcessor）：ESM2 蛋白嵌入作基因 token，按 log 表达加权抽取 1024 个基因，按染色体排序，取 <cls>。"""
    v = ucfg["variants"][variant]
    names = match_symbols(cs.genes, uce_vocab(uce_dir))
    x, genes = collapse_to_tokens(cs.counts, names)
    work.mkdir(parents=True, exist_ok=True)
    a = ad.AnnData(X=x, obs=pd.DataFrame(index=cs.cells.cell_id.values), var=pd.DataFrame(index=genes))
    a.write_h5ad(work / f"{variant}.h5ad")
    args = Namespace(
        adata_path=str(work / f"{variant}.h5ad"), dir=str(work) + "/", species="human",
        filter=True, skip=True, model_loc=str(uce_dir / "model_files" / v["file"]),
        batch_size=v["batch_size"], pad_length=ucfg["pad_length"], pad_token_idx=0,
        chrom_token_left_idx=1, chrom_token_right_idx=2, cls_token_idx=3, CHROM_TOKEN_OFFSET=143574,
        sample_size=ucfg["sample_size"], CXG=True, nlayers=v["nlayers"], output_dim=1280,
        d_hid=5120, token_dim=5120, multi_gpu=False,
        spec_chrom_csv_path="./model_files/species_chrom.csv", token_file="./model_files/all_tokens.torch",
        protein_embeddings_dir="./model_files/protein_embeddings/",
        offset_pkl_path="./model_files/species_offsets.pkl")
    sys.path.insert(0, str(uce_dir))
    with _cwd(uce_dir):
        from accelerate import Accelerator
        from evaluate import AnndataProcessor
        proc = AnndataProcessor(args, Accelerator(project_dir=args.dir))
        proc.preprocess_anndata()
        proc.generate_idxs()
        proc.run_evaluation()
    out = ad.read_h5ad(work / f"{variant}_uce_adata.h5ad")
    if list(out.obs_names) != list(cs.cells.cell_id.values):
        raise RuntimeError("UCE 过滤后细胞顺序或数量与输入不一致")
    info = {"n_genes_vocab": int(out.n_vars),
            "setting": f"{v['nlayers']}-layer; cls token; sample_size={ucfg['sample_size']}"}
    return np.asarray(out.obsm["X_uce"], dtype=np.float32), info
