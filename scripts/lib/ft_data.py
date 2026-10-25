"""Stage 03f 的模型输入：按各基础模型官方预处理准备逐细胞输入，与 Stage 03d 的冻结嵌入完全相同。

- Geneformer V2：直接读取 Stage 03d 用官方 TranscriptomeTokenizer 生成的秩编码（<cls> … <eos>，≤4096），
  104M 与 316M 共用同一词表与编码。
- scGPT 全人类：六个计数矩阵共有基因经 HGNC 映射到 scGPT 词表（同名合并），保留原始整数计数；
  批处理时用 scGPT 官方 DataCollator 逐细胞分箱（51 箱）并随机抽取 ≤max_length 个非零基因（含 <cls>）。
这里不读取任何电生理终点；终点与供体信息来自 data.load_type。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from fm_data import collapse_to_tokens, load_cellset, match_symbols
from repro import ROOT


@dataclass
class ModelInputs:
    kind: str                     # geneformer | scgpt
    cell_ids: np.ndarray          # 与 rows 对应的细胞编号
    rows: object                  # geneformer：list[np.ndarray]；scgpt：csr 计数矩阵
    gene_ids: np.ndarray = None   # scgpt：列 → 词表编号
    pad_id: int = 0
    extra: dict = None

    def index(self, cell_ids: np.ndarray) -> np.ndarray:
        pos = {c: i for i, c in enumerate(self.cell_ids)}
        miss = [c for c in cell_ids if c not in pos]
        if miss:
            raise ValueError(f"{len(miss)} 个细胞没有模型输入，例如 {miss[:3]}")
        return np.array([pos[c] for c in cell_ids])


def geneformer_inputs(cfg: dict) -> ModelInputs:
    """读取 Stage 03d 的 Geneformer V2 编码（官方 tokenizer 输出）。"""
    import datasets
    import pickle
    from geneformer import TOKEN_DICTIONARY_FILE
    tok_dir = ROOT / cfg["finetune"]["geneformer"]["token_dataset"]
    ds = datasets.load_from_disk(str(tok_dir))
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        tdict = pickle.load(f)
    rows = [np.asarray(r, dtype=np.int64) for r in ds["input_ids"]]
    if any(r[0] != tdict["<cls>"] for r in rows):
        raise RuntimeError("Geneformer 编码首位不是 <cls>")
    return ModelInputs("geneformer", np.asarray(ds["cell_id"]).astype(str), rows, pad_id=int(tdict["<pad>"]),
                       extra={"cls_id": int(tdict["<cls>"]), "eos_id": int(tdict["<eos>"]),
                              "token_dir": str(tok_dir)})


def scgpt_inputs(cfg: dict, model_dir: Path) -> ModelInputs:
    """与 Stage 03d（fm_models.embed_scgpt）相同的词表匹配与基因合并，保留原始计数。"""
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    cs = load_cellset(cfg)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    x, genes = collapse_to_tokens(cs.counts, match_symbols(cs.genes, set(vocab.get_stoi().keys())))
    gene_ids = np.array(vocab(list(genes)), dtype=np.int64)
    args = json.loads((model_dir / "args.json").read_text())
    return ModelInputs("scgpt", cs.cells.cell_id.values.astype(str), sp.csr_matrix(x), gene_ids,
                       pad_id=int(vocab[args["pad_token"]]),
                       extra={"vocab": vocab, "args": args, "cls_id": int(vocab["<cls>"]), "n_genes": len(genes)})


# ---------------------------------------------------------------- 数据集与批处理

class CellDataset(Dataset):
    """idx：ModelInputs 行号；y：标准化终点（NaN = 缺测）；g1：葡萄糖独热。"""

    def __init__(self, mi: ModelInputs, idx: np.ndarray, y: np.ndarray, g1: np.ndarray):
        self.mi, self.idx, self.y, self.g1 = mi, idx, y.astype(np.float32), g1.astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        out = {"y": torch.from_numpy(self.y[i]), "g1": torch.from_numpy(self.g1[i]), "pos": i}
        if self.mi.kind == "geneformer":
            out["ids"] = torch.from_numpy(self.mi.rows[r])
        else:
            row = self.mi.rows[r]
            genes = np.insert(self.mi.gene_ids[row.indices], 0, self.mi.extra["cls_id"])
            vals = np.insert(row.data.astype(np.float32), 0, self.mi.extra["args"]["pad_value"])
            out["genes"], out["expressions"] = torch.from_numpy(genes).long(), torch.from_numpy(vals).float()
        return out


class Collate:
    """Geneformer：右侧填充 <pad>；scGPT：官方 DataCollator（分箱 + 抽样 + 填充，不做掩码）。"""

    def __init__(self, mi: ModelInputs, max_length: int | None = None):
        self.mi = mi
        if mi.kind == "scgpt":
            from scgpt.data_collator import DataCollator
            a = mi.extra["args"]
            self.dc = DataCollator(do_padding=True, pad_token_id=mi.pad_id, pad_value=a["pad_value"], do_mlm=False,
                                   do_binning=True, max_length=max_length, sampling=True, keep_first_n_tokens=1)

    def __call__(self, batch):
        out = {k: torch.stack([b[k] for b in batch]) for k in ("y", "g1")}
        out["pos"] = torch.tensor([b["pos"] for b in batch])
        if self.mi.kind == "geneformer":
            L = max(len(b["ids"]) for b in batch)
            ids = torch.full((len(batch), L), self.mi.pad_id, dtype=torch.long)
            for i, b in enumerate(batch):
                ids[i, :len(b["ids"])] = b["ids"]
            out["ids"], out["mask"] = ids, (ids != self.mi.pad_id).long()
        else:
            d = self.dc([{"id": b["pos"], "genes": b["genes"], "expressions": b["expressions"]} for b in batch])
            out["genes"], out["expr"] = d["gene"], d["expr"]
        return out


def make_loader(mi, idx, y, g1, batch, shuffle, seed, workers, max_length=None):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(CellDataset(mi, idx, y, g1), batch_size=batch, shuffle=shuffle, generator=gen,
                      collate_fn=Collate(mi, max_length), num_workers=workers, drop_last=False,
                      worker_init_fn=_worker_seed if workers else None, persistent_workers=False)


def _worker_seed(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)                     # scGPT 抽样与分箱平局用 numpy 随机数
