"""Stage 03f 的回归模型：预训练基础模型骨干 + 回归头（9 个标准化终点联合预测，葡萄糖浓度作为额外输入）。

微调方式（两种模型都按各自官方微调实现的冻结机制）：
- Geneformer V2：官方 Classifier 的 freeze_layers 机制——冻结嵌入层与底部 n_freeze 个 Transformer 层，
  微调其余顶层；细胞表示 = 顶层非特殊基因 token 的平均（与 Stage 03d 冻结嵌入的池化方式相同）。
- scGPT：官方微调教程冻结输入编码器（基因 token 与表达值编码器），在此基础上同样冻结底部 n_freeze 层，
  只微调顶层；细胞表示 = <cls> 位置输出（与 Stage 03d 相同）。
冻结部分在 eval 模式、无梯度下前向（不计算其激活的梯度），节省显存与时间。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class Head(nn.Module):
    """LayerNorm(细胞表示) ⊕ 葡萄糖独热 → MLP → 9 个终点。"""

    def __init__(self, d: int, n_out: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d + 3, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, h, g1):
        return self.mlp(torch.cat([self.norm(h.float()), g1], 1))


class GeneformerRegressor(nn.Module):
    def __init__(self, model_dir: Path, n_freeze: int, n_out: int, hidden: int, dropout: float):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(str(model_dir), add_pooling_layer=False, attn_implementation="sdpa")
        self.n_layers = len(self.bert.encoder.layer)
        self.n_freeze = n_freeze
        self.head = Head(self.bert.config.hidden_size, n_out, hidden, dropout)
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False
        self.special = None

    def set_special(self, ids: list[int]):
        self.special = torch.tensor(ids)

    def frozen_eval(self):
        self.bert.embeddings.eval()
        for layer in self.bert.encoder.layer[:self.n_freeze]:
            layer.eval()

    def forward(self, batch):
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
        ids, mask = batch["ids"], batch["mask"]
        dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        ext = _prepare_4d_attention_mask_for_sdpa(mask, dtype, tgt_len=ids.shape[1])
        with torch.no_grad():
            h = self.bert.embeddings(input_ids=ids)
            for layer in self.bert.encoder.layer[:self.n_freeze]:
                h = layer(h, attention_mask=ext)[0]
        h = h.detach()
        for layer in self.bert.encoder.layer[self.n_freeze:]:
            h = layer(h, attention_mask=ext)[0]
        keep = mask.bool() & ~torch.isin(ids, self.special.to(ids.device))
        w = keep.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return self.head(pooled, batch["g1"])


class ScGPTRegressor(nn.Module):
    def __init__(self, model_dir: Path, vocab, n_freeze: int, n_out: int, hidden: int, dropout: float):
        super().__init__()
        import torchtext
        torchtext.disable_torchtext_deprecation_warning()
        from scgpt.model import TransformerModel
        cfg = json.loads((model_dir / "args.json").read_text())
        m = TransformerModel(
            ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"], d_hid=cfg["d_hid"],
            nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
            dropout=cfg["dropout"], pad_token=cfg["pad_token"], pad_value=cfg["pad_value"],
            do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
            explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False)
        # 与 Stage 03d 相同：flash-attn 的 Wqkv 命名改为 PyTorch 原生注意力命名后严格核对
        state = torch.load(model_dir / "best_model.pt", map_location="cpu")
        state = {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}
        own = m.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing = sorted(k for k in set(own) - set(loaded) if not k.startswith("cls_decoder."))
        if missing:
            raise RuntimeError(f"scGPT 权重未覆盖的参数: {missing[:10]}")
        m.load_state_dict(loaded, strict=False)
        # 只保留编码部分；解码器、MVC 与分类头不参与回归
        self.encoder, self.value_encoder = m.encoder, m.value_encoder
        self.layers = m.transformer_encoder.layers
        self.n_layers, self.n_freeze, self.pad_id = len(self.layers), n_freeze, vocab[cfg["pad_token"]]
        self.head = Head(cfg["embsize"], n_out, hidden, dropout)
        for mod in [self.encoder, self.value_encoder, *self.layers[:n_freeze]]:
            for p in mod.parameters():
                p.requires_grad = False

    def frozen_eval(self):
        for mod in [self.encoder, self.value_encoder, *self.layers[:self.n_freeze]]:
            mod.eval()

    def forward(self, batch):
        src, val = batch["genes"], batch["expr"]
        pad = src.eq(self.pad_id)
        with torch.no_grad():
            h = self.encoder(src) + self.value_encoder(val)
            for layer in self.layers[:self.n_freeze]:
                h = layer(h, src_key_padding_mask=pad)
        h = h.detach()
        for layer in self.layers[self.n_freeze:]:
            h = layer(h, src_key_padding_mask=pad)
        return self.head(h[:, 0, :], batch["g1"])


def build(name: str, cfg: dict, mi, n_out: int) -> nn.Module:
    """按 config finetune.models.<name> 构建模型；n_freeze = round(层数 × freeze_fraction)。"""
    fcfg = cfg["finetune"]
    mc = fcfg["models"][name]
    root = Path(__file__).resolve().parents[2] / cfg["foundation"]["model_dir"]
    if mc["family"] == "geneformer":
        d = root / "geneformer" / cfg["foundation"]["geneformer"]["variants"][name]
        n_layers = json.loads((d / "config.json").read_text())["num_hidden_layers"]
        m = GeneformerRegressor(d, int(round(n_layers * fcfg["freeze_fraction"])), n_out, fcfg["head_hidden"],
                                fcfg["head_dropout"])
        m.set_special([mi.pad_id, mi.extra["cls_id"], mi.extra["eos_id"]])
        return m
    d = root / "scgpt_human"
    n_layers = mi.extra["args"]["nlayers"]
    return ScGPTRegressor(d, mi.extra["vocab"], int(round(n_layers * fcfg["freeze_fraction"])), n_out,
                          fcfg["head_hidden"], fcfg["head_dropout"])


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, wd: float) -> list:
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    return [{"params": body, "lr": lr_backbone, "weight_decay": wd},
            {"params": head, "lr": lr_head, "weight_decay": wd}]


def n_params(model: nn.Module) -> tuple[int, int]:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr
