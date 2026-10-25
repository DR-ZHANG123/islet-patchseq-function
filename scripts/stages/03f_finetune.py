"""Stage 03f：单细胞基础模型微调（Geneformer V2-104M/316M、scGPT 全人类）预测 9 个电生理终点。

- 每个基础模型加回归头（9 个终点联合，遮蔽 MSE，终点在训练细胞上标准化；葡萄糖浓度作为额外输入），
  冻结嵌入层与底部一半 Transformer 层、微调顶层一半（细节见 scripts/lib/ft_models.py、ft_train.py）。
- 留出供体交叉验证与 Stage 03 完全相同：重复 rep 的种子 = seed + 1000·rep，donor_folds 分折，折种子 = 重复种子 + 37·fold；
  运行 config finetune.repeats 中的重复。早停只用训练供体内部的验证供体；测试折不参与任何选择。
- HPAP 外部供体：开发供体汇总完成后，在全部开发供体上按同一流程训练，只评价一次。
分步运行（--part）：
  cv        开发供体交叉验证（每块 GPU 一个进程；逐折写 data/interim/03f_finetune/folds/，已完成的折跳过）
  summary   汇总三个层面指标与配对差值（参照：Stage 03 cell_branch / context_model；同一模型的冻结嵌入头）
  external  HPAP 外部供体（需 summary 已完成）
环境：islet_patchseq_fm（environment_fm.yml）。用法：
  python scripts/stages/03f_finetune.py --part cv [--models geneformer_v2_104m,scgpt_human]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "03f_finetune"
INT = ROOT / "data" / "interim" / STAGE
FOLDS = INT / "folds"
OUT = ROOT / "results" / STAGE
_STATE: dict = {}


# ---------------- 子进程：每个进程固定一块 GPU，缓存数据 ----------------
def _init(queue):
    gpu = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _STATE["gpu"] = gpu


def _inputs(cfg, family):
    import ft_data
    if family not in _STATE:
        _STATE[family] = (ft_data.geneformer_inputs(cfg) if family == "geneformer"
                          else ft_data.scgpt_inputs(cfg, ROOT / cfg["foundation"]["model_dir"] / "scgpt_human"))
    return _STATE[family]


def _arrays(cfg, ct, external: bool):
    """开发（与外部）细胞的终点、葡萄糖独热、供体与细胞编号；外部检验时开发在前、外部在后。"""
    from data import load_type
    from features import glucose_onehot
    key = ("arr", ct, external)
    if key not in _STATE:
        dev = load_type(cfg, ct, "dev")
        parts = [dev] + ([load_type(cfg, ct, "external")] if external else [])
        cat = lambda f: np.concatenate([f(t) for t in parts])  # noqa: E731
        _STATE[key] = {"y": np.vstack([t.y for t in parts]), "glucose": cat(lambda t: t.glucose),
                       "donor": cat(lambda t: t.donor), "cell_id": cat(lambda t: t.cells.cell_id.values.astype(str)),
                       "n_dev": len(dev.donor), "endpoints": dev.endpoints}
        _STATE[key]["g1"] = glucose_onehot(_STATE[key]["glucose"])
    return _STATE[key]


def fold_path(model, ct, tag):
    return FOLDS / f"{model}_{ct}_{tag}"


def run_job(job: dict) -> dict:
    import torch
    import baselines as bl
    from data import donor_folds
    from ft_train import fit_predict
    cfg = load_config()
    m, ct, rep, fold = job["model"], job["cell_type"], job["repeat"], job["fold"]
    ext = rep == "external"
    a = _arrays(cfg, ct, ext)
    mi = _inputs(cfg, cfg["finetune"]["models"][m]["family"])
    rows = mi.index(a["cell_id"])
    if ext:
        seed = cfg["seed"]
        tr, te = np.arange(a["n_dev"]), np.arange(a["n_dev"], len(a["donor"]))
        tag = "external"
    else:
        rseed = cfg["seed"] + 1000 * rep
        folds = donor_folds(a["donor"], cfg["cv"]["outer_folds"], rseed)
        seed = rseed + 37 * fold
        tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
        tag = f"r{rep}_f{fold}"
    inner = bl.inner_splits(a["donor"][tr], cfg["cv"]["inner_folds"], seed)
    val = tr[inner[0][1]]                            # 第一个内层折的训练供体作为验证供体（早停）
    set_global_seed(seed)
    t0 = time.time()

    def log(r):
        print(f"[gpu{_STATE['gpu']}] {m} {ct} {tag} {r['phase']} ep{r['epoch']} "
              f"train={r['train_loss']:.3f} val={r.get('val_loss', float('nan')):.3f} {r['seconds']:.0f}s", flush=True)

    claim = fold_path(m, ct, tag).with_suffix(".claim")
    if fold_path(m, ct, tag).with_suffix(".parquet").exists():
        return None
    try:                                             # 多个调度进程并行时，同一任务只运行一次
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return None
    pred, rec, info = fit_predict(m, cfg, mi, rows, a["y"], a["g1"], a["donor"], tr, te, val, seed, "cuda:0", log)
    eps = a["endpoints"]
    out = []
    for k, ep in enumerate(eps):
        ok = ~np.isnan(a["y"][te, k])
        d = pd.DataFrame({"cell_id": a["cell_id"][te][ok], "endpoint": ep, "model": f"{m}_finetuned",
                          "y": a["y"][te, k][ok], "pred": pred[ok, k]})
        if not ext:
            d["repeat"], d["fold"] = rep, fold
        out.append(d)
    base = fold_path(m, ct, tag)
    meta = {"model": m, "cell_type": ct, "repeat": str(rep), "fold": fold, "seed": int(seed), "gpu": _STATE["gpu"],
            "device_name": torch.cuda.get_device_name(0), **info, "wall_seconds": time.time() - t0}
    pd.DataFrame(rec).assign(model=m, cell_type=ct, repeat=str(rep), fold=fold).to_csv(
        base.with_suffix(".log.tsv"), sep="\t", index=False)
    pd.concat(out, ignore_index=True).to_parquet(base.with_suffix(".parquet.part"))
    os.replace(base.with_suffix(".parquet.part"), base.with_suffix(".parquet"))
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    claim.unlink(missing_ok=True)
    return meta


def free_gpus(gpus, limit_mb=2000):
    """只用当前显存占用低于 limit_mb 的 GPU（服务器与他人共用）。"""
    import subprocess
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    used = {int(a): int(b) for a, b in (ln.split(",") for ln in o.strip().splitlines())}
    return [g for g in gpus if used.get(g, 1e9) < limit_mb]


def dispatch(cfg, jobs, gpus):
    import multiprocessing as mp
    jobs = [j for j in jobs if not fold_path(j["model"], j["cell_type"], j["tag"]).with_suffix(".parquet").exists()]
    print(f"待运行 {len(jobs)} 个任务")
    if not jobs:
        return
    gpus = free_gpus(gpus)
    if not gpus:
        raise RuntimeError("没有空闲 GPU")
    print(f"使用 GPU {gpus}")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for g in gpus:
        q.put(g)
    with ProcessPoolExecutor(len(gpus), mp_context=ctx, initializer=_init, initargs=(q,)) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                continue
            print(f"[done] {r['model']} {r['cell_type']} r{r['repeat']} f{r['fold']} "
                  f"epochs={r['selected_epochs']} {r['wall_seconds'] / 60:.1f} min", flush=True)


def job_list(cfg, models, external, small_first=False):
    size = {"geneformer_v2_316m": 3, "geneformer_v2_104m": 2, "scgpt_human": 1}
    jobs = []
    for m in models:
        for ct in cfg["dataset"]["cell_types"]:
            if external:
                jobs.append({"model": m, "cell_type": ct, "repeat": "external", "fold": -1, "tag": "external"})
                continue
            for rep in cfg["finetune"]["repeats"]:
                for f in range(cfg["cv"]["outer_folds"]):
                    jobs.append({"model": m, "cell_type": ct, "repeat": rep, "fold": f, "tag": f"r{rep}_f{f}"})
    # 大模型、α 细胞（细胞多）优先，缩短总时间
    jobs = sorted(jobs, key=lambda j: (-size.get(j["model"], 0), j["cell_type"] != "Alpha", str(j["repeat"]), j["fold"]))
    return jobs[::-1] if small_first else jobs


def collect(cfg, models, external):
    preds, logs, info = {}, [], []
    for ct in cfg["dataset"]["cell_types"]:
        parts = []
        for j in job_list(cfg, models, external):
            if j["cell_type"] != ct:
                continue
            b = fold_path(j["model"], ct, j["tag"])
            if not b.with_suffix(".parquet").exists():
                raise RuntimeError(f"缺少 {b.name}；先运行 --part {'external' if external else 'cv'}")
            parts.append(pd.read_parquet(b.with_suffix(".parquet")))
            logs.append(pd.read_csv(b.with_suffix(".log.tsv"), sep="\t"))
            info.append(json.loads(b.with_suffix(".json").read_text()))
        preds[ct] = pd.concat(parts, ignore_index=True)
    return preds, pd.concat(logs, ignore_index=True), pd.DataFrame(info)


def finish(cfg, models):
    """写（或更新）PROVENANCE.json：记录全部已存在的结果文件。"""
    outs = sorted(OUT.glob("*.tsv")) + sorted(INT.glob("pred_*.parquet"))
    ins = [ROOT / "results/01_dataset/cells.tsv", ROOT / cfg["paths"]["processed"] / "counts.npz"]
    ins += sorted((ROOT / "data/interim/03_benchmark").glob("pred_*_donors.parquet"))
    ins += [ROOT / "data/interim/03e_sota" / f"pred_{ct}_emb_{m}.parquet" for ct in cfg["dataset"]["cell_types"]
            for m in models]
    ins += sorted((ROOT / cfg["finetune"]["geneformer"]["token_dataset"]).glob("*.arrow"))
    ins += [ROOT / "data/interim/06_external_hpap" / f"pred_{ct}.parquet" for ct in cfg["dataset"]["cell_types"]]
    ins += [ROOT / "data/interim/03e_sota" / f"pred_external_{ct}.parquet" for ct in cfg["dataset"]["cell_types"]]
    fi = OUT / "fold_info.tsv"
    gpu_h = float(pd.read_csv(fi, sep="\t").wall_seconds.sum() / 3600) if fi.exists() else None
    write_provenance(STAGE, ins, outs, seed=cfg["seed"],
                     extra={"models": models, "repeats": cfg["finetune"]["repeats"], "gpu_hours": gpu_h,
                            "external_done": (OUT / "metrics_external.tsv").exists()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cv", "summary", "external"])
    ap.add_argument("--models", default=None, help="逗号分隔；缺省为 config finetune.models 全部")
    ap.add_argument("--small-first", action="store_true", help="从小模型开始（在另一组 GPU 上并行启动第二个调度进程时用）")
    ap.add_argument("--gpus", default=None, help="逗号分隔的 GPU 编号；缺省为 config finetune.gpus（只用空闲者）")
    a = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    models = a.models.split(",") if a.models else list(cfg["finetune"]["models"])
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else cfg["finetune"]["gpus"]
    for d in (INT, FOLDS, OUT):
        d.mkdir(parents=True, exist_ok=True)
    import ft_eval
    if a.part == "cv":
        dispatch(cfg, job_list(cfg, models, False, a.small_first), gpus)
    elif a.part == "summary":
        preds, logs, info = collect(cfg, models, False)
        for ct, p in preds.items():
            p.to_parquet(INT / f"pred_{ct}.parquet")
        label = "development_repeat" + "_".join(str(r) for r in cfg["finetune"]["repeats"])
        met, dif = ft_eval.development(cfg, preds, models, label)
        met.to_csv(OUT / "metrics_cv.tsv", sep="\t", index=False)
        dif.to_csv(OUT / "paired_differences.tsv", sep="\t", index=False)
        logs.to_csv(OUT / "training_log.tsv", sep="\t", index=False)
        info.to_csv(OUT / "fold_info.tsv", sep="\t", index=False)
        finish(cfg, models)
        print(met.pivot_table(index=["cell_type", "endpoint"], columns="model", values="within_pearson").round(3)
              .to_string())
    else:
        if not (OUT / "metrics_cv.tsv").exists():
            raise RuntimeError("开发供体汇总（--part summary）尚未完成，外部检验不能运行")
        dispatch(cfg, job_list(cfg, models, True), gpus)
        preds, logs, info = collect(cfg, models, True)
        for ct, p in preds.items():
            p.to_parquet(INT / f"pred_external_{ct}.parquet")
        met, dif = ft_eval.external(cfg, preds, models)
        met.to_csv(OUT / "metrics_external.tsv", sep="\t", index=False)
        dif.to_csv(OUT / "paired_differences_external.tsv", sep="\t", index=False)
        old = pd.read_csv(OUT / "fold_info.tsv", sep="\t")
        old = old[old["repeat"].astype(str) != "external"]
        pd.concat([old, info], ignore_index=True).to_csv(OUT / "fold_info.tsv", sep="\t", index=False)
        oldl = pd.read_csv(OUT / "training_log.tsv", sep="\t")
        oldl = oldl[oldl["repeat"].astype(str) != "external"]
        pd.concat([oldl, logs], ignore_index=True).to_csv(OUT / "training_log.tsv", sep="\t", index=False)
        finish(cfg, models)
        print(met.pivot_table(index=["cell_type", "endpoint"], columns="model", values="within_pearson").round(3)
              .to_string())


if __name__ == "__main__":
    main()
