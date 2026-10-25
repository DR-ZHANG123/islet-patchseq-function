"""Stage 04i：模型比较的多重检验校正与重抽样 P 值分辨率（汇总，不拟合模型）。

来源（config multiplicity.sources）：Stage 03c、03e、03f 的配对差值表；另以集成（cell_branch）为参照，用 Stage 03e
的同一函数与同一随机种子（同一组按供体重抽样）重算全部方法的配对差值（multiplicity.recompute_references），
并核对同时重算的 context_model 参照差值与 Stage 03e 存档表一致。
统一口径：difference = 方法 − 参照（正值 = 方法更好；Stage 03c 原表为 参照 − 方法，已取反）。
家族 = 来源 × 划分/集合 × 参照 × 细胞类型 × 层面，族内包含全部方法 × 终点；在族内做 Benjamini–Hochberg。
重抽样 P 值为 0 时（n_boot 次重抽样差值无一越过 0）报告上界 “< 1/n_boot”，BH 校正时以 1/n_boot 代入。
输出 results/04i_multiplicity/{model_comparisons_adjusted.tsv, family_summary.tsv, recompute_check.tsv}。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
import sota_summary  # noqa: E402

STAGE = "04i_multiplicity"
COLS = ["source", "cell_type", "set", "reference", "method", "endpoint", "level", "difference", "lo", "hi", "p_boot"]


def load_source(name: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / name / "paired_differences.tsv", sep="\t")
    if name == "03c_summary":                       # 原表：参照 − 方法
        d = d.rename(columns={"split": "set", "comparator": "method"})
        d["reference"] = "context_model"
        d["difference"], d["lo"], d["hi"] = -d.difference, -d.hi, -d.lo
    d["source"] = name
    return d[COLS]


def recompute(cfg: dict, refs: list) -> pd.DataFrame:
    """以 refs 为参照重算 Stage 03e 开发供体的配对差值（与 sota_summary.run 相同的输入与随机种子）。"""
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t").set_index("cell_id")
    jobs = []
    for ct in cfg["dataset"]["cell_types"]:
        new = pd.concat([pd.read_parquet(f) for f in sota_summary.new_pred_files(ct)], ignore_index=True)
        allp = pd.concat([sota_summary.reference_preds(ct), new], ignore_index=True).join(
            cells[["donor", "glucose"]], on="cell_id")
        jobs += [(ct, ep, s) for ep, s in allp.groupby("endpoint")]
    res = Parallel(n_jobs=min(cfg["multiplicity"]["jobs"], len(jobs)), verbose=2)(
        delayed(sota_summary.boot_endpoint)(ct, ep, s, refs, cfg["summary"]["n_boot"], cfg["seed"],
                                            cfg["cv"]["min_group"], "development") for ct, ep, s in jobs)
    d = pd.DataFrame([x for _, dd in res for x in dd])
    d["source"] = "03e_sota_recomputed"
    return d[COLS]


def adjust(d: pd.DataFrame, n_boot: int, fdr: float) -> pd.DataFrame:
    d = d.copy()
    floor = 1.0 / n_boot
    d["p_at_resolution_limit"] = d.p_boot == 0
    d["p_upper_bound"] = np.where(d.p_at_resolution_limit, floor, d.p_boot)
    d["p_report"] = [f"< {floor:g}" if z else f"{p:.3g}" for z, p in zip(d.p_at_resolution_limit, d.p_boot)]
    d["n_boot"] = n_boot
    fam = ["source", "set", "reference", "cell_type", "level"]
    d["family"] = d[fam].astype(str).agg("/".join, axis=1)
    ok = d.p_upper_bound.notna()
    d["family_size"] = d[ok].groupby("family").p_upper_bound.transform("size")
    d.loc[ok, "q_bh"] = d[ok].groupby("family").p_upper_bound.transform(
        lambda p: false_discovery_control(p.values, method="bh"))
    d["significant_fdr"] = d.q_bh < fdr
    d["direction"] = np.select([d.significant_fdr & (d.difference > 0), d.significant_fdr & (d.difference < 0)],
                               ["method_better", "method_worse"], "not_significant")
    return d


def family_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, src, st, ref, ct, lev), s in d.groupby(["family", "source", "set", "reference", "cell_type", "level"]):
        better = s[s.direction == "method_better"]
        rows.append({"family": fam, "source": src, "set": st, "reference": ref, "cell_type": ct, "level": lev,
                     "n_methods": s.method.nunique(), "n_endpoints": s.endpoint.nunique(), "n_tests": len(s),
                     "n_p_below_0.05": int((s.p_boot < 0.05).sum()),
                     "n_p_at_resolution_limit": int(s.p_at_resolution_limit.sum()),
                     "n_better_fdr": len(better), "n_worse_fdr": int((s.direction == "method_worse").sum()),
                     "better_fdr": "; ".join(f"{m}:{e}" for m, e in zip(better.method, better.endpoint))})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    mc = cfg["multiplicity"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    srcs = [s for s in mc["sources"] if (ROOT / "results" / s / "paired_differences.tsv").exists()]
    parts = [load_source(s) for s in srcs]
    refs = list(dict.fromkeys(mc["recompute_references"] + ["context_model"]))
    rec = recompute(cfg, refs)
    # 核对：重算的 context_model 参照差值应与 Stage 03e 存档表一致
    stored = parts[srcs.index("03e_sota")] if "03e_sota" in srcs else None
    check = pd.DataFrame()
    if stored is not None:
        key = ["cell_type", "reference", "method", "endpoint", "level"]
        check = rec[rec.reference == "context_model"].merge(stored, on=key, suffixes=("_recomputed", "_stored"))
        check["abs_diff"] = (check.difference_recomputed - check.difference_stored).abs()
        check["p_equal"] = np.isclose(check.p_boot_recomputed, check.p_boot_stored) | (
            check.p_boot_recomputed.isna() & check.p_boot_stored.isna())
        check = check[key + ["difference_recomputed", "difference_stored", "abs_diff", "p_equal"]]
    parts.append(rec[rec.reference.isin(mc["recompute_references"])])
    d = adjust(pd.concat(parts, ignore_index=True), cfg["summary"]["n_boot"], mc["fdr"])
    fs = family_summary(d)
    tabs = {"model_comparisons_adjusted.tsv": d, "family_summary.tsv": fs, "recompute_check.tsv": check}
    for fn, df in tabs.items():
        df.to_csv(out / fn, sep="\t", index=False)
    inputs = [ROOT / "results" / s / "paired_differences.tsv" for s in srcs]
    write_provenance(STAGE, inputs, [out / f for f in tabs], seed=cfg["seed"],
                     extra={"families": "source × set × reference × cell_type × level", "fdr": mc["fdr"]})
    if len(check):
        print("重算核对：最大差值", check.abs_diff.max(), " P 一致比例", check.p_equal.mean())
    print(fs.drop(columns=["family", "better_fdr"]).to_string())


if __name__ == "__main__":
    main()
