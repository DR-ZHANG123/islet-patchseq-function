"""Stage 09g 用函数：精选胞吐/颗粒基因集与 ETV1 作为候选转录携带者、由基因集携带缺陷的模拟、
早期与晚期胞吐 T2D 比的差、按葡萄糖浓度分层的 T2D 比。

记号同 Stage 09e/09f：总效应 = 供体等权线性模型（截距 + T2D + 糖尿病前期 + 葡萄糖 + 来源中心）的 T2D 系数；
直接效应 = 加入条件变量后的 T2D 系数；携带比例 = (总 − 直接) / 总。区间来自状态 × 中心分层的供体重抽样。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from t2d_conditioned import boot_df, crossfit, dml_theta
from t2d_robust import donor_mean, summarise_boot, synth_y, t2d_coef, within_r


# ---------- 精选基因集 ----------

def read_gmt(path) -> dict[str, list[str]]:
    out = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[f[0]] = [g for g in f[2:] if g]
    return out


def panel_genes(gmt: dict, terms: list[str], extra: list[str], symbols: np.ndarray, xlog: np.ndarray,
                rows: np.ndarray, min_detect: float) -> tuple[np.ndarray, pd.DataFrame]:
    """基因集并集 ∩ 本数据基因，且在 rows 细胞中检出比例 ≥ min_detect。返回列索引与基因表。"""
    missing = [t for t in terms if t not in gmt]
    if missing:
        raise KeyError(f"gene-set terms not found: {missing}")
    src = {}
    for t in terms:
        for g in gmt[t]:
            src.setdefault(g, []).append(t.split("(")[-1].rstrip(")"))
    for g in extra:
        src.setdefault(g, []).append("named")
    pos = pd.Series(np.arange(len(symbols)), index=symbols)
    pos = pos[~pos.index.duplicated()]
    idx = pos.reindex(list(src)).dropna().astype(int)
    det = (xlog[np.ix_(rows, idx.values)] > 0).mean(0)
    tab = pd.DataFrame({"gene": idx.index, "col": idx.values, "detected": det,
                        "source": [";".join(src[g]) for g in idx.index]})
    tab["kept"] = tab.detected >= min_detect
    return tab[tab.kept].col.values, tab


def standardise(xlog: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """基因集表达：在 rows 细胞上逐基因标准化；其余细胞置零。"""
    v = xlog[:, cols]
    mu, sd = v[rows].mean(0), v[rows].std(0)
    sd[sd == 0] = 1.0
    z = np.zeros_like(v, dtype=float)
    z[rows] = (v[rows] - mu) / sd
    return z


def panel_pcs(z: np.ndarray, rows: np.ndarray, k: int, seed: int) -> np.ndarray:
    u, s, _ = randomized_svd(z[rows], k, random_state=seed)
    out = np.zeros((len(z), k))
    out[rows] = u * s
    return out


def panel_sets(df: pd.DataFrame, P: np.ndarray, gene: np.ndarray, kb: int) -> dict[str, np.ndarray]:
    p = P[df.cell_row.values]
    g = gene[df.cell_row.values]
    return {"panel_pcs": p, "panel_between": donor_mean(p, df.donor.values)[:, :kb],
            "candidate": g[:, None], "candidate_between": donor_mean(g, df.donor.values)[:, None]}


def panel_shares(df: pd.DataFrame, P, gene, kb, y=None) -> dict:
    tot = t2d_coef(df, None, y)
    out = {"total": tot}
    for name, M in panel_sets(df, P, gene, kb).items():
        d = t2d_coef(df, M, y)
        out[f"direct_{name}"] = d
        out[f"prop_{name}"] = (tot - d) / tot if tot != 0 else np.nan
    return out


def boot_panel(df, P, gene, kb, n_boot, rng) -> list[dict]:
    point = panel_shares(df, P, gene, kb)
    return summarise_boot(point, [panel_shares(boot_df(df, rng), P, gene, kb) for _ in range(n_boot)])


def ridge_share(df: pd.DataFrame, z: np.ndarray, folds: dict, alphas, n_boot: int, rng, y=None) -> list[dict]:
    """交叉拟合岭回归（基因集全部基因 + 协变量），残差固定后按供体重抽样。"""
    ry, rt = crossfit(df, z, folds, np.asarray(alphas), y=y)
    tot = t2d_coef(df, None, y)
    theta = dml_theta(ry, rt, df.donor.values)
    point = {"total": tot, "direct_panel_ridge": theta, "prop_panel_ridge": (tot - theta) / tot}
    r = df.assign(ry=ry, rt=rt, yy=df.y.values if y is None else y)
    boots = []
    for _ in range(n_boot):
        b = boot_df(r, rng)
        t = t2d_coef(b, None, b.yy.values)
        th = dml_theta(b.ry.values, b.rt.values, b.donor.values)
        boots.append({"total": t, "direct_panel_ridge": th, "prop_panel_ridge": (t - th) / t})
    return summarise_boot(point, boots)


# ---------- 候选基因：T2D 差与供体内关联 ----------

def candidate_stats(df: pd.DataFrame, gene: np.ndarray, n_boot: int, rng) -> list[dict]:
    """候选基因（标准化 log 表达）的 T2D 系数，以及 ND、T2D 细胞中与结局的供体内相关。"""
    def stats(d):
        g = gene[d.cell_row.values]
        out = {"gene_t2d": t2d_coef(d, y=g), "detected_nd": float((g[d.status.values == "ND"] > g.min()).mean())}
        for st in ("ND", "T2D"):
            s = d[d.status == st]
            out[f"within_r_{st}"] = within_r(s, s.y.values, gene[s.cell_row.values]) if len(s) > 10 else np.nan
        out["within_r_diff"] = out["within_r_T2D"] - out["within_r_ND"]
        return out
    point = stats(df)
    return summarise_boot(point, [stats(boot_df(df, rng)) for _ in range(n_boot)])


# ---------- 模拟：缺陷由基因集中 T2D 差异最大的基因携带 ----------

def panel_carrier(df: pd.DataFrame, zp: np.ndarray, n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    """逐基因 T2D 系数（供体等权），取绝对值最大的 n 个，按方向加权平均为携带者（全部细胞行）。"""
    coefs = np.array([t2d_coef(df, y=zp[df.cell_row.values, j]) for j in range(zp.shape[1])])
    top = np.argsort(-np.abs(coefs))[:n_genes]
    return (zp[:, top] * np.sign(coefs[top])).mean(1), top


def simulate_panel(df, P, gene, kb, pc50, carrier_all, pis, n_boot, rng) -> list[dict]:
    """携带者取细胞值；估计量：基因集主成分、供体均值主成分、50 个转录组主成分。"""
    def est(d, ys):
        tot = t2d_coef(d, None, ys)
        sets = panel_sets(d, P, gene, kb)
        out = {}
        for name in ("panel_pcs", "panel_between"):
            out[name] = (tot - t2d_coef(d, sets[name], ys)) / tot
        out["full50"] = (tot - t2d_coef(d, pc50[d.cell_row.values], ys)) / tot
        return out
    rows = []
    for pi in pis:
        c = carrier_all[df.cell_row.values]
        ys = synth_y(df, c, pi)
        pt = est(df, ys)
        boots = []
        for _ in range(n_boot):
            b = boot_df(df, rng)
            boots.append(est(b, synth_y(b, carrier_all[b.cell_row.values], pi)))
        bb = pd.DataFrame(boots).replace([np.inf, -np.inf], np.nan)
        for e, v in pt.items():
            x = bb[e].dropna().values
            rows.append({"pi": pi, "estimator": e, "estimate": v, "lo": np.percentile(x, 2.5),
                         "hi": np.percentile(x, 97.5), "share_boot_above_0": float(np.mean(x > 0)),
                         "within_r_y_carrier": within_r(df, ys, c)})
    return rows


# ---------- 早期与晚期胞吐、葡萄糖分层 ----------

def joint_boot(frames: dict, rng) -> dict:
    """各表共用同一次供体抽样（状态 × 中心分层；重复抽到的供体视为不同簇）。"""
    allr = pd.concat([f[["donor", "status", "site"]] for f in frames.values()]).drop_duplicates("donor")
    draws = []
    for _, g in allr.groupby(["status", "site"]):
        draws += list(rng.choice(g.donor.values, len(g), replace=True))
    out = {}
    for key, f in frames.items():
        pos = {d: np.where(f.donor.values == d)[0] for d in f.donor.unique()}
        idx, cid = [], []
        for k, d in enumerate(draws):
            if d in pos:
                idx.append(pos[d])
                cid.append(np.full(len(pos[d]), f"b{k}"))
        b = f.iloc[np.concatenate(idx)].copy()
        b["donor"] = np.concatenate(cid)
        out[key] = b.reset_index(drop=True)
    return out


def early_late(frames: dict, n_boot: int, rng) -> list[dict]:
    """正值反应的 log 几何均值 T2D 系数：晚期、早期及其差（同一供体抽样）。"""
    def stats(fr):
        e, l = t2d_coef(fr["early"]), t2d_coef(fr["late"])
        return {"early": e, "late": l, "late_minus_early": l - e}
    point = stats(frames)
    boots = [stats(joint_boot(frames, rng)) for _ in range(n_boot)]
    rows = summarise_boot(point, boots)
    b = pd.DataFrame(boots)["late_minus_early"].values
    for r in rows:
        if r["quantity"] == "late_minus_early":
            r["p_boot_two_sided"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / (1 + len(b))))
    return rows


def glucose_strata(df: pd.DataFrame, levels: list, n_boot: int, rng) -> list[dict]:
    """各葡萄糖浓度内的 T2D 系数（供体等权），以及最高与最低浓度之差。"""
    def stats(d):
        out = {}
        for g in levels:
            s = d[d.glucose == g]
            try:
                out[f"glucose_{g}"] = t2d_coef(s)
            except (ValueError, np.linalg.LinAlgError):
                out[f"glucose_{g}"] = np.nan
        out[f"diff_{levels[-1]}_minus_{levels[0]}"] = out[f"glucose_{levels[-1]}"] - out[f"glucose_{levels[0]}"]
        mid = levels[1:-1]
        for g in mid:
            others = [out[f"glucose_{x}"] for x in levels if x != g]
            out[f"diff_{g}_minus_others"] = out[f"glucose_{g}"] - float(np.mean(others))
        return out
    point = stats(df)
    boots = [stats(boot_df(df, rng)) for _ in range(n_boot)]
    rows = summarise_boot(point, boots)
    bt = pd.DataFrame(boots)
    for r in rows:
        if r["quantity"].startswith("diff_"):
            b = bt[r["quantity"]].dropna().values
            r["p_boot_two_sided"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / (1 + len(b))))
        if r["quantity"].startswith("glucose_"):
            s = df[df.glucose == int(r["quantity"].split("_")[1])]
            st = s.groupby("donor").status.first()
            r |= {"n_cells": len(s), "n_nd_donors": int((st == "ND").sum()), "n_t2d_donors": int((st == "T2D").sum())}
    return rows
