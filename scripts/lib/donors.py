"""供体临床信息整合：HumanIslets（ADI IsletCore）> PanKbase（HPAP 等）> JCI 2025 补充表 > GEO 样本注释。

统一的糖代谢分组沿用 HumanIslets 的计算规则：临床 T1D → T1D；临床 T2D 或 HbA1c ≥ 6.5% → T2D；
HbA1c 5.8–6.4% → 糖尿病前期（preT2D）；其余为无糖尿病（ND）。自身抗体阳性但无糖尿病的 HPAP 供体记为 AAb+。
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def computed_status(clinical: str | float, hba1c: float) -> str:
    c = str(clinical) if isinstance(clinical, str) else ""
    if c == "T1D":
        return "T1D"
    if c == "T2D" or (pd.notna(hba1c) and hba1c >= 6.5):
        return "T2D"
    if pd.notna(hba1c) and 5.8 <= hba1c < 6.5:
        return "preT2D"
    if c == "AAb+":
        return "AAb+"
    return "ND"


def from_humanislets(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    clin = d["diagnosis"].map({"Type1": "T1D", "Type2": "T2D"}).fillna("ND")
    return pd.DataFrame({"donor": d.record_id, "clinical": clin, "age": d.donorage, "sex": d.donorsex,
                         "bmi": d.bodymassindex, "hba1c": d.hba1c, "source_meta": "HumanIslets"})


def from_pankbase(path: Path) -> pd.DataFrame:
    recs = json.loads(Path(path).read_text())
    rows = []
    for r in recs:
        desc = (r.get("diabetes_status_description") or "").lower()
        aab = any(r.get(k) for k in ("aab_gada", "aab_ia2", "aab_iaa", "aab_znt8"))
        if "type 1" in desc:
            clin = "T1D"
        elif "type 2" in desc:
            clin = "T2D"
        elif aab:
            clin = "AAb+"
        else:
            clin = "ND"
        rows.append({"donor": r["study_donor_id"], "clinical": clin, "age": r.get("age"),
                     "sex": r.get("gender") or r.get("genetic_sex"), "bmi": r.get("bmi"),
                     "hba1c": r.get("hba1c"), "source_meta": "PanKbase"})
    return pd.DataFrame(rows)


def from_jci(path: Path) -> pd.DataFrame:
    j = pd.read_excel(path, sheet_name="S1.1. Donor Information", header=3)
    j = j[j["Donors"].notna()].copy()
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    return pd.DataFrame({"donor": j["Donors"].astype(str).str.strip(),
                         "clinical": j["Diabetes Diagnosis"].map({"T1D": "T1D", "ND": "ND", "T2D": "T2D"}),
                         "age": num(j["Age (Years)"]), "sex": j["Sex"], "bmi": num(j["BMI (kg/m^2)"]),
                         "hba1c": num(j["HbA1c (%)"]), "source_meta": "JCI2025"})


def parse_geo_soft(path: Path) -> pd.DataFrame:
    rows, cur = [], None
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                cur = {"gsm": line.split("= ")[1]}
                rows.append(cur)
            elif cur is not None and line.startswith("!Sample_title"):
                t = line.split("= ", 1)[1]
                m = re.search(r"\[(.*)\]", t)
                cur["cell_id"] = m.group(1) if m else t
            elif cur is not None and line.startswith("!Sample_characteristics_ch1"):
                k, v = line.split("= ", 1)[1].split(": ", 1)
                cur[k.strip().lower()] = v.strip()
    return pd.DataFrame(rows)


def from_geo(soft: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """供体级诊断：GEO 每个细胞的诊断注释，按 HumanIslets 细胞—供体对应汇总（须一致）。"""
    col = "diagnosis" if "diagnosis" in soft else "disease state"
    m = soft.merge(cells[["cell_id", "record_id"]], on="cell_id")
    lab = m[col].map({"Nondiabetic": "ND", "ND": "ND", "T2D": "T2D", "T1D": "T1D"})
    sex = m["sex"] if "sex" in m else pd.Series(np.nan, index=m.index)
    g = pd.DataFrame({"donor": m.record_id, "clinical": lab, "sex": sex}).dropna(subset=["clinical"])
    agree = g.groupby("donor").clinical.nunique()
    if (agree > 1).any():
        raise ValueError(f"inconsistent GEO diagnosis within donor: {agree[agree > 1].index.tolist()}")
    out = g.drop_duplicates("donor").copy()
    out["source_meta"] = "GEO"
    return out


def harmonize(sources: list[pd.DataFrame], donors: list[str]) -> pd.DataFrame:
    """按优先级逐字段合并（前面的来源优先），并计算统一分组。"""
    out = pd.DataFrame({"donor": donors})
    fields = ["clinical", "age", "sex", "bmi", "hba1c"]
    for f in fields:
        out[f] = pd.Series([np.nan] * len(out), dtype=object)
    out["source_meta"] = ""
    for src in sources:
        s = src.drop_duplicates("donor").set_index("donor")
        for f in fields:
            if f not in s:
                continue
            fill = out[f].isna() & out.donor.isin(s.index)
            out.loc[fill, f] = out.loc[fill, "donor"].map(s[f])
        new = (out.source_meta == "") & out.donor.isin(s.index)
        out.loc[new, "source_meta"] = s["source_meta"].iloc[0]
    for f in ("age", "bmi", "hba1c"):
        out[f] = pd.to_numeric(out[f], errors="coerce")
    out["sex"] = out["sex"].astype(str).str.capitalize().replace({"M": "Male", "F": "Female", "Nan": np.nan})
    out["status"] = [computed_status(c, h) for c, h in zip(out.clinical, out.hba1c)]
    out["site"] = out.donor.str.extract(r"^(HPAP|nPOD|AF|AG|H|R)")[0].map(
        {"R": "ADI IsletCore", "HPAP": "HPAP", "H": "Alberta CIL", "nPOD": "nPOD", "AF": "Other", "AG": "Other"})
    return out
