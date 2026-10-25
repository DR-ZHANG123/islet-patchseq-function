"""PanKbase 整合 10x 胰岛图谱：按供体汇总的 β/α 细胞伪 bulk 与供体临床信息。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from repro import ROOT

GSE221156_PROJECT = "PRJNA913127"


def _read_matrix(path):
    samples = path.open().readline().rstrip("\n").split("\t")
    df = pd.read_csv(path, sep="\t", header=None, skiprows=1, index_col=0, low_memory=False)
    df.columns = samples
    meta = df.iloc[:4].T
    meta.columns = ["center_donor_id", "donor_accession", "biosample_accession", "treatment"]
    counts = df.iloc[4:].apply(pd.to_numeric).T          # 样本 × 基因
    return counts, meta


def _donor_key(sample: str, center_id: str) -> str:
    return sample.split("__")[0] if sample.startswith("HPAP-") else center_id


def load_pankbase_pseudobulk(cfg: dict, cell_type: str):
    """返回 (供体 × 基因 log1p(CPM) 伪 bulk, 供体信息表)。只用未处理样本；同一供体多个样本的计数相加。"""
    raw = ROOT / cfg["paths"]["raw"] / "pankbase_sc"
    counts, meta = _read_matrix(raw / f"{cell_type}_sample_gex_total_counts.fixed.txt")
    keep = meta.treatment == cfg["atlas"]["treatment"]
    counts, meta = counts[keep], meta[keep]
    ncell = pd.read_csv(raw / f"{cell_type}_cell_prop.txt", sep="\t", index_col=0).Freq
    meta = meta.assign(n_cells=ncell.reindex(meta.index).fillna(0).values,
                       donor=[_donor_key(s, c) for s, c in zip(meta.index, meta.center_donor_id)])
    summed = counts.groupby(meta.donor.values).sum()
    cells = meta.groupby("donor").n_cells.sum()
    summed = summed[cells.reindex(summed.index) >= cfg["atlas"]["min_cells"]]
    lib = summed.sum(1).replace(0, 1)
    pb = np.log1p(summed.div(lib, axis=0) * 1e6)
    info = donor_info(cfg, meta)
    return pb, info.reindex(pb.index).assign(n_cells=cells.reindex(pb.index).values)


def donor_info(cfg: dict, meta: pd.DataFrame) -> pd.DataFrame:
    raw = ROOT / cfg["paths"]["raw"] / "pankbase_sc"
    fields = json.loads((raw / "pankbase_donor_fields.json").read_text())
    by_rrid = {f["rrid"].replace("RRID:", ""): f for f in fields if f.get("rrid")}
    runs = pd.read_csv(raw / "sra_run_studies.tsv", sep="\t", index_col=0)
    rows = []
    for d, s in meta.groupby("donor"):
        f = by_rrid.get(s.center_donor_id.iloc[0], {})
        samp = s.index[0].split("__")[0]
        if samp.startswith("SRR") and samp in runs.index:
            source = "GSE221156" if runs.loc[samp, "study_accession"] == GSE221156_PROJECT else "other GEO"
        else:
            source = (f.get("collections") or ["unknown"])[0]
        rows.append({"donor": d, "rrid": s.center_donor_id.iloc[0], "source": source,
                     "diagnosis": f.get("diabetes_status_description"), "age": f.get("age"), "bmi": f.get("bmi"),
                     "hba1c": f.get("hba1c"), "sex": f.get("gender")})
    out = pd.DataFrame(rows).set_index("donor")
    for c in ("age", "bmi", "hba1c"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
