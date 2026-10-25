"""Stage 00：下载全部公开数据并记录 MD5。

- HumanIslets（api2.xialab.ca）：单细胞电生理表 ephys_cell.csv、六个单细胞表达 HDF5（α/β × 1/5/10 mM）、
  供体级表（donor、isolation、ephys_donor、gsis、cell_pro）；胰岛灌流胰岛素分泌表（peri_gluc，附 function_summary）
  单独放在 humanislets_omics/，不触动已下载的供体表。
- GEO：GSE124742、GSE164875、GSE270484 的原始计数表（用于核对 HumanIslets 表达值的还原）。
- Camunas-Soler 2020 仓库：细胞元数据与 siRNA 敲低结果。
- JCI 2025（PMC12646676）补充表：供体临床信息。
- PanKbase：HPAP 等供体的糖尿病诊断。
- Bandesh 等 2026（EMBO J）Dataset EV7：β 细胞 T2D vs ND 差异表达表（外部 T2D 特征）。
已存在且非空的文件不重复下载（--force 时重下）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, md5, write_provenance  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def fetch(url: str, dest: Path, force: bool, timeout: int = 600) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            return dest
        except requests.RequestException as exc:  # 网络抖动重试
            print(f"  retry {attempt + 1} {url}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def humanislets(cfg: dict, out: Path, force: bool) -> list[Path]:
    api = cfg["download"]["humanislets_api"]
    files: list[Path] = []
    sess = requests.Session()
    usr = sess.get(api + "login", headers=UA, timeout=60).text.strip()
    for fn in cfg["download"]["humanislets_patchseq_files"]:
        dest = out / fn
        if dest.exists() and not force:
            files.append(dest)
            continue
        z = sess.get(f"{api}patchscdownload/{usr}", params={"filename": fn, "version": "v1"},
                     headers=UA, timeout=600).text
        if not z.endswith("RES-OK"):
            raise RuntimeError(f"HumanIslets refused {fn}: {z[:200]}")
        zpath = fetch(f"{api}download/{usr}/{z[:-6]}", out / f"{fn}.zip", True)
        with zipfile.ZipFile(zpath) as zf:
            member = [m for m in zf.namelist() if m.endswith(fn)][0]
            dest.write_bytes(zf.read(member))
        zpath.unlink()
        files.append(dest)
    # 供体级表：先上传供体列表，再请求表格
    import pandas as pd
    donors = sorted(pd.read_csv(out / "ephys_cell.csv").record_id.unique())
    tables = cfg["download"]["humanislets_donor_tables"]
    if force or not all((out / f"{t}.csv").exists() for t in tables):
        up = sess.get(f"{api}uploaddonors/{usr}", params={"ids": ",".join(donors), "type": "redcap"},
                      headers=UA, timeout=120).text
        print("  HumanIslets donors matched:", up)
        z = sess.get(f"{api}hitables/{usr}", params={"tables": ",".join(tables), "filetype": "csv",
                                                      "version": "v1"}, headers=UA, timeout=600).text
        zpath = fetch(f"{api}download/{usr}/{z.replace('RES-OK', '')}", out / "donor_tables.zip", True)
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.namelist():
                name = Path(m).name
                if name.endswith((".csv", ".txt")):
                    (out / name).write_bytes(zf.read(m))
        zpath.unlink()
    files += [out / f"{t}.csv" for t in tables]
    return files


def humanislets_secretion(cfg: dict, out: Path, donors: list[str], force: bool) -> list[Path]:
    """胰岛灌流胰岛素分泌（本研究供体）：hitables 接口返回 peri_gluc.csv 与 function_summary.csv（各附 README）。"""
    api = cfg["download"]["humanislets_api"]
    tables = cfg["download"]["humanislets_secretion_tables"]
    want = [out / f"{t}.csv" for t in tables] + [out / "function_summary.csv"]
    if all(p.exists() for p in want) and not force:
        return want
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    usr = sess.get(api + "login", headers=UA, timeout=60).text.strip()
    up = sess.get(f"{api}uploaddonors/{usr}", params={"ids": ",".join(donors), "type": "redcap"},
                  headers=UA, timeout=120).text
    print("  HumanIslets donors matched:", up)
    z = sess.get(f"{api}hitables/{usr}", params={"tables": ",".join(tables), "filetype": "csv", "version": "v1"},
                 headers=UA, timeout=600).text
    if not z.endswith("RES-OK"):
        raise RuntimeError(f"HumanIslets refused {tables}: {z[:200]}")
    zpath = fetch(f"{api}download/{usr}/{z.replace('RES-OK', '')}", out / "secretion_tables.zip", True)
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():
            name = Path(m).name
            if name.endswith((".csv", ".txt")):
                (out / name).write_bytes(zf.read(m))
    zpath.unlink()
    return want


def pankbase(cfg: dict, out: Path, donors: list[str], force: bool) -> list[Path]:
    """检索 PanKbase 全部供体，按 HumanIslets / PANC-DB 标识匹配本研究供体，再取匹配供体的完整记录。"""
    dest = out / "pankbase_matched_donors.json"
    if dest.exists() and not force:
        return [dest]
    api = cfg["download"]["pankbase_api"]
    r = requests.get(api + "/search/?type=HumanDonor&limit=all&format=json", headers=UA, timeout=600)
    r.raise_for_status()
    wanted = set(donors)
    matched = {}
    for g in r.json()["@graph"]:
        for ident in g.get("human_donor_identifiers") or []:
            local = ident.split(":", 1)[-1]
            if local in wanted:
                matched[local] = g["accession"]
    records = []
    for local, acc in sorted(matched.items()):
        rec = requests.get(f"{api}/human-donors/{acc}/?format=json", headers=UA, timeout=120).json()
        rec = {k: v for k, v in rec.items() if k != "audit"}
        rec["study_donor_id"] = local
        records.append(rec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records, indent=1))
    return [dest]


def pankbase_atlas(cfg: dict, out: Path, force: bool) -> list[Path]:
    """PanKbase 整合 10x 图谱的按样本 β/α 伪 bulk 计数、细胞数，以及全部供体的临床字段。"""
    files = []
    base = cfg["download"]["pankbase_s3"]
    for ct in cfg["dataset"]["cell_types"]:
        for suffix in ("_sample_gex_total_counts.fixed.txt", "_cell_prop.txt"):
            files.append(fetch(f"{base}/{ct}{suffix}", out / f"{ct}{suffix}", force))
    dest = out / "pankbase_donor_fields.json"
    if force or not dest.exists():
        fields = "&".join(f"field={f}" for f in ("accession", "rrid", "diabetes_status_description", "age", "bmi",
                                                   "hba1c", "gender", "human_donor_identifiers", "collections"))
        r = requests.get(f"{cfg['download']['pankbase_api']}/search/?type=HumanDonor&limit=all&format=json&{fields}",
                         headers=UA, timeout=600)
        r.raise_for_status()
        dest.write_text(json.dumps(r.json()["@graph"], indent=1))
    files.append(dest)
    # SRA 运行号 → 原始研究（ENA），用于按来源研究分层
    runs_out = out / "sra_run_studies.tsv"
    if force or not runs_out.exists():
        header = (out / f"{cfg['dataset']['cell_types'][0]}_sample_gex_total_counts.fixed.txt").open().readline()
        runs = sorted({c.split("__")[0] for c in header.rstrip("\n").split("\t") if c.startswith("SRR")})
        rows = ["run\tstudy_accession\tgsm\tsample_title"]
        for run in runs:
            q = requests.get("https://www.ebi.ac.uk/ena/portal/api/filereport",
                             params={"accession": run, "result": "read_run",
                                     "fields": "study_accession,sample_alias,sample_title"}, headers=UA, timeout=120)
            line = q.text.strip().split("\n")[-1].split("\t")
            rows.append("\t".join([run] + line[1:4]))
        runs_out.write_text("\n".join(rows) + "\n")
    files.append(runs_out)
    return files


def genesets(cfg: dict, out: Path, force: bool) -> list[Path]:
    """Enrichr 发布的基因集库（gseapy 接口），存为 GMT，保证富集分析可离线复现。"""
    import gseapy as gp
    files = []
    out.mkdir(parents=True, exist_ok=True)
    for lib in cfg["programmes"]["libraries"]:
        dest = out / f"{lib}.gmt"
        if force or not dest.exists():
            sets = gp.get_library(lib, organism="Human")
            dest.write_text("".join(f"{k}\tNA\t" + "\t".join(v) + "\n" for k, v in sets.items()))
        files.append(dest)
    return files


def geo_soft(cfg: dict, out: Path, force: bool) -> list[Path]:
    files = []
    for acc in cfg["download"]["geo_soft"]:
        rel = f"{acc[:-3]}nnn/{acc}/soft/{acc}_family.soft.gz"
        files.append(fetch("https://ftp.ncbi.nlm.nih.gov/geo/series/" + rel, out / f"{acc}_family.soft.gz", force))
    return files


def jci_supplement(cfg: dict, out: Path, force: bool) -> list[Path]:
    dest = out / "jci-135-195267-s109.xlsx"
    if dest.exists() and not force:
        return [dest]
    pmcid = cfg["download"]["jci2025_pmcid"]
    zpath = fetch(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles",
                  out / "supplementary.zip", True)
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():
            if m.endswith(".xlsx"):
                (out / Path(m).name).write_bytes(zf.read(m))
    zpath.unlink()
    return sorted(out.glob("*.xlsx"))


def bandesh2026(cfg: dict, out: Path, force: bool) -> list[Path]:
    url = cfg["download"]["bandesh2026_esm"]
    return [fetch(url, out / Path(url).name, force)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    outputs: list[Path] = []
    print("HumanIslets ...")
    outputs += humanislets(cfg, raw / "humanislets", args.force)
    print("GEO ...")
    for acc, rel in cfg["download"]["geo_counts"].items():
        outputs.append(fetch("https://ftp.ncbi.nlm.nih.gov/geo/series/" + rel,
                             raw / "geo" / Path(rel).name, args.force))
    print("Camunas-Soler repository ...")
    for fn in cfg["download"]["camunas_files"]:
        outputs.append(fetch(cfg["download"]["camunas_repo"] + fn, raw / "camunas" / fn, args.force))
    print("JCI 2025 supplement ...")
    outputs += jci_supplement(cfg, raw / "jci2025", args.force)
    print("GEO SOFT ...")
    outputs += geo_soft(cfg, raw / "geo_soft", args.force)
    print("PanKbase ...")
    import pandas as pd
    donors = sorted(pd.read_csv(raw / "humanislets" / "ephys_cell.csv").record_id.unique())
    outputs += pankbase(cfg, raw / "pankbase", donors, args.force)
    print("HumanIslets islet perifusion ...")
    outputs += humanislets_secretion(cfg, raw / "humanislets_omics", donors, args.force)
    print("gene sets ...")
    outputs += genesets(cfg, raw / "genesets", args.force)
    print("PanKbase 10x atlas ...")
    outputs += pankbase_atlas(cfg, raw / "pankbase_sc", args.force)
    print("Bandesh 2026 supplement ...")
    outputs += bandesh2026(cfg, raw / "bandesh2026", args.force)
    manifest = ROOT / "metadata" / "download_md5.tsv"
    manifest.parent.mkdir(exist_ok=True)
    with open(manifest, "w") as fh:
        fh.write("file\tbytes\tmd5\n")
        for p in outputs:
            fh.write(f"{p.relative_to(ROOT)}\t{p.stat().st_size}\t{md5(p)}\n")
    write_provenance("00_download", [], outputs + [manifest], seed=None,
                     extra={"note": "public data download; see metadata/data_sources.tsv"})
    print("done:", len(outputs), "files")


if __name__ == "__main__":
    main()
