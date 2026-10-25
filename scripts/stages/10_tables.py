"""Stage 10：稿件表格（从结果表生成，不手工录入）。

Table 1：按糖代谢状态汇总供体与细胞特征（供体数、年龄、女性比例、BMI、HbA1c、β 与 α 细胞数、来源中心）。
Table S2：九个电生理终点的定义、单位、变换与可用细胞/供体数。
输出 results/10_tables/{table1.tsv, table1.tex, tableS2.tsv, tableS2.tex}。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, write_provenance  # noqa: E402

STAGE = "10_tables"
ORDER = [("ND", "No diabetes"), ("preT2D", "Prediabetes"), ("T2D", "Type 2 diabetes"), ("T1D", "Type 1 diabetes")]


def med_iqr(s: pd.Series, digits: int = 0) -> str:
    s = s.dropna()
    q1, m, q3 = s.quantile([0.25, 0.5, 0.75])
    f = f"{{:.{digits}f}}"
    return f"{f.format(m)} ({f.format(q1)}–{f.format(q3)})"


MEASURES = [("size", "Cell size", "Membrane capacitance", "pF", "log"),
            ("exocytosis", "Total exocytosis", "Capacitance increase over a train of ten depolarizations", "fF/pF",
             "log(1 + x)"),
            ("early_exocytosis", "Early exocytosis", "Capacitance increase during the first depolarization", "fF/pF",
             "log(1 + x)"),
            ("late_exocytosis", "Late exocytosis", "Capacitance increase during the later depolarizations", "fF/pF",
             "log(1 + x)"),
            ("calcium_entry", "Ca²⁺ entry", "Integrated Ca²⁺ charge during depolarization", "pC/pF", "log(1 + x)"),
            ("early_ca_current", "Early Ca²⁺ current", "Peak Ca²⁺ current amplitude at −10 mV", "pA/pF",
             "log(1 + x)"),
            ("late_ca_current", "Late Ca²⁺ current", "Sustained Ca²⁺ current amplitude at −10 mV", "pA/pF",
             "log(1 + x)"),
            ("sodium_current", "Peak Na⁺ current", "Peak Na⁺ current amplitude", "pA/pF", "log(1 + x)"),
            ("na_half_inactivation", "Na⁺ half-inactivation", "Voltage of half-maximal Na⁺ current inactivation",
             "mV", "none")]


def latex_table(t: pd.DataFrame, spec: str) -> str:
    esc = lambda v: (str(v).replace("%", "\\%").replace("–", "--").replace("²⁺", "\\textsuperscript{2+}")
                     .replace("⁺", "\\textsuperscript{+}").replace("²", "\\textsuperscript{2}")
                     .replace("−", "$-$"))  # noqa: E731
    rows = ["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(esc(c) for c in t.columns) + " \\\\",
            "\\midrule"]
    rows += [" & ".join(esc(v) for v in r) + " \\\\" for r in t.itertuples(index=False)]
    return "\n".join(rows + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def table_s2(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, name, definition, unit, tr in MEASURES:
        b = cells[(cells.cell_type == "Beta") & cells[key].notna()]
        a = cells[(cells.cell_type == "Alpha") & cells[key].notna()]
        rows.append({"Measure": name, "Definition": definition, "Unit": unit, "Analysis scale": tr,
                     "Beta cells (donors)": f"{len(b)} ({b.donor.nunique()})",
                     "Alpha cells (donors)": f"{len(a)} ({a.donor.nunique()})"})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/01_dataset/cells.tsv", sep="\t")
    cells = cells[cells.qc_pass]
    donors = pd.read_csv(ROOT / "results/01_dataset/donors.tsv", sep="\t")
    donors = donors[donors.donor.isin(cells.donor)]
    ext = cfg["cv"]["external_site"]
    rows = []
    for key, label in ORDER + [("all", "All donors")]:
        d = donors if key == "all" else donors[donors.status == key]
        c = cells[cells.donor.isin(d.donor)]
        rows.append({"Group": label, "Donors": len(d),
                     "Development/external donors": f"{(d.site != ext).sum()}/{(d.site == ext).sum()}",
                     "Age, years": med_iqr(d.age), "Female, n (%)": f"{(d.sex == 'Female').sum()} "
                     f"({100 * (d.sex == 'Female').sum() / d.sex.notna().sum():.0f})",
                     "BMI, kg/m²": med_iqr(d.bmi, 1), "HbA1c, %": med_iqr(d.hba1c, 1),
                     "Beta cells (donors)": f"{(c.cell_type == 'Beta').sum()} ({c[c.cell_type == 'Beta'].donor.nunique()})",
                     "Alpha cells (donors)": f"{(c.cell_type == 'Alpha').sum()} ({c[c.cell_type == 'Alpha'].donor.nunique()})"})
    t = pd.DataFrame(rows)
    t.to_csv(out / "table1.tsv", sep="\t", index=False)
    cols = list(t.columns)
    short = {"No diabetes": "ND", "Prediabetes": "Prediabetes", "Type 2 diabetes": "T2D", "Type 1 diabetes": "T1D",
             "All donors": "All"}
    tg = t[t.Group != "All donors"]           # 合计写在表注中，表格保持在版心宽度内
    tex = ["{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{l" + "c" * (len(tg)) + "}", "\\toprule",
           " & " + " & ".join(tg.Group.map(short)) + " \\\\", "\\midrule"]
    for col in cols[1:]:
        tex.append(col.replace("%", "\\%").replace("²", "\\textsuperscript{2}") + " & " +
                   " & ".join(str(v).replace("%", "\\%").replace("–", "--") for v in tg[col]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}"]
    (out / "table1.tex").write_text("\n".join(tex) + "\n")
    print(t.to_string())
    s2 = table_s2(cells)
    s2.to_csv(out / "tableS2.tsv", sep="\t", index=False)
    (out / "tableS2.tex").write_text(latex_table(s2, "p{2.3cm}p{4.6cm}lp{1.4cm}cc"))
    print(s2.to_string())
    write_provenance(STAGE, [ROOT / "results/01_dataset/cells.tsv", ROOT / "results/01_dataset/donors.tsv"],
                     [out / "table1.tsv", out / "table1.tex", out / "tableS2.tsv", out / "tableS2.tex"])


if __name__ == "__main__":
    main()
