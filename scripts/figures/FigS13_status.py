"""Supplementary Fig. S13：只用 ND 供体训练的模型在 T2D 与 ND 供体中的供体内精度之差（95% 与 90% 区间；阴影为 ±0.1）。
只读 results/09_t2d_function/coupling_differences.tsv。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MM, apply_style, save_fig, save_source  # noqa: E402
import fig5_panels as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    apply_style()
    margin = yaml.safe_load(open(ROOT / "config/run.yaml"))["t2d"]["equivalence_margin"]
    cd = pd.read_csv(ROOT / "results/09_t2d_function/coupling_differences.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(170 * MM, 62 * MM))
    P.coupling(ax, cd, margin)
    save_source(cd[cd.quantity == "diff_T2D"], ROOT, "FigS13", "encoding_by_status")
    save_fig(fig, ROOT, "FigS13")


if __name__ == "__main__":
    main()
