"""Shared figure style (85/170 mm width, 300 dpi, vector PDF).

Every figure script imports this, writes its panel inputs to
figures/source_data/<figure>_<panel>.tsv, and saves PDF + PNG. Retouching a panel
must never require rerunning an analysis stage.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

# 糖代谢状态
STATUS_COLORS = {"ND": "#3A6EA5", "preT2D": "#B07D1A", "T2D": "#C0392B", "T1D": "#7B5EA7"}
STATUS_LABELS = {"ND": "No diabetes", "preT2D": "Prediabetes", "T2D": "Type 2 diabetes", "T1D": "Type 1 diabetes"}
# 细胞类型
TYPE_COLORS = {"Beta": "#2A9D8F", "Alpha": "#C8811B"}
# 模型（本研究模型用深色，对照用灰阶与浅色）。图中名称与正文一致：通用方法名，不自造模型名
MODEL_COLORS = {"context_model": "#B2182B", "cell_branch": "#E36C4C", "ridge": "#2C6FB0",
                "gradient_boosting": "#1A9850", "gradient_boosting_tuned": "#66BD63",
                "knn_predictive_set": "#8E6BB0", "ps_faithful_nd": "#5B3A8C", "mlp_cell": "#E08214", "mlp_cell_tuned": "#E08214",
                "donor_branch": "#F4A582", "linear_cell_branch": "#5E9BD1", "linear_context_model": "#E7877B",
                "elastic_net": "#7FA7D1", "pls": "#A9C4E2", "srrr": "#4D7EA8", "mlp_concat": "#F2B061",
                "mlp_two_within": "#F7CB8F", "scvi": "#6A3D9A", "scgpt_human": "#8C564B",
                "geneformer_v2_104m": "#BC80BD", "geneformer_v2_316m": "#9E5BA8", "uce_4layer": "#7F7F7F",
                "uce_33layer": "#4D4D4D"}
REPRESENTATIONS = {"scvi": "scVI", "scgpt_human": "scGPT", "geneformer_v2_104m": "Geneformer 104M",
                   "geneformer_v2_316m": "Geneformer 316M", "uce_4layer": "UCE 4-layer", "uce_33layer": "UCE 33-layer"}
HEADS = {"ridge": "ridge", "gbm": "boosting", "cell_branch": "ensemble", "context": "ensemble + donor ref.",
         "donor_branch": "donor reference alone"}
MODEL_LABELS = {"context_model": "Ensemble + donor reference", "cell_branch": "Ridge–boosting ensemble",
                "context_model_full_gate": "Ensemble + donor reference (weight 1)",
                "donor_branch": "Donor reference alone", "linear_cell_branch": "Ridge (within-donor term)",
                "linear_context_model": "Ridge (within-donor term) + donor reference",
                "ridge": "Ridge", "elastic_net": "Elastic net", "pls": "PLS", "srrr": "sRRR",
                "gradient_boosting": "Gradient boosting (fixed)", "gradient_boosting_tuned": "Gradient boosting (tuned)",
                "knn_predictive_set": "Outcome-specific kNN",
                "ps_faithful_nd": "Predictive set (original procedure)", "mlp_cell": "Single-cell MLP (fixed)",
                "mlp_cell_tuned": "Single-cell MLP (tuned)",
                "mlp_concat": "MLP + donor mean (fixed)", "mlp_concat_tuned": "MLP + donor mean (tuned)",
                "mlp_two": "Two-branch MLP (fixed)", "mlp_two_tuned": "Two-branch MLP (tuned)",
                "mlp_two_within": "Two-branch MLP + within-donor term (fixed)",
                "mlp_two_within_tuned": "Two-branch MLP + within-donor term (tuned)"}
MODEL_SHORT = {"context_model": "Ensemble + donor ref.", "cell_branch": "Ensemble",
               "linear_context_model": "Ridge (within) + donor ref.", "linear_cell_branch": "Ridge (within-donor term)",
               "mlp_two_within": "Two-branch MLP + within term (fixed)",
               "mlp_two_within_tuned": "Two-branch MLP + within term (tuned)",
               "knn_predictive_set": "Outcome-specific kNN", "ps_faithful_nd": "Predictive set (original procedure)"}


def model_label(model: str, short: bool = False) -> str:
    """方法的图中名称；学习表示上的预测头写作“表示（头）”。"""
    if short and model in MODEL_SHORT:
        return MODEL_SHORT[model]
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    for rep, name in REPRESENTATIONS.items():
        if model.startswith(rep + "_"):
            return f"{name} ({HEADS[model[len(rep) + 1:]]})"
    raise KeyError(model)


def model_color(model: str) -> str:
    for rep in REPRESENTATIONS:
        if model.startswith(rep + "_"):
            return MODEL_COLORS[rep]
    return MODEL_COLORS.get(model, "#888888")


ENDPOINT_LABELS = {"size": "Cell size", "exocytosis": "Total exocytosis", "early_exocytosis": "Early exocytosis",
                   "late_exocytosis": "Late exocytosis", "calcium_entry": "Ca$^{2{+}}$ entry",
                   "early_ca_current": "Early Ca$^{2{+}}$ current", "late_ca_current": "Late Ca$^{2{+}}$ current",
                   "sodium_current": "Peak Na$^{{+}}$ current", "na_half_inactivation": "Na$^{{+}}$ half-inactivation"}
MM = 1 / 25.4      # 毫米 → 英寸；单栏 85 mm，双栏 170 mm，最大高度 225 mm


def apply_style(base_fontsize: float = 8.0) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        # 上下标（Na⁺、Ca²⁺）用正文字体排版，避免 DejaVu 数学字体带来的字形与间距不一致
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.sf": "Liberation Sans",
        "mathtext.cal": "Liberation Sans",
    })


def panel_case(letter: str) -> str:
    """子图字母大小写：FIG_PANEL_CASE=upper 或 lower（默认）。"""
    return letter.upper() if os.environ.get("FIG_PANEL_CASE", "lower").lower() == "upper" else letter.lower()


def fig_out_dir(root: Path) -> Path:
    """图片输出目录：FIG_OUT_DIR（相对项目根目录或绝对路径），默认 figures/；source_data 始终在 figures/source_data。"""
    d = Path(os.environ.get("FIG_OUT_DIR", "figures"))
    d = d if d.is_absolute() else root / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Register a panel letter; save_fig places all of a figure's letters together.

    Letters are positioned from the drawn extent of each panel rather than from
    offsets inside it, so that letters on one row share a baseline and letters in
    one column share a left edge, however large each panel is. `dx` and `dy` are
    accepted for compatibility and no longer used.
    """
    fig = ax.figure
    if not hasattr(fig, "_panel_labels"):
        fig._panel_labels = []
    fig._panel_labels.append((ax, panel_case(letter)))


def _spans_overlap(a0, a1, b0, b1) -> bool:
    inter = min(a1, b1) - max(a0, b0)
    return inter > 0.5 * min(a1 - a0, b1 - b0)


def align_panel_labels(fig, pad_pt: float = 4.0, col_tol: float = 0.07) -> None:
    """Place registered panel letters on shared rows and columns.

    Rows: panels whose axes overlap vertically by more than half of the shorter
    one form a row, and every letter of the row sits just above the highest drawn
    top in it (titles and labels included). Columns: letters whose panels start
    within `col_tol` of the figure width share the leftmost of their left edges.
    """
    col_tol = getattr(fig, "_col_tol", col_tol)
    items = getattr(fig, "_panel_labels", [])
    if not items:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rec = []
    for ax, letter in items:
        tb = inv.transform_bbox(ax.get_tightbbox(r))
        pos = ax.get_position()
        rec.append({"ax": ax, "letter": letter, "x0": tb.x0, "top": tb.y1,
                    "y0": pos.y0, "y1": pos.y1, "px0": pos.x0, "px1": pos.x1})
    # rows
    rows = []
    for it in sorted(rec, key=lambda d: -d["y1"]):
        for row in rows:
            if any(_spans_overlap(it["y0"], it["y1"], o["y0"], o["y1"]) for o in row):
                row.append(it); break
        else:
            rows.append([it])
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows:
        y = max(o["top"] for o in row) + pad
        for o in row:
            o["y"] = y
    # columns: panels that overlap horizontally, or whose drawn left edges lie
    # within `col_tol` of each other, share one left edge (the leftmost)
    parent = list(range(len(rec)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i_ in range(len(rec)):
        for j_ in range(i_ + 1, len(rec)):
            a_, b_ = rec[i_], rec[j_]
            # overlap is measured against the wider panel, so a full-width panel
            # does not chain every column of the row above into one
            inter = min(a_["px1"], b_["px1"]) - max(a_["px0"], b_["px0"])
            wide = max(a_["px1"] - a_["px0"], b_["px1"] - b_["px0"])
            if inter > 0.5 * wide or abs(a_["x0"] - b_["x0"]) < col_tol:
                parent[find(i_)] = find(j_)
    groups = {}
    for k, o in enumerate(rec):
        groups.setdefault(find(k), []).append(o)
    for col in groups.values():
        x = min(o["x0"] for o in col)
        for o in col:
            o["x"] = x
    for o in rec:
        t = fig.text(o["x"], o["y"], o["letter"], fontsize=11, fontweight="bold",
                     ha="left", va="bottom")
        t._panel_letter = True
    fig._panel_labels = []


def save_fig(fig, root: Path, name: str) -> list[Path]:
    align_panel_labels(fig)
    if os.environ.get("FIG_OVERLAP_REPORT"):
        from figchecks import data_overlap_report, font_report
        for msg in overlap_report(fig) + data_overlap_report(fig) + font_report(fig):
            print(f"OVERLAP {name}: {msg}")
    outdir = fig_out_dir(root)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def save_source(df: pd.DataFrame, root: Path, figure: str, panel: str) -> Path:
    d = root / "figures" / "source_data"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{figure}_{panel}.tsv"
    df.to_csv(p, sep="\t", index=False, float_format="%.6g")
    return p


ACRONYMS = {
    "Nadpplus": "NADP+", "Nadph": "NADPH", "Nadp": "NADP", "Nad": "NAD",
    "Perk": "PERK", "Atp": "ATP", "Dna": "DNA", "Rna": "RNA", "Mrna": "mRNA",
    "Tca": "TCA", "Ros": "ROS", "Tnf": "TNF", "Nfkb": "NF-kB", "Il": "IL",
    "Coa": "CoA", "Acyl": "acyl", "Eif2ak4": "EIF2AK4", "Gcn2": "GCN2",
    "Ucp1": "UCP1", "Iv": "IV", "Iii": "III", "Ii": "II", "I": "I",
    "Mhc": "MHC", "Hla": "HLA", "Nk": "NK", "Net": "NET", "Ifn": "IFN",
    "Gtpase": "GTPase", "Rho": "Rho", "Icosanoid": "eicosanoid",
}


PHRASES = {
    "unfolded protein response": "UPR",
    "reactive oxygen species": "ROS",
    "oxidative phosphorylation": "OXPHOS",
    "electron transport chain": "electron transport chain",
}

_ACR_LC = {k.lower(): v for k, v in ACRONYMS.items()}


def tidy_label(text: str, max_len: int | None = None) -> str:
    """Restore acronym casing in MSigDB-derived labels and optionally truncate."""
    out = str(text)
    for a, b in PHRASES.items():
        out = out.replace(a, b)
    words = [_ACR_LC.get(w.lower(), w) for w in out.split(" ")]
    out = " ".join(words)
    if out:
        out = out[0].upper() + out[1:]
    if max_len and len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def overlap_report(fig, point_pad: float = 1.0) -> list[str]:
    """Text that collides with other text or sits on plotted points.

    Every visible text artist (labels, annotations, legends, tick labels) is
    measured after drawing. Two texts whose boxes intersect, or a text box that
    covers a scatter point, are reported. Heatmap cell values are exempt because
    they are meant to sit on their cells.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.legend import Legend
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels exist for ticks outside the view as well; only the drawn ones count
    all_ticks, drawn = set(), set()
    for ax_ in fig.axes:
        for axis in (ax_.xaxis, ax_.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {id(tk.label1), id(tk.label2)}
            for tk in axis._update_ticks():
                drawn |= {id(tk.label1), id(tk.label2)}
    texts = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_ticks and id(t) not in drawn:
            continue
        if getattr(t, "_overlap_exempt", False):
            continue
        texts.append(t)
    # Text.get_window_extent on an Annotation measures the text alone; the
    # annotation's own method also spans its leader line
    boxes = [(t, mpl.text.Text.get_window_extent(t, r)) for t in texts]
    issues = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            if ti.get_rotation() and tj.get_rotation():
                continue        # axis-aligned boxes of two rotated labels overlap falsely
            if bi.overlaps(bj) and bi.width > 0 and bj.width > 0:
                issues.append(f"text '{ti.get_text()[:30]}' overlaps '{tj.get_text()[:30]}'")
    legend_boxes = [lg.get_window_extent(r) for lg in fig.findobj(Legend)]
    for col in fig.findobj(PathCollection):
        if not col.get_visible() or col.axes is None or col.get_offsets().size == 0:
            continue
        pts = col.get_offset_transform().transform(np.asarray(col.get_offsets()))
        for t, b in boxes:
            if t.axes is not col.axes and t.figure is fig and t.axes is not None:
                continue
            inside = ((pts[:, 0] > b.x0 + point_pad) & (pts[:, 0] < b.x1 - point_pad) &
                      (pts[:, 1] > b.y0 + point_pad) & (pts[:, 1] < b.y1 - point_pad))
            if inside.any():
                issues.append(f"text '{t.get_text()[:30]}' covers {int(inside.sum())} point(s)")
        for b in legend_boxes:
            inside = ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) &
                      (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1))
            if inside.any() and col.get_label().startswith("_"):
                issues.append(f"legend covers {int(inside.sum())} point(s)")
    return sorted(set(issues))


def place_labels(ax, xs, ys, labels, fontsize: float = 5.4, italic: bool = False,
                 color: str = "#222222", min_r: float = 9, max_r: float = 70,
                 avoid_axes=None) -> list:
    """Annotate points with labels that land in empty space.

    For each label, candidate positions on rings of growing radius around its
    point are scored against every plotted point in the axes, every label already
    placed, the legend and the axes frame; the first position touching nothing is
    taken, otherwise the one covering the fewest points. A thin leader line joins
    label and point. Axes limits are frozen first, so later autoscaling cannot
    move the points out from under their labels.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Bbox
    fig = ax.figure
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts, bg = [], []
    for col in ax.collections:
        if isinstance(col, PathCollection) and col.get_offsets().size:
            xy = col.get_offset_transform().transform(np.asarray(col.get_offsets()))
            (bg if col.get_gid() == "background" else pts).append(xy)
    line_segs = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            d = ln.get_transform().transform(np.asarray(xy, float))
            pts.append(d)
            line_segs += [(d[i], d[i + 1]) for i in range(len(d) - 1)]
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    bg = np.vstack(bg) if bg else np.zeros((0, 2))
    frame = ax.get_window_extent(r)
    blocked = [lg.get_window_extent(r) for lg in ([ax.get_legend()] if ax.get_legend() else [])]
    blocked += [mpl.text.Text.get_window_extent(t, r) for t in ax.texts if t.get_text().strip()]
    for other in (avoid_axes or []):
        blocked.append(other.get_window_extent(r))
    px = fig.dpi / 72.0
    placed, out, segs = [], [], []

    def segments_cross(a, b, c, d):
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)

    def seg_hits_box(a, b, box, n=24):
        """引线（点 a → 标签中心 b，截去落在自身标签内的部分）是否穿过另一标签框。"""
        for t in np.linspace(0, 1, n):
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            if box.x0 < x < box.x1 and box.y0 < y < box.y1:
                return True
        return False
    def clip_hits(a, b, box):
        """Liang–Barsky：线段 a→b 是否与矩形 box 相交（用于参考线、数据线穿过标签的判断）。"""
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p_, q_ in ((-dx, a[0] - box.x0), (dx, box.x1 - a[0]), (-dy, a[1] - box.y0), (dy, box.y1 - a[1])):
            if p_ == 0:
                if q_ < 0:
                    return False
                continue
            t = q_ / p_
            if p_ < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True
    def near_leader(a, b, q, tol):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not len(q):
            return 0
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        dist = np.linalg.norm(q - (a + t[:, None] * ab), axis=1)
        own = np.linalg.norm(q - a, axis=1) < 1e-6
        return int(((dist < tol) & ~own & (t > 0.05)).sum())
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    for x, y, lab in zip(xs, ys, labels):
        t = ax.text(0, 0, lab, fontsize=fontsize, fontstyle="italic" if italic else "normal")
        w, h = t.get_window_extent(r).width, t.get_window_extent(r).height
        t.remove()
        p0 = ax.transData.transform((x, y))
        best, best_cost = None, None
        for rad in np.arange(min_r, max_r + 1, 4):
            for a in angles:
                cx, cy = p0[0] + rad * px * np.cos(a), p0[1] + rad * px * np.sin(a)
                b = Bbox.from_bounds(cx - w / 2 - 1.5, cy - h / 2 - 1.5, w + 3, h + 3)
                if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
                    continue
                if any(b.overlaps(o) for o in placed + blocked):
                    continue
                if any(seg_hits_box(p0, (cx, cy), o) for o in placed):
                    continue
                if any(seg_hits_box(sa, sb, b) for sa, sb in segs):
                    continue
                if any(segments_cross(p0, (cx, cy), sa, sb) for sa, sb in segs):
                    continue
                inside = lambda q: int(((q[:, 0] > b.x0) & (q[:, 0] < b.x1) & (q[:, 1] > b.y0) & (q[:, 1] < b.y1)).sum())  # noqa: E731
                # 压在突出显示的点上代价极高；压在背景点上按点数计代价；距离越远代价越高
                # 参考线：标签框外扩 2 pt 仍与之相交即计罚，保证文字与线之间留白
                bl = Bbox.from_extents(b.x0 - 2 * px, b.y0 - 2 * px, b.x1 + 2 * px, b.y1 + 2 * px)
                # 突出显示的点按标记半径（约 2.5 pt）计入，标签底板不得压住任何点（包括自身）的边缘
                bp = Bbox.from_extents(b.x0 - 2.5 * px, b.y0 - 2.5 * px, b.x1 + 2.5 * px, b.y1 + 2.5 * px)
                n_hi = int(((pts[:, 0] > bp.x0) & (pts[:, 0] < bp.x1) & (pts[:, 1] > bp.y0) & (pts[:, 1] < bp.y1)).sum())
                cost = n_hi * 1000 + inside(bg) * 12 + rad + 600 * sum(clip_hits(sa, sb, bl) for sa, sb in line_segs)
                # 引线擦过其他突出显示的点（4 pt 以内）会造成指认歧义
                cost += 300 * near_leader(p0, (cx, cy), pts, 4 * px)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (cx - p0[0], cy - p0[1], b), cost
        if best is None:
            raise SystemExit(f"place_labels: no room for '{lab}'")
        dx, dy, b = best
        placed.append(b)
        segs.append((p0, (p0[0] + dx, p0[1] + dy)))
        out.append(ax.annotate(lab, (x, y), xytext=(dx / px, dy / px),
                               textcoords="offset points", ha="center", va="center",
                               fontsize=fontsize, color=color, zorder=7,
                               fontstyle="italic" if italic else "normal",
                               # an invisible plate around the text clips the leader
                               # line at its edge, so the line never runs through it
                               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
                               arrowprops=dict(arrowstyle="-", lw=0.35, color="#777777",
                                               shrinkA=0, shrinkB=1.5)))
    return out
