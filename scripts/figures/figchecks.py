"""Figure checks beyond text–text overlap: minimum font size, and text or legends sitting on lines,
error bars or bars. Called from figstyle.save_fig when FIG_OVERLAP_REPORT is set."""
from __future__ import annotations

import matplotlib as mpl


def font_report(fig, min_pt: float = 6.0) -> list[str]:
    """Visible texts set smaller than `min_pt` (minimum size)."""
    out = set()
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_pt - 1e-6:
            out.add(f"font {t.get_fontsize():.1f} pt: '{t.get_text()[:25]}'")
    return sorted(out)


def data_overlap_report(fig, pad: float = 0.5) -> list[str]:
    """Text or legends inside an axes that sit on lines, error bars or bars of that axes.

    Lines (including error-bar segments) are sampled densely along their drawn
    path; bars are tested as rectangles. Axis spines, tick marks and texts marked
    `_overlap_exempt` are ignored; reference lines spanning the whole axes
    (axhline/axvline) are ignored for legends only when `_ref_line` is set.
    """
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    issues = []

    def sample(xy, n=40):
        xy = np.asarray(xy, float)
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy) < 2:
            return xy
        out = [xy[:1]]
        for a, b in zip(xy[:-1], xy[1:]):
            t = np.linspace(0, 1, n)[1:, None]
            out.append(a + t * (b - a))
        return np.vstack(out)
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_visible() and t.get_text().strip()
                 and not getattr(t, "_overlap_exempt", False)]
        boxes = [(t.get_text()[:30], mpl.text.Text.get_window_extent(t, r)) for t in texts]
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            boxes.append(("legend", lg.get_window_extent(r)))
        if not boxes:
            continue
        pts = []
        for ln in ax.lines:
            if ln.get_visible() and not getattr(ln, "_overlap_exempt", False) and len(ln.get_xydata()):
                pts.append(sample(ln.get_transform().transform(ln.get_xydata())))
        for col in ax.collections:
            if isinstance(col, LineCollection) and col.get_visible() and not getattr(col, "_overlap_exempt", False):
                tr = col.get_transform()
                for seg in col.get_segments():
                    if len(seg):
                        pts.append(sample(tr.transform(seg)))
        rects = [p_.get_window_extent(r) for p_ in ax.patches
                 if isinstance(p_, Rectangle) and p_.get_visible() and not getattr(p_, "_overlap_exempt", False)
                 and p_.get_width() != 0 and p_.get_height() != 0]
        pts = np.vstack(pts) if pts else np.zeros((0, 2))
        for lab, b in boxes:
            inside = ((pts[:, 0] > b.x0 + pad) & (pts[:, 0] < b.x1 - pad) &
                      (pts[:, 1] > b.y0 + pad) & (pts[:, 1] < b.y1 - pad))
            if inside.any():
                issues.append(f"{'legend' if lab == 'legend' else 'text ' + repr(lab)} on a line/error bar")
            if any(b.overlaps(rc) for rc in rects):
                issues.append(f"{'legend' if lab == 'legend' else 'text ' + repr(lab)} on a bar")
    return sorted(set(issues))
