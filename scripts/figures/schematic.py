"""示意图（Fig. 1、图形摘要）的共用画布：毫米坐标、方框、箭头、文字、图标与版面自查。

文字、方框、箭头和代码图标由 matplotlib 绘制（见 icons_vector.py）；Canvas.image 可另外嵌入透明背景的
PNG 图标（Fig. 1 的 AI 生成示意图标，见 assets/ai_icons/README.md），其余元素仍为矢量。
输出 SVG（文字保留为可编辑文本）、PDF（嵌入字体）与 300 dpi PNG。

版面自查（check）在合成前运行，命中即报告：文字与文字重叠、文字压图标、文字压箭头或分隔线、
文字跨越方框边线（只有一部分在框内）、箭头穿过图标主体、字号低于下限。设置 FIG_OVERLAP_REPORT=1 时
另外运行 figstyle.overlap_report 与 figchecks 的数据重叠和字号检查。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from icons_vector import ICONS

PT = 72 / 25.4                          # mm → pt


class Canvas:
    """左上角为原点、单位为毫米的作图面。"""

    def __init__(self, width_mm: float, height_mm: float):
        self.W, self.H = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_mm); self.ax.set_ylim(0, height_mm); self.ax.axis("off")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.boxes: list[tuple[float, float, float, float]] = []
        self.segments: list[tuple[float, float, float, float]] = []
        self.icons: dict[str, tuple[float, float, float]] = {}
        self.images: list[tuple[str, float, float, float, float]] = []

    def mm(self, x, y):
        return x, self.H - y

    def box(self, x, y, w, h, fc, ec, lw=0.8, r=1.6, ls="-"):
        self.ax.add_patch(FancyBboxPatch(self.mm(x, y + h), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw, ls=ls))
        self.boxes.append((x, y, x + w, y + h))

    def arrow(self, x0, y0, x1, y1, color="#555555", lw=0.9, style="-|>", ls="-", scale=7):
        self.ax.add_patch(FancyArrowPatch(self.mm(x0, y0), self.mm(x1, y1), arrowstyle=style, mutation_scale=scale,
                                          lw=lw, color=color, shrinkA=0, shrinkB=0, ls=ls))
        self.segments.append((x0, y0, x1, y1))

    def line(self, xs, ys, color="#DDDDDD", lw=0.6, **kw):
        self.ax.plot(xs, [self.H - y for y in ys], color=color, lw=lw, solid_capstyle="butt", **kw)
        for i in range(len(xs) - 1):
            self.segments.append((xs[i], ys[i], xs[i + 1], ys[i + 1]))

    def text(self, x, y, s, size=6.5, **kw):
        kw.setdefault("ha", "left"); kw.setdefault("va", "center")
        return self.ax.text(*self.mm(x, y), s, fontsize=size, **kw)

    def icon(self, name, x, y, s):
        """在左上角 (x, y)、边长 s 毫米的正方形内绘制矢量图标，并登记其范围供版面自查。"""
        ICONS[name](self.ax, x, y, s, self.H)
        self.icons[name] = (x, y, s)

    def image(self, path, x, y, w, dpi=600):
        """把透明背景的 PNG 图标按原宽高比置于左上角 (x, y)、宽 w 毫米处（按 dpi 重采样），返回高度（毫米）。"""
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        h = w * im.height / im.width
        px = max(1, int(round(w / 25.4 * dpi)))
        im = im.resize((px, max(1, int(round(px * im.height / im.width)))), Image.LANCZOS)
        self.ax.imshow(np.asarray(im), extent=(x, x + w, self.H - (y + h), self.H - y), interpolation="lanczos",
                       zorder=2)
        self.ax.set_xlim(0, self.W); self.ax.set_ylim(0, self.H)
        self.images.append((Path(path).stem, x, y, w, h))
        return h

    def add_axes(self, x, y, w, h):
        """左上角 (x, y) 处宽 w、高 h（毫米）的子坐标轴。"""
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])

    # ------------------------------------------------------------------ 自查
    def _text_boxes(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        out = []
        for t in self.fig.findobj(mpl.text.Text):
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            out.append((t.get_text().replace("\n", " ")[:28], t.get_fontsize(), x0, self.H - y1, x1, self.H - y0))
        return out

    def check(self, min_pt: float = 6.0, icon_pad: float = 0.12) -> list[str]:
        tb = self._text_boxes()
        issues = [f"font {fs:.1f} pt < {min_pt}: '{lab}'" for lab, fs, *_ in tb if fs < min_pt - 1e-6]
        for i in range(len(tb)):
            for j in range(i + 1, len(tb)):
                a, b = tb[i][2:], tb[j][2:]
                if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                    issues.append(f"text '{tb[i][0]}' overlaps '{tb[j][0]}'")
        for name, (ix, iy, s) in self.icons.items():
            p = icon_pad * s
            for lab, _, x0, y0, x1, y1 in tb:
                if x0 < ix + s - p and x1 > ix + p and y0 < iy + s - p and y1 > iy + p:
                    issues.append(f"text '{lab}' overlaps icon {name}")
            for (ax0, ay0, ax1, ay1) in self.segments:
                t = np.linspace(0, 1, 60)
                px, py = ax0 + t * (ax1 - ax0), ay0 + t * (ay1 - ay0)
                q = 0.2 * s
                if ((px > ix + q) & (px < ix + s - q) & (py > iy + q) & (py < iy + s - q)).any():
                    issues.append(f"arrow/line crosses icon {name}")
        for name, ix, iy, w, h in self.images:         # 位图图标：内缩 5% 的矩形不得压文字或被箭头穿过
            px_, py_ = 0.05 * w, 0.05 * h
            for lab, _, x0, y0, x1, y1 in tb:
                if x0 < ix + w - px_ and x1 > ix + px_ and y0 < iy + h - py_ and y1 > iy + py_:
                    issues.append(f"text '{lab}' overlaps image {name}")
            for (ax0, ay0, ax1, ay1) in self.segments:
                t = np.linspace(0, 1, 60)
                px, py = ax0 + t * (ax1 - ax0), ay0 + t * (ay1 - ay0)
                if ((px > ix + 0.15 * w) & (px < ix + 0.85 * w) & (py > iy + 0.15 * h) & (py < iy + 0.85 * h)).any():
                    issues.append(f"arrow/line crosses image {name}")
        for lab, _, x0, y0, x1, y1 in tb:
            for (ax0, ay0, ax1, ay1) in self.segments:
                t = np.linspace(0, 1, 120)
                px, py = ax0 + t * (ax1 - ax0), ay0 + t * (ay1 - ay0)
                if ((px > x0 + 0.15) & (px < x1 - 0.15) & (py > y0 + 0.15) & (py < y1 - 0.15)).any():
                    issues.append(f"text '{lab}' on an arrow/line")
            for (bx0, by0, bx1, by1) in self.boxes:
                hit = x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0
                inside = x0 >= bx0 + 0.4 and x1 <= bx1 - 0.4 and y0 >= by0 + 0.4 and y1 <= by1 - 0.4
                if hit and not inside:
                    issues.append(f"text '{lab}' crosses a box edge")
        if os.environ.get("FIG_OVERLAP_REPORT"):
            from figchecks import data_overlap_report, font_report
            from figstyle import overlap_report
            issues += [f"report: {m}" for m in overlap_report(self.fig) + data_overlap_report(self.fig)
                       + font_report(self.fig, min_pt)]
        return sorted(set(issues))

    # ------------------------------------------------------------------ 输出
    def compose(self, out_stem: Path, png_px: int) -> None:
        """按画布原尺寸（不裁边）输出 SVG、PDF 与 PNG；PNG 宽 png_px 像素并记录为 300 dpi。"""
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        vec_dpi = 600 if self.images else None          # 嵌入的 PNG 图标在 PDF/SVG 中保留 600 dpi，且不合并为一张图
        with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": "standard",
                             "image.composite_image": False}):
            self.fig.savefig(out_stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0, dpi=vec_dpi or "figure")
            self.fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0, dpi=vec_dpi or "figure")
            self.fig.savefig(out_stem.with_suffix(".png"), bbox_inches=None, pad_inches=0,
                             dpi=png_px / (self.W / 25.4), facecolor="white")
        plt.close(self.fig)
        from PIL import Image                     # 记录 300 dpi，便于按物理尺寸置入
        png = out_stem.with_suffix(".png")
        Image.open(png).save(png, dpi=(300, 300))
