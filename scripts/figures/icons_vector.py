"""示意图图标：全部由代码以矢量图形绘制（matplotlib 图形元素），扁平科学插图风格。

每个图标函数在示意画布（schematic.Canvas，毫米坐标，原点在左上角）上占据左上角 (x, y)、边长 s 毫米的正方形；
函数内部使用单位坐标（0–1，y 向上），由 _unit() 换算到画布坐标。不读取任何外部图片或路径文件。
图标：供体人体轮廓（胰腺突出）、胰岛细胞团（α/β 细胞两色）、单个胰岛细胞（细胞核、分泌颗粒、线粒体）、
膜片钳电极接触细胞、同一供体的一组细胞、转录本、液滴测序通道、质膜下的胰岛素颗粒；另有表达谱条形图与电生理曲线两个小图形。
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

# 配色：与 figstyle.TYPE_COLORS 一致（β 细胞青绿、α 细胞琥珀），浅色填充、深色描边
CELL = {"beta": {"fill": "#BFE6DF", "edge": "#2A9D8F", "inner": "#E3F4F1", "nuc": "#5FB8AC"},
        "alpha": {"fill": "#F8DDAE", "edge": "#C8811B", "inner": "#FCEFD6", "nuc": "#E0A54B"}}
NUC_FILL, NUC_EDGE, NUCLEOLUS = "#C9B6E4", "#7B5EA7", "#7B5EA7"
GRAN_HALO, GRAN_CORE = "#F7C873", "#C8811B"
MITO_FILL, MITO_EDGE = "#F5C1B4", "#C8644F"
BODY_FILL, BODY_EDGE = "#DCE1E6", "#9AA4AE"
PANC_FILL, PANC_EDGE, PANC_GLOW = "#F4B860", "#B9711A", "#F8D9A6"
GLASS_FILL, GLASS_EDGE, WIRE = "#E3EEF6", "#7A97AE", "#9AA4AE"


def _unit(ax, x, y, s, H):
    """单位正方形（y 向上）→ 画布数据坐标的变换；画布 y 轴为 H - 毫米。"""
    return Affine2D().scale(s).translate(x, H - y - s) + ax.transData


def _blob(cx, cy, r, seed, amp=0.045, n=120, aspect=1.0):
    """略不规则的光滑闭合轮廓（细胞膜），返回 n×2 顶点。"""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + sum(amp / k * np.sin(k * th + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 5)))
    return np.c_[cx + rr * np.cos(th) * aspect, cy + rr * np.sin(th)]


def _chaikin(pts, it=4):
    """Chaikin 角切割平滑闭合折线。"""
    p = np.asarray(pts, float)
    for _ in range(it):
        q = np.roll(p, -1, axis=0)
        p = np.c_[0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q].reshape(-1, 2)
    return p


def _mito(ax, T, cx, cy, L, w, ang, lw):
    """线粒体：椭圆外膜 + 内部之字形嵴。"""
    ax.add_patch(Ellipse((cx, cy), L, w, angle=ang, fc=MITO_FILL, ec=MITO_EDGE, lw=lw, transform=T))
    t = np.linspace(-0.36, 0.36, 7) * L
    zz = np.where(np.arange(7) % 2 == 0, -0.22, 0.22) * w
    a = np.deg2rad(ang)
    ax.plot(cx + t * np.cos(a) - zz * np.sin(a), cy + t * np.sin(a) + zz * np.cos(a), color=MITO_EDGE,
            lw=lw * 0.7, transform=T, solid_joinstyle="round")


def _granules(ax, T, cx, cy, r, n, rg, seed, avoid=()):
    """致密核心分泌颗粒：浅色晕 + 深色核心，均匀散布在胞质环带内，避开细胞核与线粒体。"""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        a, d = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0.12, 0.72))
        px, py = cx + d * np.cos(a), cy + d * np.sin(a)
        if all(np.hypot(px - ax_, py - ay_) > ar for ax_, ay_, ar in avoid) and \
                all(np.hypot(px - qx, py - qy) > 2.4 * rg for qx, qy in pts):
            pts.append((px, py))
    for px, py in pts:
        ax.add_patch(Circle((px, py), rg, fc=GRAN_HALO, ec="none", transform=T))
        ax.add_patch(Circle((px, py), rg * 0.45, fc=GRAN_CORE, ec="none", transform=T))


def islet_cell(ax, T, cx, cy, r, kind="beta", seed=0, detail=2, lw=0.8):
    """一个胰岛细胞。detail=2：细胞核、核仁、颗粒、线粒体、内质网；1：细胞核与颗粒；0：仅细胞核。"""
    c = CELL[kind]
    ax.add_patch(Polygon(_blob(cx, cy, r, seed), closed=True, fc=c["fill"], ec=c["edge"], lw=lw,
                         joinstyle="round", transform=T))
    if detail >= 2:                                   # 胞质内层浅色，形成双层膜的立体感
        ax.add_patch(Polygon(_blob(cx, cy, r * 0.9, seed), closed=True, fc=c["inner"], ec="none", transform=T))
    nx, ny, nr = cx - 0.08 * r, cy + 0.02 * r, (0.36 if detail else 0.38) * r
    if detail == 0:
        ax.add_patch(Circle((nx, ny), nr, fc=c["nuc"], ec="none", transform=T))
        return
    ax.add_patch(Polygon(_blob(nx, ny, nr, seed + 7, amp=0.03), closed=True, fc=NUC_FILL, ec=NUC_EDGE,
                         lw=lw * 0.8, transform=T))
    ax.add_patch(Circle((nx + 0.08 * r, ny + 0.06 * r), 0.11 * r, fc=NUCLEOLUS, ec="none", alpha=0.85,
                        transform=T))
    avoid = [(nx, ny, nr * 1.2)]
    if detail >= 2:
        mitos = [(cx + 0.52 * r, cy - 0.35 * r, 35), (cx - 0.35 * r, cy - 0.58 * r, -15),
                 (cx - 0.48 * r, cy + 0.52 * r, 60)]
        for mx, my, ang in mitos:
            _mito(ax, T, mx, my, 0.30 * r, 0.13 * r, ang, lw * 0.7)
            avoid.append((mx, my, 0.22 * r))
        for k, rad in enumerate((0.50, 0.58)):          # 内质网：围绕细胞核的两条弧线
            th = np.linspace(0.35, 1.25, 40) * np.pi
            ax.plot(nx + rad * r * np.cos(th), ny + rad * r * np.sin(th), color=c["edge"], lw=lw * 0.55,
                    alpha=0.6, transform=T, solid_capstyle="round")
            avoid.append((nx + rad * r * np.cos(0.8 * np.pi), ny + rad * r * np.sin(0.8 * np.pi), 0.12 * r))
    _granules(ax, T, cx, cy, r * 0.88, 22 if detail >= 2 else 11, (0.05 if detail >= 2 else 0.065) * r, seed + 3, avoid)


# ------------------------------------------------------------------ 图标
def donor(ax, x, y, s, H):
    """供体：人体正面轮廓，腹部胰腺以琥珀色突出（光晕 + 器官形状）。"""
    T = _unit(ax, x, y, s, H)
    half = [(0.50, 0.785), (0.535, 0.78), (0.54, 0.745), (0.60, 0.73), (0.645, 0.715), (0.665, 0.68),
            (0.675, 0.58), (0.69, 0.47), (0.70, 0.40), (0.69, 0.355), (0.672, 0.36), (0.665, 0.42),
            (0.652, 0.52), (0.64, 0.60), (0.625, 0.60), (0.62, 0.52), (0.625, 0.44), (0.615, 0.30),
            (0.605, 0.15), (0.60, 0.06), (0.615, 0.03), (0.57, 0.025), (0.555, 0.08), (0.545, 0.25),
            (0.52, 0.40), (0.50, 0.415)]
    right = np.array(half)
    left = right[::-1].copy(); left[:, 0] = 1 - left[:, 0]
    body = _chaikin(np.r_[right, left[1:-1]], 3)
    body[:, 1] = 0.09 + (body[:, 1] - 0.025) * 0.86             # 竖向留出头部空间
    ax.add_patch(Polygon(body, closed=True, fc=BODY_FILL, ec=BODY_EDGE, lw=0.6, joinstyle="round", transform=T))
    ax.add_patch(Ellipse((0.5, 0.812), 0.115, 0.135, fc=BODY_FILL, ec=BODY_EDGE, lw=0.6, transform=T))
    ax.add_patch(Ellipse((0.505, 0.555), 0.22, 0.13, angle=12, fc=PANC_GLOW, ec="none", alpha=0.75, transform=T))
    # 胰腺（以下 y 坐标整体上移 0.05，与躯干一致）：胰头在观察者左侧（供体右侧），胰体与胰尾向右上方渐细
    top = [(0.425, 0.555), (0.44, 0.580), (0.48, 0.575), (0.53, 0.585), (0.575, 0.600), (0.595, 0.598)]
    bot = [(0.60, 0.585), (0.57, 0.573), (0.525, 0.555), (0.48, 0.545), (0.455, 0.525), (0.43, 0.528)]
    ax.add_patch(Polygon(_chaikin(top + bot, 3), closed=True, fc=PANC_FILL, ec=PANC_EDGE, lw=0.5,
                         joinstyle="round", transform=T))


def islet(ax, x, y, s, H):
    """胰岛：约 19 个紧密排列的细胞，β 细胞青绿、α 细胞琥珀，外围淡色轮廓。"""
    T = _unit(ax, x, y, s, H)
    ax.add_patch(Circle((0.5, 0.5), 0.445, fc="#EEF6F4", ec="#BFD9D4", lw=0.5, transform=T))
    pos = [(0.0, 0.0)] + [(0.185 * np.cos(a), 0.185 * np.sin(a)) for a in np.deg2rad(np.arange(30, 390, 60))]
    pos += [(0.345 * np.cos(a), 0.345 * np.sin(a)) for a in np.deg2rad(np.arange(0, 360, 30))]
    rng = np.random.default_rng(11)
    alpha_idx = {3, 8, 11, 14, 17, 5}
    order = sorted(range(len(pos)), key=lambda i: -np.hypot(*pos[i]))   # 外圈先画，内圈压在上面
    for i in order:
        px, py = pos[i]
        jx, jy = rng.normal(0, 0.012, 2)
        islet_cell(ax, T, 0.5 + px + jx, 0.5 + py + jy, 0.104, "alpha" if i in alpha_idx else "beta",
                   seed=20 + i, detail=0, lw=0.55)


def single_cell(ax, x, y, s, H):
    """单个胰岛 β 细胞：细胞核、核仁、分泌颗粒、线粒体与内质网。"""
    islet_cell(ax, _unit(ax, x, y, s, H), 0.5, 0.5, 0.44, "beta", seed=3, detail=2, lw=0.9)


def patch(ax, x, y, s, H):
    """膜片钳：拉制玻璃电极从右侧接触细胞膜，电极内为银丝。"""
    T = _unit(ax, x, y, s, H)
    cx, cy, r = 0.36, 0.5, 0.31
    islet_cell(ax, T, cx, cy, r, "beta", seed=5, detail=2, lw=0.85)
    tip = cx + r * 0.97                                # 电极尖端轻压细胞膜（封接）
    xs = np.linspace(tip, 0.94, 40)
    hw = 0.022 + 0.058 * np.sqrt((xs - tip) / (0.94 - tip))       # 拉制电极：尖端细、向后逐渐变粗
    outer = np.r_[np.c_[xs, cy + hw], np.c_[xs[::-1], cy - hw[::-1]]]
    inner = np.r_[np.c_[xs[1:], cy + 0.6 * hw[1:]], np.c_[xs[:0:-1], cy - 0.6 * hw[:0:-1]]]
    ax.add_patch(Polygon(outer, closed=True, fc=GLASS_FILL, ec=GLASS_EDGE, lw=0.7, joinstyle="round",
                         alpha=0.95, transform=T))
    ax.add_patch(Polygon(inner, closed=True, fc="#F4F9FC", ec="none", transform=T))
    ax.plot([0.80, 0.94], [cy, cy], color=WIRE, lw=0.8, transform=T, solid_capstyle="round")


def reference(ax, x, y, s, H):
    """同一供体的一组细胞：六个大小略不同的 β 细胞围成一圈，虚线圆示意同一供体。"""
    T = _unit(ax, x, y, s, H)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.5 + 0.41 * np.cos(th), 0.53 + 0.41 * np.sin(th), color="#9AA4AE", lw=0.5, ls=(0, (2, 1.6)),
            transform=T)
    for k, (a, rr) in enumerate(zip(np.deg2rad(np.arange(90, 450, 60)), (0.15, 0.125, 0.14, 0.12, 0.145, 0.13))):
        islet_cell(ax, T, 0.5 + 0.25 * np.cos(a), 0.53 + 0.25 * np.sin(a), 0.88 * rr, "beta", seed=40 + k, detail=1,
                   lw=0.6)


def mrna(ax, x, y, s, H):
    """转录本：三条波浪形 RNA 链，带 5′ 帽（圆点）与 3′ poly(A) 尾（短竖线）。"""
    T = _unit(ax, x, y, s, H)
    t = np.linspace(0.08, 0.80, 120)
    for k, (yy, col) in enumerate(((0.72, "#2A9D8F"), (0.50, "#5FB8AC"), (0.28, "#2A9D8F"))):
        ax.plot(t, yy + 0.05 * np.sin(2 * np.pi * (t * 3.2 + k * 0.3)), color=col, lw=1.1, transform=T,
                solid_capstyle="round")
        ax.add_patch(Circle((0.07, yy + 0.05 * np.sin(2 * np.pi * (0.08 * 3.2 + k * 0.3))), 0.035, fc="#7B5EA7",
                            ec="none", transform=T))
        for j in range(4):
            ax.plot([0.83 + j * 0.035] * 2, [yy - 0.03, yy + 0.03], color="#9AA4AE", lw=0.7, transform=T)


def droplet_atlas(ax, x, y, s, H):
    """液滴单细胞测序：微流控通道中的液滴，部分包裹单个细胞。"""
    T = _unit(ax, x, y, s, H)
    ax.add_patch(Rectangle((0.02, 0.32), 0.96, 0.36, fc="#EEF3F7", ec="#9AA4AE", lw=0.6, transform=T))
    for k, cx in enumerate((0.14, 0.33, 0.52, 0.71, 0.88)):
        ax.add_patch(Circle((cx, 0.5), 0.085, fc="#F7FBFD", ec="#7A97AE", lw=0.6, transform=T))
        if k in (0, 2, 3):
            islet_cell(ax, T, cx, 0.5, 0.045, "beta" if k != 3 else "alpha", seed=60 + k, detail=0, lw=0.45)


def granule_membrane(ax, x, y, s, H):
    """质膜与胰岛素颗粒：膜下已停靠/引发的颗粒（早期释放）与深处向膜移动的颗粒（持续释放）；膜上 Ca2+ 通道。"""
    T = _unit(ax, x, y, s, H)
    ax.add_patch(Rectangle((0.0, 0.845), 1.0, 0.06, fc="#FBEAE3", ec="none", transform=T))      # 双层膜
    for yy in (0.845, 0.905):
        ax.plot([0.0, 0.58], [yy, yy], color="#C8644F", lw=0.7, transform=T)
        ax.plot([0.73, 1.0], [yy, yy], color="#C8644F", lw=0.7, transform=T)
    for x0 in (0.585, 0.665):                                                       # Ca2+ 通道：两个亚基围成孔道
        ax.add_patch(FancyBboxPatch((x0, 0.815), 0.05, 0.12, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="#2C6FB0", ec="#1F4E7A", lw=0.5, transform=T))
    ax.add_patch(Circle((0.66, 0.975), 0.018, fc="#2C6FB0", ec="none", transform=T))      # Ca2+ 离子
    ax.annotate("", xy=(0.66, 0.79), xytext=(0.66, 0.955), xycoords=T, textcoords=T,
                arrowprops=dict(arrowstyle="-|>", color="#2C6FB0", lw=0.6, mutation_scale=5))
    for cx in (0.18, 0.36, 0.84):
        ax.add_patch(Circle((cx, 0.74), 0.075, fc=GRAN_HALO, ec=GRAN_CORE, lw=0.6, transform=T))
        ax.add_patch(Circle((cx, 0.74), 0.035, fc=GRAN_CORE, ec="none", transform=T))
    for cx, cy in ((0.22, 0.30), (0.45, 0.20), (0.70, 0.34)):
        ax.add_patch(Circle((cx, cy), 0.075, fc=GRAN_HALO, ec=GRAN_CORE, lw=0.6, transform=T))
        ax.add_patch(Circle((cx, cy), 0.035, fc=GRAN_CORE, ec="none", transform=T))
        ax.annotate("", xy=(cx + 0.02, cy + 0.24), xytext=(cx, cy + 0.09), xycoords=T, textcoords=T,
                    arrowprops=dict(arrowstyle="-|>", color="#9AA4AE", lw=0.6, mutation_scale=5))


ICONS = {"donor": donor, "islet": islet, "cell": single_cell, "patch": patch, "reference": reference,
         "mrna": mrna, "droplet_atlas": droplet_atlas, "granules": granule_membrane}


# ------------------------------------------------------------------ 小图形（毫米坐标）
def expression_bars(ax, H, x, y_base, n, height, width, step, color, seed=4, lo=0.15):
    """表达谱条形图：n 根随机高度的竖条，底边在 y_base（毫米，向下为正）。"""
    for i, v in enumerate(np.random.default_rng(seed).uniform(lo, 1, n)):
        ax.add_patch(Rectangle((x + i * step, H - y_base), width, height * v, fc=color, ec="none", alpha=0.85))


def ephys_traces(ax, H, x, w, y_current, y_cap, red, blue):
    """电生理曲线：电压钳电流（快速内向钠电流 + 持续钙电流）与膜电容阶梯式增加（胞吐）。"""
    t = np.linspace(0, 1, 200)
    cur = -np.exp(-((t - 0.25) / 0.03) ** 2) * 0.9 - 0.35 * ((t > 0.3) & (t < 0.7))
    ax.plot(x + t * w, H - (y_current - cur * 4), color=red, lw=0.9)
    cap = np.cumsum(np.r_[0, np.diff((np.floor(t * 6)).astype(int)) > 0]) * 0.9
    ax.plot(x + t * w, H - (y_cap - cap * 1.0), color=blue, lw=0.9, drawstyle="steps-post")
