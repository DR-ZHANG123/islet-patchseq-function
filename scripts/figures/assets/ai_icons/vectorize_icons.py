"""一次性：把本目录的 PNG 图标转为矢量路径（vtracer 0.6；需单独安装，出图时不需要）。

先把透明度二值化（alpha < 110 视为背景），去掉边缘的半透明杂点，再按颜色分层描摹为三次贝塞尔路径，
输出 vector/<name>.svg（只含 M/C/Z 绝对坐标路径与填充色）；schematic.Canvas.vector_icon 读取这些 SVG 绘制。
用法：python vectorize_icons.py
"""
from pathlib import Path

import numpy as np
import vtracer
from PIL import Image

HERE = Path(__file__).resolve().parent


def main() -> None:
    out = HERE / "vector"
    out.mkdir(exist_ok=True)
    for png in sorted(HERE.glob("*.png")):
        im = np.array(Image.open(png).convert("RGBA"))
        bg = im[..., 3] < 110
        im[..., 3] = np.where(bg, 0, 255)
        im[bg, :3] = 255
        tmp = out / f"{png.stem}_prep.png"
        Image.fromarray(im).save(tmp)
        vtracer.convert_image_to_svg_py(str(tmp), str(out / f"{png.stem}.svg"), colormode="color",
                                        hierarchical="stacked", mode="spline", filter_speckle=6, color_precision=7,
                                        layer_difference=8, corner_threshold=60, length_threshold=4.0,
                                        max_iterations=10, splice_threshold=45, path_precision=3)
        tmp.unlink()
        print(png.stem)


if __name__ == "__main__":
    main()
