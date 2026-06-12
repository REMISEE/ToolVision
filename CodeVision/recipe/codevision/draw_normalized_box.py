#!/usr/bin/env python
"""
将归一化检测框画回原图。

默认输入框格式为 Gemini 常见的:
  [ymin, xmin, ymax, xmax]
且坐标按 0-1000 归一化。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_box(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--box 需要 4 个值，当前为: {raw}")
    return [float(x) for x in parts]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalized_to_xyxy(
    box_vals: list[float],
    width: int,
    height: int,
    scale: float = 1000.0,
    box_format: str = "yxyx",
) -> tuple[int, int, int, int]:
    if box_format == "yxyx":
        ymin, xmin, ymax, xmax = box_vals
    elif box_format == "xyxy":
        xmin, ymin, xmax, ymax = box_vals
    else:
        raise ValueError(f"不支持的 box_format: {box_format}")

    x1 = xmin / scale * width
    y1 = ymin / scale * height
    x2 = xmax / scale * width
    y2 = ymax / scale * height

    x1 = int(round(clamp(x1, 0, width - 1)))
    y1 = int(round(clamp(y1, 0, height - 1)))
    x2 = int(round(clamp(x2, 0, width - 1)))
    y2 = int(round(clamp(y2, 0, height - 1)))

    # 保证是有效框
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将归一化 bbox 还原并画到原图。")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument(
        "--box",
        type=str,
        required=True,
        help='归一化框，逗号分隔；默认格式 yxyx，例如 "544,822,563,843"',
    )
    parser.add_argument(
        "--box-format",
        type=str,
        default="yxyx",
        choices=["yxyx", "xyxy"],
        help="输入框格式，默认 yxyx",
    )
    parser.add_argument("--scale", type=float, default=1000.0, help="归一化尺度，默认 1000")
    parser.add_argument("--color", type=str, default="red", help="框颜色，默认 red")
    parser.add_argument("--width", type=int, default=3, help="线宽，默认 3")
    parser.add_argument("--label", type=str, default="", help="可选标签文本")
    parser.add_argument("--out", type=str, default="", help="输出图片路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    box_vals = parse_box(args.box)
    x1, y1, x2, y2 = normalized_to_xyxy(
        box_vals=box_vals,
        width=w,
        height=h,
        scale=args.scale,
        box_format=args.box_format,
    )

    draw = ImageDraw.Draw(img)
    draw.rectangle((x1, y1, x2, y2), outline=args.color, width=args.width)
    if args.label:
        draw.text((x1 + 2, max(0, y1 - 14)), args.label, fill=args.color)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = image_path.with_name(f"{image_path.stem}.boxed{image_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    print(f"image_size = ({w}, {h})")
    print(f"input_box(normalized, {args.box_format}, scale={args.scale}) = {box_vals}")
    print(f"pixel_box(xyxy) = [{x1}, {y1}, {x2}, {y2}]")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()

