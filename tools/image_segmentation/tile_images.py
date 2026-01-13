#!/usr/bin/env python3
"""
tile_images.py

将一个文件夹里的大图片拆分为 640 x 604 的小图片并保存到另一个文件夹。

用法：
    python tile_images.py --input_dir /path/to/input --output_dir /path/to/output
    （可以加参数 --tile_w 和 --tile_h 改尺寸）
"""

import os
import argparse
from PIL import Image, ImageOps

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def is_image_file(fname):
    return os.path.splitext(fname.lower())[1] in SUPPORTED_EXT


def tile_image(image_path, out_dir, tile_w=640, tile_h=604, pad=False, prefix=None):
    img = Image.open(image_path)
    # 自动根据 EXIF 方向调整（避免旋转问题）
    img = ImageOps.exif_transpose(img)
    W, H = img.size

    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(os.path.basename(image_path))[1].lower().lstrip('.')
    if prefix:
        base = f"{prefix}_{base}"

    rows = (H + tile_h - 1) // tile_h  # 向上取整
    cols = (W + tile_w - 1) // tile_w

    saved = 0
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = min(left + tile_w, W)
            lower = min(upper + tile_h, H)

            tile = img.crop((left, upper, right, lower))

            if pad and (tile.size[0] != tile_w or tile.size[1] != tile_h):
                # 用黑色背景填充到准确尺寸（也可改颜色）
                padded = Image.new('RGB', (tile_w, tile_h), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            # 输出文件名包含行列和像素区域，便于追溯
            out_name = f"{base}_r{r:03d}_c{c:03d}_x{left}_y{upper}_w{right - left}_h{lower - upper}.{ext}"
            out_path = os.path.join(out_dir, out_name)
            # 为保证 RGB 保存正常，convert('RGB') 防止透明通道出错（png）
            if tile.mode in ("RGBA", "LA") or (tile.mode == "P" and ext in ('jpg', 'jpeg')):
                tile = tile.convert("RGB")
            tile.save(out_path)
            saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser(description="Split images into tiles (640x604 default).")
    parser.add_argument("--input_dir", "-i", required=True, help="输入图片文件夹（只遍历该目录，不递归子目录）")
    parser.add_argument("--output_dir", "-o", required=True, help="切片输出目录（不存在则创建）")
    parser.add_argument("--tile_w", type=int, default=640, help="切片宽度（默认 640）")
    parser.add_argument("--tile_h", type=int, default=604, help="切片高度（默认 604）")
    parser.add_argument("--pad", action="store_true", help="对边缘不足大小的 tile 用黑色填充到目标尺寸")
    parser.add_argument("--prefix", type=str, default=None, help="输出文件名前缀（可选）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if is_image_file(f)]
    if not files:
        print("输入目录中没有支持的图片文件（jpg/png/tif...）")
        return

    total_tiles = 0
    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        try:
            n = tile_image(in_path, args.output_dir, args.tile_w, args.tile_h, pad=args.pad, prefix=args.prefix)
            print(f"[OK] {fname} -> {n} tiles")
            total_tiles += n
        except Exception as e:
            print(f"[ERR] 处理 {fname} 时出错: {e}")

    print(f"完成，总计保存 {total_tiles} 个切片到 {args.output_dir}")


if __name__ == "__main__":
    main()
