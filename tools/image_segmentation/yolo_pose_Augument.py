#!/usr/bin/env python3
"""
yolov8_pose_augment.py

对 yolo-pose 格式的数据进行光学/色彩类数据增强（不改变标签）并将增强后的图片与对应标签另存到目标文件夹。

增强包括：
 - Gaussian 模糊（可随机半径）
 - 模拟阳光（亮度渐变 / 光晕）
 - 模拟阴影（多边形或线性渐变遮罩）

使用说明（命令行）：
python yolov8_pose_augment.py \
    --src-images ./images \
    --src-labels ./labels \
    --dst ./augmented \
    --variants 3

依赖：
 - Python 3.8+
 - Pillow
 - numpy

安装：
pip install pillow numpy

注意：
 - 本脚本只做 photometric（光学/色彩）变换，不做几何变换，因此原始的 yolo-pose 标签文件（*.txt）无需修改，直接复制并重命名匹配增强后图片的文件名。
 - 如果某张图片没有对应的标签文件，会在日志中警告并仍保存增强图片（标签文件留空或跳过，可配置）。

"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


def list_images(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_contrast(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def apply_sunlight(img: Image.Image, strength: float = 0.6, center: Optional[Tuple[float, float]] = None, radius_ratio: float = 0.5) -> Image.Image:
    """
    模拟阳光：在图片某个区域叠加径向亮度渐变（暖黄色偏色），strength 控制叠加强度 [0..1]
    center: (x_frac, y_frac) 中心位置，若为 None 则随机在上半部分
    radius_ratio: 光晕半径占较短边的比例
    """
    img = img.convert('RGBA')
    w, h = img.size
    short = min(w, h)
    if center is None:
        cx = random.uniform(0.3, 0.7)
        cy = random.uniform(0.0, 0.35)  # 更偏上方模拟太阳
    else:
        cx, cy = center
    cx_px = int(cx * w)
    cy_px = int(cy * h)
    radius = int(radius_ratio * short)

    # 创建径向渐变掩码
    y, x = np.ogrid[0:h, 0:w]
    dist = np.sqrt((x - cx_px) ** 2 + (y - cy_px) ** 2)
    grad = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    # 平滑一下
    grad = np.power(grad, 1.2)

    # 颜色叠加（暖黄色）
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    # Warm color (R, G, B)
    warm = np.array([255, 240, 180], dtype=np.uint8)
    for i in range(3):
        overlay[:, :, i] = (warm[i] * (grad * strength)).astype(np.uint8)
    # Alpha通道为 grad*255*strength
    overlay[:, :, 3] = (grad * 255 * strength).astype(np.uint8)

    overlay_img = Image.fromarray(overlay, mode='RGBA')

    # 使用 alpha composite 叠加
    out = Image.alpha_composite(img, overlay_img)
    return out.convert('RGB')


def apply_shadow(img: Image.Image, strength: float = 0.6, n_polygons: int = 1) -> Image.Image:
    """
    模拟阴影：在图片上生成一个或多个随机多边形/椭圆遮罩并对区域进行变暗（multiply）。
    strength: 阴影深度 0..1（越大越暗）
    n_polygons: 阴影数量
    """
    img = img.convert('RGBA')
    w, h = img.size

    mask = Image.new('L', (w, h), color=0)
    draw = ImageDraw.Draw(mask)

    for _ in range(n_polygons):
        shape_type = random.choice(['polygon', 'ellipse'])
        if shape_type == 'polygon':
            # 随机生成多边形顶点，偏向边缘
            n_pts = random.randint(3, 6)
            pts = []
            for _ in range(n_pts):
                x = int(random.uniform(-0.1, 1.1) * w)
                y = int(random.uniform(0.0, 1.0) * h)
                pts.append((x, y))
            draw.polygon(pts, fill=int(255 * strength))
        else:
            # 椭圆
            x0 = int(random.uniform(-0.2, 0.8) * w)
            y0 = int(random.uniform(0.0, 0.6) * h)
            rw = int(random.uniform(0.2, 0.8) * w)
            rh = int(random.uniform(0.1, 0.6) * h)
            draw.ellipse([x0, y0, x0 + rw, y0 + rh], fill=int(255 * strength))

    # 将 mask 高斯模糊一下以软化边缘
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(8, min(w, h) // 50)))

    shadow_layer = Image.new('RGBA', (w, h), color=(0, 0, 0, 0))
    shadow_layer.putalpha(mask)

    # 将阴影以 multiply 的方式合并: 先将阴影转换为可被 alpha composite 的半透明黑色
    # 这里直接用 alpha_composite 将阴影叠到图片上，再降低亮度
    combined = Image.alpha_composite(img, shadow_layer)

    # 为了进一步暗化被遮罩区域，我们可以用 mask 对原图进行局部亮度降低
    orig = img.convert('RGB')
    combined_rgb = combined.convert('RGB')

    orig_np = np.array(orig).astype(np.float32) / 255.0
    comb_np = np.array(combined_rgb).astype(np.float32) / 255.0
    # mask_np [0,1]
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = np.expand_dims(mask_np, axis=2)

    # 线性混合：out = orig * (1 - mask*alpha) + darker * (mask*alpha)
    # 这里我们直接把被遮罩像素乘以 (1 - depth)
    depth = 0.5 + 0.45 * strength  # 调整感知
    out_np = orig_np * (1.0 - mask_np * depth)
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)

    out = Image.fromarray(out_np)
    return out


def random_augment(img: Image.Image, seed: Optional[int] = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)

    # Start from original
    out = img.copy()

    # 1) Gaussian blur 随机 radius
    if random.random() < 1.0:  # always apply some blur according to user's request
        radius = random.uniform(0.0, 2.5)  # 可调整范围
        if radius > 0.02:
            out = apply_gaussian_blur(out, radius=radius)

    # 2) Brightness/Contrast 随机
    bri = random.uniform(0.9, 1.25)
    con = random.uniform(0.9, 1.2)
    out = apply_brightness_contrast(out, brightness=bri, contrast=con)

    # 3) 随机决定是否模拟阳光（偏上方）
    if random.random() < 0.7:
        strength = random.uniform(0.2, 0.8)
        radius_ratio = random.uniform(0.25, 0.7)
        out = apply_sunlight(out, strength=strength, center=None, radius_ratio=radius_ratio)

    # 4) 随机决定是否添加阴影
    if random.random() < 0.65:
        strength = random.uniform(0.35, 0.85)
        n_poly = random.randint(1, 2)
        out = apply_shadow(out, strength=strength, n_polygons=n_poly)

    return out


def augment_dataset(src_images_dir: Path, src_labels_dir: Path, dst_dir: Path, variants: int = 3, keep_original: bool = True):
    ensure_dir(dst_dir)
    images = list_images(src_images_dir)
    if len(images) == 0:
        print(f"No images found in {src_images_dir}")
        return

    print(f"Found {len(images)} images. Generating {variants} variants per image into {dst_dir}")

    for img_path in images:
        stem = img_path.stem
        ext = img_path.suffix
        # read image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        # copy original if requested
        if keep_original:
            dst_img_path = dst_dir / f"{stem}_orig{ext}"
            shutil.copy2(img_path, dst_img_path)
            # copy label
            src_label = src_labels_dir / f"{stem}.txt"
            if src_label.exists():
                dst_label = dst_dir / f"{stem}_orig.txt"
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: label for {img_path.name} not found at {src_label}")

        # create variants
        for i in range(variants):
            seed = random.randint(0, 10_000_000)
            aug = random_augment(img, seed=seed)
            dst_img_path = dst_dir / f"{stem}_aug{i+1}{ext}"
            try:
                aug.save(dst_img_path)
            except Exception as e:
                print(f"Failed to save {dst_img_path}: {e}")
                continue

            # copy label (if exists) with same stem
            src_label = src_labels_dir / f"{stem}.txt"
            if src_label.exists():
                dst_label = dst_dir / f"{stem}_aug{i+1}.txt"
                try:
                    shutil.copy2(src_label, dst_label)
                except Exception as e:
                    print(f"Failed to copy label for {dst_img_path.name}: {e}")
            else:
                print(f"Warning: label for {img_path.name} not found at {src_label} (variant {i+1})")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='YOLO-pose photometric augmentation (labels unchanged).')
    parser.add_argument('--src-images', required=True, type=Path, help='源图片目录')
    parser.add_argument('--src-labels', required=False, type=Path, help='源标签目录，若省略则与 src-images 同目录', default=None)
    parser.add_argument('--dst', required=True, type=Path, help='输出目录')
    parser.add_argument('--variants', required=False, type=int, default=3, help='每张图生成的增强变体数量')
    parser.add_argument('--keep-original', action='store_true', help='在输出中保留原始图片与标签（加后缀 _orig）')

    args = parser.parse_args()

    src_images_dir = args.src_images
    src_labels_dir = args.src_labels if args.src_labels is not None else args.src_images
    dst_dir = args.dst

    if not src_images_dir.exists():
        print(f"src-images directory not found: {src_images_dir}")
        return
    if not src_labels_dir.exists():
        print(f"Warning: src-labels directory not found: {src_labels_dir} . Labels may be missing.")

    ensure_dir(dst_dir)

    augment_dataset(src_images_dir, src_labels_dir, dst_dir, variants=args.variants, keep_original=args.keep_original)


if __name__ == '__main__':
    main()
