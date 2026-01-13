#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_augment_yolov8_kp.py

对 YOLOv8 keypoint 数据集做阴影增强（不改变图像尺寸，复制对应标注文件）。
用法示例：
    python shadow_augment_yolov8_kp.py --input_dir ./images --output_dir ./aug_images --n_aug 3 \
        --min_shadows 1 --max_shadows 3 --modes polygon,ellipse,linear,rect

说明：
- 脚本只在像素层引入阴影（暗化），不做几何变换，所以 keypoint/bbox 标注无需修改。
- 依赖：opencv-python, numpy, tqdm
    pip install opencv-python numpy tqdm
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def _ensure_odd(k):
    return k if k % 2 == 1 else k + 1


def add_polygon_shadow(img: np.ndarray, alpha_range=(0.3, 0.7), max_vertices=8) -> np.ndarray:
    h, w = img.shape[:2]
    # 随机多边形顶点，靠近图片四周或内部
    n = random.randint(3, max_vertices)
    margin = int(min(w, h) * 0.05)
    pts = []
    for _ in range(n):
        x = random.randint(margin, w - margin)
        y = random.randint(margin, h - margin)
        pts.append([x, y])
    mask = np.zeros((h, w), dtype=np.float32)
    pts_np = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts_np, 1.0)
    # 用高斯模糊让阴影边缘柔和
    k = _ensure_odd(int(min(w, h) * 0.02))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    alpha = random.uniform(*alpha_range)
    # mask in [0,1], apply multiplicative darkening
    mask = mask * alpha
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * (1.0 - mask[..., None])
    out = (np.clip(img_f, 0, 1) * 255.0).astype(np.uint8)
    return out


def add_ellipse_shadow(img: np.ndarray, alpha_range=(0.25, 0.6)) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    # 随机椭圆中心、轴长与角度
    center = (random.randint(int(w * 0.2), int(w * 0.8)), random.randint(int(h * 0.2), int(h * 0.8)))
    axes = (random.randint(int(w * 0.1), int(w * 0.5)), random.randint(int(h * 0.1), int(h * 0.5)))
    angle = random.uniform(0, 180)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
    k = _ensure_odd(int(min(w, h) * 0.03))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    alpha = random.uniform(*alpha_range)
    mask = mask * alpha
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * (1.0 - mask[..., None])
    out = (np.clip(img_f, 0, 1) * 255.0).astype(np.uint8)
    return out


def add_linear_shadow(img: np.ndarray, alpha_range=(0.2, 0.6)) -> np.ndarray:
    # 生成沿某一方向的线性渐变阴影（如太阳被云挡）
    h, w = img.shape[:2]
    # 随机选择方向： 0=left->right,1=top->bottom,2=diag,3=anti-diag
    direction = random.choice([0, 1, 2, 3])
    if direction == 0:
        gradient = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
    elif direction == 1:
        gradient = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
    elif direction == 2:
        gx = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
        gy = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
        gradient = (gx + gy) / 2.0
    else:
        gx = np.linspace(1, 0, w)[None, :].repeat(h, axis=0)
        gy = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
        gradient = (gx + gy) / 2.0

    # 选取一个范围并可能翻转，使阴影只覆盖图像一部分
    start = random.uniform(0.0, 0.4)
    end = random.uniform(0.6, 1.0)
    gradient = np.clip((gradient - start) / (end - start + 1e-6), 0.0, 1.0)

    # 随机反转（使暗的在另一侧）
    if random.random() < 0.5:
        gradient = 1.0 - gradient

    # 添加噪声并模糊以模拟云层
    noise = (np.random.randn(h, w) * 0.1).astype(np.float32)
    mask = np.clip(gradient + noise, 0.0, 1.0)
    k = _ensure_odd(int(min(w, h) * 0.02))
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = random.uniform(*alpha_range)
    mask = mask * alpha
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * (1.0 - mask[..., None])
    out = (np.clip(img_f, 0, 1) * 255.0).astype(np.uint8)
    return out


def add_rect_shadow(img: np.ndarray, alpha_range=(0.25, 0.6), max_rects=3) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    rect_count = random.randint(1, max_rects)
    for _ in range(rect_count):
        rw = random.randint(int(w * 0.1), int(w * 0.6))
        rh = random.randint(int(h * 0.05), int(h * 0.3))
        x = random.randint(0, max(0, w - rw))
        y = random.randint(0, max(0, h - rh))
        cv2.rectangle(mask, (x, y), (x + rw, y + rh), 1.0, -1)
    k = _ensure_odd(int(min(w, h) * 0.03))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    alpha = random.uniform(*alpha_range)
    mask = mask * alpha
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * (1.0 - mask[..., None])
    out = (np.clip(img_f, 0, 1) * 255.0).astype(np.uint8)
    return out


SHADOW_FUNCS = {
    "polygon": add_polygon_shadow,
    "ellipse": add_ellipse_shadow,
    "linear": add_linear_shadow,
    "rect": add_rect_shadow,
}


def parse_modes(modes_str: str) -> List[str]:
    modes = [m.strip().lower() for m in modes_str.split(",") if m.strip()]
    valid = [m for m in modes if m in SHADOW_FUNCS]
    if not valid:
        raise ValueError(f"No valid modes found in '{modes_str}'. Valid: {list(SHADOW_FUNCS.keys())}")
    return valid


def augment_folder(input_dir: Path,
                   output_dir: Path,
                   img_ext_list: List[str],
                   label_ext: str,
                   n_aug: int,
                   modes: List[str],
                   min_shadows: int = 1,
                   max_shadows: int = 3,
                   seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in img_ext_list:
        img_paths.extend(sorted(input_dir.rglob(f"*{ext}")))

    if not img_paths:
        print(f"未在 {input_dir} 下找到指定扩展名的图片（{img_ext_list}）")
        return

    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
        except Exception as e:
            print(f"读取图片出错 {img_path}: {e}")
            continue

        base = img_path.stem
        label_path = img_path.with_suffix(label_ext)
        label_exists = label_path.exists()

        for i in range(n_aug):
            out_img = img.copy()
            # 随机阴影个数
            n_sh = random.randint(min_shadows, max_shadows)
            chosen_modes = [random.choice(modes) for _ in range(n_sh)]
            for m in chosen_modes:
                func = SHADOW_FUNCS[m]
                out_img = func(out_img)

            out_name = f"{base}_shadow_{i}{img_path.suffix}"
            out_img_path = output_dir / out_name
            # 保存（支持中文名）
            ext = img_path.suffix.lower()
            success, encimg = cv2.imencode(ext, out_img)
            if success:
                encimg.tofile(str(out_img_path))
            else:
                cv2.imwrite(str(out_img_path), out_img)

            # 复制标注文件（若存在）
            if label_exists:
                out_label_name = f"{base}_shadow_{i}{label_ext}"
                out_label_path = output_dir / out_label_name
                try:
                    shutil.copy2(str(label_path), str(out_label_path))
                except Exception as e:
                    print(f"复制标注失败 {label_path} -> {out_label_path}: {e}")

    print("增强完成。输出目录：", output_dir)


def main():
    parser = argparse.ArgumentParser(description="对 YOLOv8 keypoint 数据进行阴影增强（复制标注）")
    parser.add_argument("--input_dir", type=str, required=True, help="原始图片和标注所在目录")
    parser.add_argument("--output_dir", type=str, required=True, help="增强后保存目录")
    parser.add_argument("--img_exts", type=str, default=".jpg,.jpeg,.png", help="支持的图片扩展名，用逗号分隔")
    parser.add_argument("--label_ext", type=str, default=".txt", help="标注文件扩展名（通常 .txt）")
    parser.add_argument("--n_aug", type=int, default=2, help="每张图片生成的增强样本数量")
    parser.add_argument("--modes", type=str, default="polygon,ellipse,linear,rect",
                        help="选择哪些阴影方法（逗号分隔），可选: polygon, ellipse, linear, rect")
    parser.add_argument("--min_shadows", type=int, default=1, help="每张增强图最少的阴影数量")
    parser.add_argument("--max_shadows", type=int, default=3, help="每张增强图最多的阴影数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    img_ext_list = [e if e.startswith('.') else f".{e}" for e in args.img_exts.split(",")]
    label_ext = args.label_ext if args.label_ext.startswith('.') else f".{args.label_ext}"
    modes = parse_modes(args.modes)

    augment_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        img_ext_list=img_ext_list,
        label_ext=label_ext,
        n_aug=args.n_aug,
        modes=modes,
        min_shadows=args.min_shadows,
        max_shadows=args.max_shadows,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
