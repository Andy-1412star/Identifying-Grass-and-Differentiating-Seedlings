#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blur_augment_yolov8_kp.py

对 YOLOv8 keypoint 数据集做模糊增强（不改变图像尺寸，复制对应标注文件）。
用法示例：
    python blur_augment_yolov8_kp.py --input_dir ./images --output_dir ./aug_images --n_aug 3 --modes gaussian,median,motion
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from tqdm import tqdm


def random_gaussian_blur(img: np.ndarray, ksize_range=(3, 15)) -> np.ndarray:
    k = random.randrange(ksize_range[0], ksize_range[1] + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def random_average_blur(img: np.ndarray, ksize_range=(3, 11)) -> np.ndarray:
    k = random.randrange(ksize_range[0], ksize_range[1] + 1)
    if k % 2 == 0:
        k += 1
    return cv2.blur(img, (k, k))


def random_median_blur(img: np.ndarray, ksize_range=(3, 11)) -> np.ndarray:
    k = random.randrange(ksize_range[0], ksize_range[1] + 1)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def random_bilateral_filter(img: np.ndarray, d_range=(5, 15), sigma_range=(25, 75)) -> np.ndarray:
    d = random.randrange(d_range[0], d_range[1] + 1)
    sigma_color = random.randrange(sigma_range[0], sigma_range[1] + 1)
    sigma_space = random.randrange(sigma_range[0], sigma_range[1] + 1)
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def random_motion_blur(img: np.ndarray, kernel_size_range=(5, 25)) -> np.ndarray:
    k = random.randrange(kernel_size_range[0], kernel_size_range[1] + 1)
    if k < 3:
        k = 3
    # 随机选择方向
    direction = random.choice(['horizontal', 'vertical', 'diag', 'antidiag'])
    kernel = np.zeros((k, k))
    if direction == 'horizontal':
        kernel[k // 2, :] = np.ones(k)
    elif direction == 'vertical':
        kernel[:, k // 2] = np.ones(k)
    elif direction == 'diag':
        np.fill_diagonal(kernel, 1)
    else:
        kernel = np.fliplr(np.eye(k))
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


BLUR_FUNCS = {
    "gaussian": random_gaussian_blur,
    "average": random_average_blur,
    "median": random_median_blur,
    "bilateral": random_bilateral_filter,
    "motion": random_motion_blur,
}


def parse_modes(modes_str: str) -> List[str]:
    modes = [m.strip().lower() for m in modes_str.split(",") if m.strip()]
    valid = [m for m in modes if m in BLUR_FUNCS]
    if not valid:
        raise ValueError(f"No valid modes found in '{modes_str}'. Valid: {list(BLUR_FUNCS.keys())}")
    return valid


def augment_folder(input_dir: Path,
                   output_dir: Path,
                   img_ext_list: List[str],
                   label_ext: str,
                   n_aug: int,
                   modes: List[str],
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
                # fallback
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
        except Exception as e:
            print(f"读取图片出错 {img_path}: {e}")
            continue

        base = img_path.stem
        label_path = img_path.with_suffix(label_ext)
        if not label_path.exists():
            # 如果没有对应标注文件，仍然可以生成增强图像，但会给出提示（并且不会复制标注）
            label_exists = False
        else:
            label_exists = True

        # 也把原图（不变）复制到输出目录（可选）
        # shutil.copy2(str(img_path), str(output_dir / img_path.name))
        # if label_exists:
        #     shutil.copy2(str(label_path), str(output_dir / label_path.name))

        for i in range(n_aug):
            mode = random.choice(modes)
            func = BLUR_FUNCS[mode]
            aug_img = func(img)

            # 保存为与原图相同格式，文件名加入后缀
            out_name = f"{base}_blur_{mode}_{i}{img_path.suffix}"
            out_img_path = output_dir / out_name

            # 使用 imencode -> tofile 以支持中文路径/Windows
            ext = img_path.suffix.lower()
            # choose proper params for saving if needed
            success, encimg = cv2.imencode(ext, aug_img)
            if success:
                encimg.tofile(str(out_img_path))
            else:
                cv2.imwrite(str(out_img_path), aug_img)

            # 复制 label 文件（如果存在），并重命名为对应的增强图片同名 .txt（或指定扩展）
            if label_exists:
                out_label_name = f"{base}_blur_{mode}_{i}{label_ext}"
                out_label_path = output_dir / out_label_name
                try:
                    shutil.copy2(str(label_path), str(out_label_path))
                except Exception as e:
                    print(f"复制标注失败 {label_path} -> {out_label_path}: {e}")

    print("增强完成。输出目录：", output_dir)


def main():
    parser = argparse.ArgumentParser(description="对 YOLOv8 keypoint 数据进行模糊增强（复制标注）")
    parser.add_argument("--input_dir", type=str, required=True, help="原始图片和标注所在目录")
    parser.add_argument("--output_dir", type=str, required=True, help="增强后保存目录")
    parser.add_argument("--img_exts", type=str, default=".jpg,.jpeg,.png", help="支持的图片扩展名，用逗号分隔")
    parser.add_argument("--label_ext", type=str, default=".txt", help="标注文件扩展名（通常 .txt）")
    parser.add_argument("--n_aug", type=int, default=2, help="每张图片生成的增强样本数量")
    parser.add_argument("--modes", type=str, default="gaussian,median,motion,average,bilateral",
                        help="选择哪些模糊方法（逗号分隔），可选: gaussian, average, median, bilateral, motion")
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
        seed=args.seed
    )


if __name__ == "__main__":
    main()
