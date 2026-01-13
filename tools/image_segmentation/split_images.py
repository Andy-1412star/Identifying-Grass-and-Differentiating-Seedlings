#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_and_copy_samples.py

从数据集中随机抽取若干图片并把它们及对应的标注文件分别复制到多个目标文件夹下。

主要功能：
 - 在输入目录中搜索图片（支持多种扩展名）
 - 随机抽取 total_samples 张（可设置随机种子以复现）
 - 将抽取到的样本均匀分配到 num_folders 个子目录，每个子目录下创建 images/ 和 labels/ 子文件夹
 - 复制图片和对应的标注（按同名 stem + label_ext 寻找），可以选择在缺失标注时生成空文件或仅警告
 - 生成一个 CSV 清单，记录每个被抽取样本的原始路径与目标路径以及分配的文件夹编号

用法示例：
    python split_and_copy_samples.py \
        --input_dir ./dataset/images \
        --output_dir ./sampled \
        --total_samples 800 \
        --num_folders 4 \
        --label_ext .txt \
        --seed 42

依赖：Python 3.7+（内置库即可），可选：pip install tqdm
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List
import csv

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x


IMAGE_EXTS_DEFAULT = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def find_images(input_dir: Path, exts: List[str]) -> List[Path]:
    imgs = []
    for e in exts:
        imgs.extend(sorted(input_dir.rglob(f"*{e}")))
    return imgs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_sample(image_path: Path, label_path: Path, dest_image: Path, dest_label: Path, create_empty_label: bool):
    shutil.copy2(str(image_path), str(dest_image))
    if label_path and label_path.exists():
        shutil.copy2(str(label_path), str(dest_label))
    else:
        if create_empty_label:
            # 写入空行以表示存在该文件
            dest_label.write_text("", encoding='utf-8')


def distribute_counts(total: int, parts: int) -> List[int]:
    base = total // parts
    rem = total % parts
    counts = [base + (1 if i < rem else 0) for i in range(parts)]
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, required=True, help='图片来源目录（会递归搜索图片文件）')
    p.add_argument('--label_dir', type=str, default=None, help='标注文件目录（默认与 input_dir 相同）')
    p.add_argument('--output_dir', type=str, required=True, help='输出根目录，会在其中创建 folder_1 ... folder_N')
    p.add_argument('--total_samples', type=int, default=800, help='要随机抽取的样本总数，默认 800')
    p.add_argument('--num_folders', type=int, default=4, help='要分配到的文件夹数量，默认 4')
    p.add_argument('--img_exts', type=str, default=','.join(IMAGE_EXTS_DEFAULT),
                   help='支持的图片扩展名，用逗号分隔，例如 .jpg,.png')
    p.add_argument('--label_ext', type=str, default='.txt', help='标注文件扩展名，默认 .txt')
    p.add_argument('--seed', type=int, default=None, help='随机种子（可选，便于复现）')
    p.add_argument('--create_empty_labels', action='store_true', help='若标注缺失则在目标位置创建空文件（否则仅警告）')
    p.add_argument('--symlink', action='store_true', help='使用符号链接而不是复制（在同一文件系统下有效）')
    p.add_argument('--manifest', type=str, default='manifest.csv', help='输出的清单文件名（CSV）')

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir) if args.label_dir else input_dir
    output_dir = Path(args.output_dir)
    total = int(args.total_samples)
    num_folders = int(args.num_folders)
    img_exts = [e.strip().lower() if e.startswith('.') else f".{e.strip().lower()}" for e in args.img_exts.split(',')]
    label_ext = args.label_ext if args.label_ext.startswith('.') else f".{args.label_ext}"

    if args.seed is not None:
        random.seed(args.seed)

    # 收集图片
    images = find_images(input_dir, img_exts)
    if not images:
        print(f'在 {input_dir} 下未找到任何图片，支持后缀：{img_exts}')
        return

    if total > len(images):
        print(f'要求抽取 {total} 张，但只找到 {len(images)} 张图片。请减小 total_samples 或提供更多图片。')
        return

    # 随机抽样
    selected = random.sample(images, total)

    counts = distribute_counts(total, num_folders)
    print(f'抽取总数: {total}，分配到 {num_folders} 个文件夹，分配方案: {counts}')

    # 创建输出子目录
    ensure_dir(output_dir)
    folder_paths = []
    for i in range(num_folders):
        folder = output_dir / f'folder_{i+1}'
        imgs_sub = folder / 'images'
        labels_sub = folder / 'labels'
        ensure_dir(imgs_sub)
        ensure_dir(labels_sub)
        folder_paths.append((folder, imgs_sub, labels_sub))

    # 开始复制，并写清单
    manifest_path = output_dir / args.manifest
    with manifest_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original_image', 'original_label', 'dest_image', 'dest_label', 'folder_index'])

        idx = 0
        for folder_index, cnt in enumerate(counts):
            _, imgs_sub, labels_sub = folder_paths[folder_index]
            for _ in range(cnt):
                img_path = selected[idx]
                stem = img_path.stem
                # 在 label_dir 中查找对应标注
                label_path_candidate = label_dir / (stem + label_ext)
                if not label_path_candidate.exists():
                    # 尝试在图片同目录查找（有些数据集标签与图片不在同一目录）
                    alt = img_path.with_suffix(label_ext)
                    if alt.exists():
                        label_path = alt
                    else:
                        label_path = None
                else:
                    label_path = label_path_candidate

                dest_img = imgs_sub / img_path.name
                dest_lbl = labels_sub / (stem + label_ext)

                # 复制或创建链接
                try:
                    if args.symlink:
                        # 如果目标已存在，跳过
                        if not dest_img.exists():
                            dest_img.symlink_to(img_path.resolve())
                        if label_path and not dest_lbl.exists():
                            dest_lbl.symlink_to(label_path.resolve())
                        elif (not label_path) and args.create_empty_labels and (not dest_lbl.exists()):
                            dest_lbl.write_text('', encoding='utf-8')
                    else:
                        copy_sample(img_path, label_path, dest_img, dest_lbl, args.create_empty_labels)
                except Exception as e:
                    print(f'复制出错: {img_path} -> {dest_img} ; 错误: {e}')

                writer.writerow([str(img_path), str(label_path) if label_path else '', str(dest_img), str(dest_lbl), folder_index+1])

                idx += 1

    print('抽样并分配完成。输出目录:', output_dir)
    print('清单文件:', manifest_path)


if __name__ == '__main__':
    main()
