#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotate_augment_yolov8_kp.py

对 YOLOv8 keypoint 数据做随机顺时针旋转（90, 180, 270 deg），并同步更新标注文件。

用途示例:
    python rotate_augment_yolov8_kp.py --input_dir ./images --output_dir ./aug --n_aug 2

说明:
 - 支持两种行内标注格式：
    1) class x_center y_center width height kx1 ky1 kv1 kx2 ky2 kv2 ...
    2) class kx1 ky1 kv1 kx2 ky2 kv2 ...   （若你的标注没有 bbox）
 - 坐标按 YOLO 规范为归一化 (0..1)。脚本在内部用归一化坐标进行变换，输出仍为归一化。
 - 对旋转 90/270，图片尺寸会交换（w<->h），脚本处理好了归一化坐标的变换。
 - 依赖: opencv-python, numpy, tqdm
    pip install opencv-python numpy tqdm
"""

import argparse
import random
from pathlib import Path
import shutil
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


ROT_MAP = {
    90: ("rot90", cv2.ROTATE_90_CLOCKWISE),
    180: ("rot180", cv2.ROTATE_180),
    270: ("rot270", cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def safe_imread(p: Path):
    arr = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def safe_imwrite(p: Path, img: np.ndarray):
    ext = p.suffix.lower()
    success, enc = cv2.imencode(ext, img)
    if success:
        enc.tofile(str(p))
    else:
        cv2.imwrite(str(p), img)


# --- normalized coordinate transforms ---
def rotate_point_norm(x: float, y: float, rotation_deg: int) -> Tuple[float, float]:
    """
    输入/输出均为归一化坐标 (x in [0,1], y in [0,1]).
    rotation_deg: 90, 180, 270 (clockwise)
    返回 (new_x, new_y) 也为归一化坐标，相对于旋转后图片的尺寸。
    变换规则（推导）:
      - 90° CW: (x', y') = (1 - y, x)
      - 180°:   (x', y') = (1 - x, 1 - y)
      - 270° CW:(x', y') = (y, 1 - x)
    """
    if rotation_deg == 90:
        return 1.0 - y, x
    elif rotation_deg == 180:
        return 1.0 - x, 1.0 - y
    elif rotation_deg == 270:
        return y, 1.0 - x
    else:
        raise ValueError("rotation_deg must be one of {90,180,270}")


def rotate_bbox_norm(cx: float, cy: float, bw: float, bh: float, rotation_deg: int) -> Tuple[float,float,float,float]:
    """
    以归一化坐标为输入（中心 cx,cy 和宽高 bw,bh），将 bbox 的四个角变换后
    返回新的 (cx, cy, bw, bh)（相对于旋转后图片的归一化坐标）。
    """
    x_min = cx - bw / 2.0
    x_max = cx + bw / 2.0
    y_min = cy - bh / 2.0
    y_max = cy + bh / 2.0

    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    rotated = [rotate_point_norm(x, y, rotation_deg) for (x, y) in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    nx_min, nx_max = min(xs), max(xs)
    ny_min, ny_max = min(ys), max(ys)
    ncx = (nx_min + nx_max) / 2.0
    ncy = (ny_min + ny_max) / 2.0
    nbw = nx_max - nx_min
    nbh = ny_max - ny_min
    # clamp to [0,1]
    ncx = float(np.clip(ncx, 0.0, 1.0))
    ncy = float(np.clip(ncy, 0.0, 1.0))
    nbw = float(np.clip(nbw, 0.0, 1.0))
    nbh = float(np.clip(nbh, 0.0, 1.0))
    return ncx, ncy, nbw, nbh


def rotate_keypoints_list(kps: List[Tuple[float,float,float]], rotation_deg: int) -> List[Tuple[float,float,float]]:
    """
    kps: list of (x_norm, y_norm, v) triples. 返回旋转后的 (x,y,v) 列表 (v 原样保留)。
    """
    out = []
    for (x, y, v) in kps:
        nx, ny = rotate_point_norm(x, y, rotation_deg)
        # clamp just in case
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        out.append((nx, ny, v))
    return out


# --- parsing / formatting helpers ---
def parse_label_line(line: str):
    """
    支持两种常见变体：
     - 带 bbox: class cx cy w h kx ky kv ...
     - 无 bbox:  class kx ky kv ...
    返回 (class_id_str, has_bbox_bool, bbox_tuple_or_None, kps_list)
    bbox_tuple = (cx, cy, w, h) with floats
    kps_list = list of (x,y,v) floats
    """
    toks = line.strip().split()
    if len(toks) == 0:
        return None
    cls = toks[0]
    nums = toks[1:]
    numsf = [float(x) for x in nums] if len(nums) > 0 else []
    # case A: have at least 4 numbers and remainder divisible by 3 -> treat as bbox + kps
    if len(numsf) >= 4 and ((len(numsf) - 4) % 3 == 0):
        cx, cy, bw, bh = numsf[0:4]
        kps_vals = numsf[4:]
        has_bbox = True
    else:
        # try interpret as only keypoints
        if len(numsf) % 3 == 0:
            cx = cy = bw = bh = None
            kps_vals = numsf
            has_bbox = False
        else:
            # unknown format -> raise
            raise ValueError(f"无法解析标签行（不是标准的 bbox+kp 或 仅 kp 格式）: {line}")
    kps = []
    for i in range(0, len(kps_vals), 3):
        kx = kps_vals[i]
        ky = kps_vals[i+1]
        kv = kps_vals[i+2]
        kps.append((float(kx), float(ky), float(kv)))
    bbox = (cx, cy, bw, bh) if has_bbox else None
    return cls, has_bbox, bbox, kps


def format_label_line(cls: str, has_bbox: bool, bbox, kps: List[Tuple[float,float,float]]) -> str:
    parts = [str(cls)]
    if has_bbox:
        cx, cy, bw, bh = bbox
        parts += [f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for (x,y,v) in kps:
        # keep visibility as integer if close to int
        if float(v).is_integer():
            v_str = str(int(v))
        else:
            v_str = f"{v:.6f}"
        parts += [f"{x:.6f}", f"{y:.6f}", v_str]
    return " ".join(parts)


def process_single_label_file(lbl_path: Path, rotation_deg: int) -> List[str]:
    """
    读取一个 .txt 标签文件，返回旋转后的文本行列表（待写入新文件）。
    """
    out_lines = []
    text = lbl_path.read_text(encoding='utf-8').strip().splitlines()
    for line in text:
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls, has_bbox, bbox, kps = parsed
        new_kps = rotate_keypoints_list(kps, rotation_deg)
        new_bbox = None
        if has_bbox:
            cx, cy, bw, bh = bbox
            new_bbox = rotate_bbox_norm(cx, cy, bw, bh, rotation_deg)
        out_lines.append(format_label_line(cls, has_bbox, new_bbox, new_kps))
    return out_lines


# --- main processing ---
def augment_folder(input_dir: Path,
                   output_dir: Path,
                   img_exts: List[str],
                   label_ext: str,
                   n_aug: int = 1,
                   seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for ext in img_exts:
        img_paths.extend(sorted(input_dir.rglob(f"*{ext}")))

    if not img_paths:
        print(f"在 {input_dir} 下未找到图片，支持后缀：{img_exts}")
        return

    for img_path in tqdm(img_paths, desc="Processing images"):
        img = safe_imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        h, w = img.shape[:2]
        base = img_path.stem
        lbl_path = img_path.with_suffix(label_ext)
        label_exists = lbl_path.exists()

        for i in range(n_aug):
            rot = random.choice([90, 180, 270])  # 随机顺时针旋转选择
            suffix, cv_flag = ROT_MAP[rot]
            out_img = cv2.rotate(img, cv_flag)

            out_img_name = f"{base}_{suffix}{img_path.suffix}"
            out_img_path = output_dir / out_img_name
            safe_imwrite(out_img_path, out_img)

            # 处理并保存标注
            if label_exists:
                try:
                    new_lines = process_single_label_file(lbl_path, rot)
                    out_lbl_name = f"{base}_{suffix}{label_ext}"
                    out_lbl_path = output_dir / out_lbl_name
                    out_lbl_path.write_text("\n".join(new_lines) + ("\n" if len(new_lines)>0 else ""), encoding='utf-8')
                except Exception as e:
                    print(f"处理标注时出错: {lbl_path} -> {e}")

            else:
                # 若无标注，则可选择跳过或复制空文件（这里我们不写标注）
                pass

    print("旋转增强完成，输出目录：", output_dir)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 keypoint：随机顺时针旋转 90/180/270 并更新标注")
    parser.add_argument("--input_dir", type=str, required=True, help="原始图片目录（图片与 .txt 同目录）")
    parser.add_argument("--output_dir", type=str, required=True, help="增强后输出目录")
    parser.add_argument("--img_exts", type=str, default=".jpg,.jpeg,.png", help="图片扩展名列表，用逗号分隔")
    parser.add_argument("--label_ext", type=str, default=".txt", help="标注扩展名（通常 .txt）")
    parser.add_argument("--n_aug", type=int, default=1, help="每张图片生成多少个随机旋转样本（默认1）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    img_exts = [e if e.startswith('.') else f".{e}" for e in args.img_exts.split(",")]
    label_ext = args.label_ext if args.label_ext.startswith('.') else f".{args.label_ext}"

    augment_folder(input_dir, output_dir, img_exts, label_ext, n_aug=args.n_aug, seed=args.seed)


if __name__ == "__main__":
    main()
