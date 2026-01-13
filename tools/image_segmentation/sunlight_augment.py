#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunlight_augment_yolov8_kp.py

对 YOLOv8 keypoint 数据集做模拟阳光光感增强（不改变图像尺寸，复制对应标注文件）。

用法示例：
    python sunlight_augment_yolov8_kp.py --input_dir ./images --output_dir ./aug_images --n_aug 3

依赖：
    pip install opencv-python numpy tqdm
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def safe_imread(path: Path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def save_img(path: Path, img: np.ndarray):
    ext = path.suffix.lower()
    success, enc = cv2.imencode(ext, img)
    if success:
        enc.tofile(str(path))
    else:
        cv2.imwrite(str(path), img)


def _ensure_odd(x):
    return int(x) if int(x) % 2 == 1 else int(x) + 1


# ---------- 基本光感组件 ----------
def create_sun_glow(h: int, w: int, center: Tuple[int, int], radius: int, intensity: float) -> np.ndarray:
    """创建太阳光晕 mask，范围 [0,1]"""
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1, lineType=cv2.LINE_AA)
    k = _ensure_odd(max(3, int(radius * 0.6)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask * intensity, 0.0, 1.0)
    return mask


def create_directional_gradient(h: int, w: int, angle_deg: float, strength: float) -> np.ndarray:
    """沿某方向的线性/非线性渐变，用于模拟侧光或顶光（返回 0..1）"""
    # 构造坐标
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    # 方向单位向量
    ang = np.deg2rad(angle_deg)
    vx, vy = np.cos(ang), np.sin(ang)
    proj = (xx - cx) * vx + (yy - cy) * vy
    # 归一化并转为 0..1
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    # 增强对比使效果更戏剧化
    mask = np.power(proj, 1.0 - strength)  # strength越大越接近线性
    # 反转概率，使暗在另一侧
    if random.random() < 0.5:
        mask = 1.0 - mask
    mask = np.clip(mask * strength, 0.0, 1.0)
    # 模糊边缘
    k = _ensure_odd(max(3, int(min(h, w) * 0.02)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def create_god_rays(h: int, w: int, center: Tuple[int, int], num_rays: int, length: float, intensity: float) -> np.ndarray:
    """创建光束（god rays）mask"""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = center
    max_r = int(max(h, w) * length)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    for a in angles:
        # 生成一条从中心向外的细线，线宽和强度随机
        dx = int(np.cos(a) * max_r)
        dy = int(np.sin(a) * max_r)
        x2, y2 = cx + dx, cy + dy
        thickness = max(1, int(random.uniform(1, max(3, min(h, w) * 0.005))))
        cv2.line(mask, (cx, cy), (int(x2), int(y2)), color=random.uniform(0.4, 1.0), thickness=thickness, lineType=cv2.LINE_AA)
    # 模糊并缩放强度
    k = _ensure_odd(max(3, int(min(h, w) * 0.04)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask * intensity, 0.0, 1.0)
    # 可随机应用径向衰减（离中心越弱）
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = 1.0 - (d / (np.max(d) + 1e-9))
    mask = mask * np.clip(radial ** random.uniform(1.0, 2.5), 0.0, 1.0)
    return mask


def create_lens_flares(h: int, w: int, center: Tuple[int, int], sun_center: Tuple[int, int], count: int, intensity: float) -> np.ndarray:
    """简易镜头鬼影：在太阳连线上产生多个小亮斑"""
    mask = np.zeros((h, w), dtype=np.float32)
    sx, sy = sun_center
    cx, cy = center
    # 向中心方向的向量
    vx, vy = cx - sx, cy - sy
    for i in range(1, count + 1):
        t = i / (count + 1)
        px = int(sx + vx * t + random.uniform(-0.05 * w, 0.05 * w))
        py = int(sy + vy * t + random.uniform(-0.05 * h, 0.05 * h))
        r = max(2, int(min(h, w) * random.uniform(0.01, 0.06)))
        cv2.circle(mask, (px, py), r, color=random.uniform(0.4, 1.0), thickness=-1, lineType=cv2.LINE_AA)
    k = _ensure_odd(max(3, int(min(h, w) * 0.03)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask * intensity, 0.0, 1.0)


# ---------- 合成函数 ----------
def apply_sunlight_effect(img: np.ndarray,
                          sun_pos_mode: str = "random",   # "random" 或 "corner" 或 "fixed"
                          strength: float = 0.6,
                          max_beams: int = 30,
                          add_flares: bool = True) -> np.ndarray:
    """
    对单张图片应用随机的阳光光感合成，返回增强后的图像（uint8）
    参数可调：sun_pos_mode 决定太阳中心位置，strength 控制总体强度
    """
    h, w = img.shape[:2]
    # 决定太阳位置
    if sun_pos_mode == "corner":
        # 从四角某处随机偏移
        corner = random.choice([(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)])
        sx = int(w * corner[0] + random.uniform(-0.05 * w, 0.05 * w))
        sy = int(h * corner[1] + random.uniform(-0.05 * h, 0.05 * h))
    elif sun_pos_mode == "fixed":
        sx = int(w * 0.9)
        sy = int(h * 0.1)
    else:
        sx = int(random.uniform(0.1 * w, 0.9 * w))
        sy = int(random.uniform(0.02 * h, 0.6 * h))  # 通常太阳在上半部分
    sun_center = (sx, sy)

    out = img.astype(np.float32) / 255.0

    # 1) 主光晕（glow）
    radius = int(random.uniform(0.06 * max(h, w), 0.4 * max(h, w)))
    glow_int = strength * random.uniform(0.6, 1.2)
    glow = create_sun_glow(h, w, sun_center, radius, glow_int)
    out = np.clip(out + glow[..., None], 0.0, 1.0)

    # 2) 方向性渐变（营造亮侧/反差）
    grad_strength = strength * random.uniform(0.2, 0.9)
    angle = random.uniform(-60, 60) + (0 if sx < w/2 else 180)  # 根据太阳左右调整主方向
    grad = create_directional_gradient(h, w, angle, grad_strength)
    out = np.clip(out + grad[..., None] * 0.6, 0.0, 1.0)

    # 3) 光束（god rays）可选
    if random.random() < 0.7:
        num_rays = random.randint(max(6, int(max_beams * 0.2)), max_beams)
        beams = create_god_rays(h, w, sun_center, num_rays, length=random.uniform(0.6, 1.5), intensity=0.2 * strength)
        out = np.clip(out + beams[..., None], 0.0, 1.0)

    # 4) 镜头鬼影（flares）
    if add_flares and random.random() < 0.6:
        center = (w // 2, h // 2)
        fl = create_lens_flares(h, w, center, sun_center, count=random.randint(1, 4), intensity=0.5 * strength)
        out = np.clip(out + fl[..., None], 0.0, 1.0)

    # 5) Bloom / 高光扩散（把图中非常亮的区域进一步扩散）
    gray = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_mask = np.clip((gray - random.uniform(0.6, 0.85)) / (1.0 - 0.6), 0.0, 1.0)
    bmask = cv2.GaussianBlur(bright_mask, (_ensure_odd(int(min(h, w) * 0.04)),) * 2, 0)
    out = np.clip(out + bmask[..., None] * random.uniform(0.1, 0.35) * strength, 0.0, 1.0)

    # 6) 微调色温和对比（让阳光更暖）
    # 转到HSV，轻微提升亮度和降低饱和度 / 调暖色调
    hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # 提升 V
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + 0.08 * strength * random.uniform(0.6, 1.2)), 0, 255)
    # 轻微增加色调偏向黄色/橙（向低H方向移动）
    hsv[..., 0] = (hsv[..., 0] - (2.0 * strength * random.uniform(0.5, 1.5))) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # 7) 最后轻微 gamma / 对比调整
    gamma = random.uniform(0.9, 1.12)
    out = np.clip(out ** (1.0 / gamma), 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return out


# ---------- 批处理与复制标注 ----------
def augment_folder(input_dir: Path,
                   output_dir: Path,
                   img_ext_list: List[str],
                   label_ext: str,
                   n_aug: int,
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
        img = safe_imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        base = img_path.stem
        label_path = img_path.with_suffix(label_ext)
        label_exists = label_path.exists()

        for i in range(n_aug):
            # 随机选择一些参数
            mode_choice = random.choice(["random", "corner", "fixed"])
            strength = random.uniform(0.35, 0.95)  # 总体强度
            max_beams = random.randint(10, 50)
            add_flares = random.random() < 0.8

            out_img = apply_sunlight_effect(img, sun_pos_mode=mode_choice, strength=strength, max_beams=max_beams, add_flares=add_flares)

            out_name = f"{base}_sun_{i}{img_path.suffix}"
            out_img_path = output_dir / out_name
            save_img(out_img_path, out_img)

            # 复制标注文件（若存在）
            if label_exists:
                out_label_name = f"{base}_sun_{i}{label_ext}"
                out_label_path = output_dir / out_label_name
                try:
                    shutil.copy2(str(label_path), str(out_label_path))
                except Exception as e:
                    print(f"复制标注失败 {label_path} -> {out_label_path}: {e}")

    print("增强完成。输出目录：", output_dir)


def main():
    parser = argparse.ArgumentParser(description="对 YOLOv8 keypoint 数据进行阳光光感增强（复制标注）")
    parser.add_argument("--input_dir", type=str, required=True, help="原始图片和标注所在目录")
    parser.add_argument("--output_dir", type=str, required=True, help="增强后保存目录")
    parser.add_argument("--img_exts", type=str, default=".jpg,.jpeg,.png", help="支持的图片扩展名，用逗号分隔")
    parser.add_argument("--label_ext", type=str, default=".txt", help="标注文件扩展名（通常 .txt）")
    parser.add_argument("--n_aug", type=int, default=2, help="每张图片生成的增强样本数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    img_ext_list = [e if e.startswith('.') else f".{e}" for e in args.img_exts.split(",")]
    label_ext = args.label_ext if args.label_ext.startswith('.') else f".{args.label_ext}"

    augment_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        img_ext_list=img_ext_list,
        label_ext=label_ext,
        n_aug=args.n_aug,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
