#!/usr/bin/env python3
"""
yolo_to_labelme.py

将 YOLO-pose（每张图对应 .txt）转换为单张图片对应的 LabelMe 风格 JSON 文件，
输出到 SJson/ 目录，格式尽量与用户示例一致。

假设每行常见格式之一：
 - class cx cy w h kx ky [v]    (归一化或像素)
 - class kx ky [v]              (仅关键点)
脚本对这些变体有容错处理。
"""

import os
import json
import cv2
from pathlib import Path

# -------- 配置 --------
IMAGES_DIR = "tiles/test_images"
LABELS_DIR = "tiles/test_labels"
OUT_DIR = "SJson"
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_BBOX_RATIO = 0.05   # 若无 bbox 则基于最小边生成小 bbox
RECT_LABEL = "corn"
POINT_LABEL = "CenterGrowthPoint"
LABELME_VERSION = "3.1.1"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- 工具函数 --------
def is_prob_val(v):
    return 0.0 <= v <= 1.01

def to_pixel(val, scale):
    """如果看起来像归一化值（0~1），则乘以 scale，否则返回原值（像素）"""
    return val * scale if is_prob_val(val) else val

def parse_yolo_pose_label(label_path, img_w, img_h, default_bbox_ratio=DEFAULT_BBOX_RATIO):
    """
    解析单个 label 文件，返回一个列表：每项为 dict 包含
    {'class','cx','cy','bw','bh','kx','ky','v'} （像素坐标）
    兼容格式：cx cy w h kx ky [v]、kx ky [v]、仅 bbox（会跳过）等。
    """
    anns = []
    if not os.path.exists(label_path):
        return anns

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        # 解析 class
        try:
            cls = int(float(parts[0]))
        except:
            # 如果第一列不是数字，跳过
            continue
        # 将后续 token 尽量解析为 float
        nums = []
        for tok in parts[1:]:
            try:
                nums.append(float(tok))
            except:
                pass

        # 常见： cx cy w h kx ky [v]
        if len(nums) >= 6:
            cx_raw, cy_raw, bw_raw, bh_raw, kx_raw, ky_raw = nums[:6]
            v = int(nums[6]) if len(nums) >= 7 else 2
            cx = to_pixel(cx_raw, img_w)
            cy = to_pixel(cy_raw, img_h)
            bw = to_pixel(bw_raw, img_w)
            bh = to_pixel(bh_raw, img_h)
            kx = to_pixel(kx_raw, img_w)
            ky = to_pixel(ky_raw, img_h)
            anns.append({'class': cls, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh, 'kx': kx, 'ky': ky, 'v': int(v)})
            continue

        # 只有两个数：当作 kx ky
        if len(nums) == 2:
            kx_raw, ky_raw = nums
            kx = to_pixel(kx_raw, img_w)
            ky = to_pixel(ky_raw, img_h)
            min_side = min(img_w, img_h)
            bw = bh = max(1.0, min_side * default_bbox_ratio)
            cx = kx
            cy = ky
            anns.append({'class': cls, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh, 'kx': kx, 'ky': ky, 'v': 2})
            continue

        # 三个数：当作 kx ky v
        if len(nums) == 3:
            kx_raw, ky_raw, v_raw = nums
            kx = to_pixel(kx_raw, img_w)
            ky = to_pixel(ky_raw, img_h)
            v = int(v_raw)
            min_side = min(img_w, img_h)
            bw = bh = max(1.0, min_side * default_bbox_ratio)
            cx = kx
            cy = ky
            anns.append({'class': cls, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh, 'kx': kx, 'ky': ky, 'v': v})
            continue

        # 4个数（bbox only）或其他不支持的格式：跳过
        # (你要求关键点用于可视化，因此 bbox-only 无法生成 point)
        continue

    return anns

def make_labelme_json_dict(img_name, img_w, img_h, anns):
    """
    生成类似 LabelMe 的 JSON 字典，尽量匹配你提供的示例结构。
    anns: 列表，每项 {'class','cx','cy','bw','bh','kx','ky','v'}
    """
    shapes = []
    for a in anns:
        # 计算矩形两点 (x_min,y_min) 和 (x_max,y_max)
        x_min = float(a['cx'] - a['bw']/2.0)
        y_min = float(a['cy'] - a['bh']/2.0)
        x_max = float(a['cx'] + a['bw']/2.0)
        y_max = float(a['cy'] + a['bh']/2.0)
        # 矩形 shape（与你的示例相同字段）
        rect_shape = {
            "label": RECT_LABEL,
            "points": [
                [x_min, y_min],
                [x_max, y_max]
            ],
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(rect_shape)

        # point shape （更详细字段，模拟你示例里 point 的结构）
        point_shape = {
            "label": POINT_LABEL,
            "score": None,
            "points": [
                [float(a['kx']), float(a['ky'])]
            ],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "point",
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }
        shapes.append(point_shape)

    lm = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w
    }
    return lm

# -------- 主流程 --------
def main():
    img_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXTS)]
    if not img_files:
        print("未找到 images/ 下的图片，请检查目录。")
        return

    total = 0
    for img_file in img_files:
        img_path = os.path.join(IMAGES_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取图片 {img_path}，跳过。")
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(LABELS_DIR, Path(img_file).stem + ".txt")
        anns = parse_yolo_pose_label(label_path, w, h)
        # 如果没有任何有效标注，也可以选择仍生成空 json 或跳过；这里我们会生成空 shapes 的 json
        lm_dict = make_labelme_json_dict(img_file, w, h, anns)
        out_path = os.path.join(OUT_DIR, Path(img_file).stem + ".json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(lm_dict, f, ensure_ascii=False, indent=2)
        total += 1

    print(f"完成：为 {total} 张图片生成 LabelMe 风格 JSON，保存到 {OUT_DIR}/")

if __name__ == "__main__":
    main()
