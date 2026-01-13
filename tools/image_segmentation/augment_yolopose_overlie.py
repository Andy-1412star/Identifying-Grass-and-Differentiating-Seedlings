#!/usr/bin/env python3
"""
augment_yolopose_mosaic_noblack.py

功能摘要：
- 从 images/ 随机选 4 张图做 2x2 mosaic（可缩放到均一 cell）
- mosaic 支持随机 overlap（覆盖），覆盖比例可配置
- 从 mosaic 在“内容区域内”随机裁 NUM_RANDOM_CROPS 张 1024x1024（避免黑边）
- 若内容区域小于裁剪尺寸，可等比放大 mosaic（及标注）以满足裁剪
- 对 mosaic 随机旋转后再裁 NUM_RANDOM_CROPS 张（同样保证裁剪在内容内）
- 关键点（dim=2 或 dim=3）与 bbox 做相应变换并写出 YOLO-pose 格式 label
"""
import os
import cv2
import numpy as np
import random
from pathlib import Path

# ----------------- 配置区（按需修改） -----------------
IMAGES_DIR = Path("new_datasets/images")
LABELS_DIR = Path("new_datasets/labels")
OUT_IMAGES = Path("outCorn/images")   # 输出图片文件夹
OUT_LABELS = Path("outCorn/labels")   # 输出 label 文件夹
CROP_SIZE = 1024
NUM_RANDOM_CROPS = 3
RANDOM_SEED = 42

# mosaic 参数
MAX_OVERLAP_RATIO = 0.3    # 每张图最大覆盖比例（相对于 cell 大小）
SCALE_TO_UNIFORM_CELL = True  # 是否将四张图缩放到统一 cell 大小（更均匀）

# 旋转参数
ROTATE_ANGLE_RANGE = (-45, 45)  # 随机角度范围

# crop 内容策略
ALLOW_SCALE_UP = True     # 若 mosaic 内容区域小于 crop，是否允许等比放大 mosaic（默认允许）
MAX_SCALE_UP = 4.0        # 最大放大倍数上限
SAVE_EMPTY_CROPS = False  # 若为 True，即使没有对象也保存（debug）；默认 False（避免空图）
KEEP_IF_BBOX_OVERLAP = True
BBOX_IOU_THRESHOLD = 0.02
# ------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# ---------- 基本工具函数 ----------
def read_label_file(label_path: Path):
    objs = []
    if not label_path.exists():
        return objs
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except:
                continue
            if len(vals) < 5:
                continue
            cls = int(vals[0])
            x, y, w, h = vals[1:5]
            rest = vals[5:]
            if len(rest) > 0 and len(rest) % 3 == 0:
                dim = 3
                n_kpts = len(rest) // 3
                kpts = []
                for i in range(n_kpts):
                    px = rest[3*i]
                    py = rest[3*i+1]
                    v  = int(rest[3*i+2])
                    kpts.append((px, py, v))
            elif len(rest) >= 0 and len(rest) % 2 == 0:
                dim = 2
                n_kpts = len(rest) // 2
                kpts = []
                for i in range(n_kpts):
                    px = rest[2*i]
                    py = rest[2*i+1]
                    kpts.append((px, py))
            else:
                continue
            objs.append({
                "cls": cls,
                "bbox": (x,y,w,h),
                "kpts": kpts,
                "dim": dim
            })
    return objs

def write_label_file(label_path: Path, objs, dim):
    with open(label_path, "w") as f:
        for ob in objs:
            cls = int(ob["cls"])
            bx, by, bw, bh = ob["bbox"]
            line = f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
            if dim == 3:
                for (px,py,v) in ob["kpts"]:
                    line += f" {px:.6f} {py:.6f} {int(v)}"
            else:
                for (px,py) in ob["kpts"]:
                    line += f" {px:.6f} {py:.6f}"
            f.write(line + "\n")

def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def get_rotation_matrix_and_size(w, h, angle_deg):
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0,2] += (new_w / 2.0) - center[0]
    M[1,2] += (new_h / 2.0) - center[1]
    return M, new_w, new_h

def transform_points(pts, M):
    """
    pts: list of (x,y)
    M: 2x3
    返回 list of (x,y)
    """
    if len(pts) == 0:
        return []
    arr = np.array(pts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1,2)
    ones = np.ones((arr.shape[0],1), dtype=np.float32)
    aug = np.hstack([arr, ones])
    trans = (M @ aug.T).T
    return [tuple(p) for p in trans]

# ---------- 读取并转为绝对坐标 ----------
def load_image_and_labels(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    label_path = LABELS_DIR / (image_path.stem + ".txt")
    objs = read_label_file(label_path)
    abs_objs = []
    for ob in objs:
        dim = ob["dim"]
        cxn, cyn, wn, hn = ob["bbox"]
        cx = cxn * w; cy = cyn * h; bw = wn * w; bh = hn * h
        if dim == 3:
            abs_kps = [(px * w, py * h, int(v)) for px, py, v in ob["kpts"]]
        else:
            abs_kps = [(px * w, py * h) for px, py in ob["kpts"]]
        abs_objs.append({
            "cls": ob["cls"],
            "bbox_abs": (cx, cy, bw, bh),
            "kpts_abs": abs_kps,
            "dim": dim
        })
    return img, abs_objs

# ---------- mosaic（支持缩放 & 随机覆盖） ----------
def build_mosaic(selected_image_paths, max_overlap_ratio=0.3, scale_to_uniform=True, DROP_INSTANCES_IF_ALL_KEYPTS_COVERED=False):
    """
    构造 mosaic 并在拼接阶段处理被覆盖的关键点。
    - selected_image_paths: list of 4 Path
    - 返回 mosaic_img, mosaic_objs（kpts_abs 已剔除/标记被覆盖点）
    - 如果 DROP_INSTANCES_IF_ALL_KEYPTS_COVERED=True，则当某实例的所有关键点都被覆盖时直接丢弃该实例。
    """
    imgs = []
    objs_all = []
    sizes = []
    for p in selected_image_paths:
        img, objs = load_image_and_labels(p)
        imgs.append(img)
        objs_all.append(objs)
        sizes.append(img.shape[:2])  # (h, w)

    # target cell size (average)
    widths = [s[1] for s in sizes]
    heights = [s[0] for s in sizes]
    tgt_w = int(np.mean(widths)) or 1024
    tgt_h = int(np.mean(heights)) or 1024

    # resize to uniform cell if needed
    resized_imgs = []
    resized_objs = []
    for img, objs in zip(imgs, objs_all):
        h,w = img.shape[:2]
        if scale_to_uniform:
            img_r = cv2.resize(img, (tgt_w, tgt_h))
            sx = tgt_w / w; sy = tgt_h / h
        else:
            img_r = img.copy()
            sx = 1.0; sy = 1.0
        objs_r = []
        for ob in objs:
            cx, cy, bw, bh = ob["bbox_abs"]
            cx *= sx; cy *= sy; bw *= sx; bh *= sy
            if ob["dim"] == 3:
                kps = [(px*sx, py*sy, int(v)) for (px,py,v) in ob["kpts_abs"]]
            else:
                kps = [(px*sx, py*sy) for (px,py) in ob["kpts_abs"]]
            objs_r.append({
                "cls": ob["cls"],
                "bbox_abs": (cx, cy, bw, bh),
                "kpts_abs": kps,
                "dim": ob["dim"]
            })
        resized_imgs.append(img_r)
        resized_objs.append(objs_r)

    cell_w = tgt_w; cell_h = tgt_h
    max_ov_x = int(cell_w * max_overlap_ratio)
    max_ov_y = int(cell_h * max_overlap_ratio)

    # random placements (top-left, top-right, bottom-left, bottom-right)
    tl_dx = random.randint(0, max_ov_x); tl_dy = random.randint(0, max_ov_y)
    tr_dx = random.randint(cell_w - max_ov_x, cell_w); tr_dy = random.randint(0, max_ov_y)
    bl_dx = random.randint(0, max_ov_x); bl_dy = random.randint(cell_h - max_ov_y, cell_h)
    br_dx = random.randint(cell_w - max_ov_x, cell_w); br_dy = random.randint(cell_h - max_ov_y, cell_h)

    placements = [
        (tl_dx, tl_dy),
        (tr_dx, tr_dy),
        (bl_dx, bl_dy),
        (br_dx, br_dy)
    ]

    # canvas size (bounding box of placement)
    xs = [dx for dx,_ in placements] + [dx + cell_w for dx,_ in placements]
    ys = [dy for _,dy in placements] + [dy + cell_h for _,dy in placements]
    canvas_w = max(xs); canvas_h = max(ys)
    canvas_w = max(canvas_w, cell_w * 2); canvas_h = max(canvas_h, cell_h * 2)

    # mosaic image + occupancy mask (0 = empty, otherwise image index 1..4)
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=resized_imgs[0].dtype)
    occ = np.zeros((canvas_h, canvas_w), dtype=np.int16)  # occupancy map: which image wrote this pixel (1-based)

    # place images in order (later images will overwrite earlier ones)
    for idx, (img_r, (dx,dy)) in enumerate(zip(resized_imgs, placements)):
        h_r, w_r = img_r.shape[:2]
        x1 = dx; y1 = dy
        x2 = min(dx + w_r, canvas_w); y2 = min(dy + h_r, canvas_h)
        w_write = x2 - x1; h_write = y2 - y1
        if w_write > 0 and h_write > 0:
            # 写入像素
            mosaic[y1:y1+h_write, x1:x1+w_write] = img_r[0:h_write, 0:w_write]
            # 更新占用掩码：标为 idx+1（保证非零）
            occ[y1:y1+h_write, x1:x1+w_write] = idx + 1

    # 根据 occupancy map 处理关键点被覆盖的情况
    mosaic_objs = []
    for img_idx, (dx,dy) in enumerate(placements):
        objs = resized_objs[img_idx]
        for ob in objs:
            cls = ob["cls"]
            dim = ob["dim"]
            bw = ob["bbox_abs"][2]
            bh = ob["bbox_abs"][3]
            kept_kpts = []
            any_kept = False
            if dim == 3:
                for (px,py,v) in ob["kpts_abs"]:
                    x_abs = px + dx
                    y_abs = py + dy
                    xi = int(round(x_abs)); yi = int(round(y_abs))
                    # boundary check
                    if xi < 0 or yi < 0 or xi >= occ.shape[1] or yi >= occ.shape[0]:
                        # 超出画布，算作被覆盖/丢失
                        kept_kpts.append((0.0, 0.0, 0))
                        continue
                    # 如果 occ 表明当前位置最后写入的是当前图（img_idx+1），说明未被覆盖
                    if occ[yi, xi] == img_idx + 1 and int(v) != 0:
                        kept_kpts.append((x_abs, y_abs, int(v)))
                        any_kept = True
                    else:
                        # 被覆盖或不可见 -> 标记为不可见（0）
                        kept_kpts.append((0.0, 0.0, 0))
            else:
                for (px,py) in ob["kpts_abs"]:
                    x_abs = px + dx
                    y_abs = py + dy
                    xi = int(round(x_abs)); yi = int(round(y_abs))
                    if xi < 0 or yi < 0 or xi >= occ.shape[1] or yi >= occ.shape[0]:
                        kept_kpts.append((0.0, 0.0))
                        continue
                    if occ[yi, xi] == img_idx + 1:
                        kept_kpts.append((x_abs, y_abs))
                        any_kept = True
                    else:
                        kept_kpts.append((0.0, 0.0))

            # 如果用户想在全部关键点被覆盖时删除该实例，可以用 DROP_INSTANCES_IF_ALL_KEYPTS_COVERED
            if DROP_INSTANCES_IF_ALL_KEYPTS_COVERED and not any_kept:
                # 丢弃该实例（所有关键点被覆盖或不可见）
                continue

            # 这里 bbox_abs 我保持原始物体的尺寸（中心坐标也按平移修正），后续 crop 阶段会根据保留的关键点重新生成 bbox
            # 计算 bbox center 平移
            cx0, cy0, bw0, bh0 = ob["bbox_abs"]
            cx_m = cx0 + dx; cy_m = cy0 + dy
            mosaic_objs.append({
                "cls": cls,
                "bbox_abs": (cx_m, cy_m, bw0, bh0),
                "kpts_abs": kept_kpts,
                "dim": dim
            })

    return mosaic, mosaic_objs



# ---------- 内容区域 / 等比放大与在内容内选点 ----------
def get_content_bbox(img):
    """
    返回内容区域（非全 0 像素）的 bbox (xmin, ymin, xmax, ymax)。
    如果图全黑则返回 (0,0,w,h)。
    """
    mask = np.any(img != 0, axis=2)
    ys, xs = np.where(mask)
    h, w = img.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(w, xmax); ymax = min(h, ymax)
    return xmin, ymin, xmax, ymax

def scale_mosaic_and_objs(mosaic_img, mosaic_objs, scale):
    if abs(scale - 1.0) < 1e-6:
        return mosaic_img, mosaic_objs
    h, w = mosaic_img.shape[:2]
    new_w = int(w * scale + 0.5); new_h = int(h * scale + 0.5)
    mosaic_r = cv2.resize(mosaic_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    objs_r = []
    for ob in mosaic_objs:
        cx, cy, bw, bh = ob["bbox_abs"]
        cx *= scale; cy *= scale; bw *= scale; bh *= scale
        if ob["dim"] == 3:
            kps = [(px*scale, py*scale, int(v)) for (px,py,v) in ob["kpts_abs"]]
        else:
            kps = [(px*scale, py*scale) for (px,py) in ob["kpts_abs"]]
        objs_r.append({
            "cls": ob["cls"],
            "bbox_abs": (cx, cy, bw, bh),
            "kpts_abs": kps,
            "dim": ob["dim"]
        })
    return mosaic_r, objs_r

def choose_random_crop_inside_content(mosaic_img, mosaic_objs, crop_w, crop_h, allow_scale_up=True, max_scale=4.0):
    """
    保证裁剪窗口完全位于内容区域内（无黑边）。
    若内容区域小于裁剪尺寸并允许放大，则等比放大 mosaic（最多放大到 max_scale）。
    返回 (mosaic_img_new, mosaic_objs_new, crop_x, crop_y, scaled_flag)
    若无法取到合法 crop，返回 (mosaic_img, mosaic_objs, None, None, False)
    """
    img = mosaic_img; objs = mosaic_objs
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = get_content_bbox(img)
    content_w = xmax - xmin; content_h = ymax - ymin

    # 若内容区域够，则在内容区域内取随机左上点
    if content_w >= crop_w and content_h >= crop_h:
        max_x = xmax - crop_w; max_y = ymax - crop_h
        crop_x = random.randint(xmin, max_x)
        crop_y = random.randint(ymin, max_y)
        return img, objs, int(crop_x), int(crop_y), False

    # 内容小且允许放大：计算所需 scale（等比）
    if allow_scale_up and content_w > 0 and content_h > 0:
        scale_needed = max(crop_w / max(content_w, 1e-6), crop_h / max(content_h, 1e-6))
        scale_needed = min(scale_needed, max_scale)
        if scale_needed > 1.0:
            img_s, objs_s = scale_mosaic_and_objs(img, objs, scale_needed)
            xmin2, ymin2, xmax2, ymax2 = get_content_bbox(img_s)
            content_w2 = xmax2 - xmin2; content_h2 = ymax2 - ymin2
            if content_w2 >= crop_w and content_h2 >= crop_h:
                max_x = xmax2 - crop_w; max_y = ymax2 - crop_h
                crop_x = random.randint(xmin2, max_x); crop_y = random.randint(ymin2, max_y)
                return img_s, objs_s, int(crop_x), int(crop_y), True
            else:
                return img_s, objs_s, None, None, True

    # 无法生成合法 crop
    return img, objs, None, None, False

# ---------- 裁剪并写标签（与之前逻辑一致，保留 bbox 备选策略） ----------
def crop_and_generate_labels(mosaic_img, mosaic_objs, crop_x, crop_y, crop_w, crop_h, out_basename, out_index, out_folder_images, out_folder_labels):
    H, W = mosaic_img.shape[:2]
    # 确保 crop 在图内（前面逻辑应保证）
    if crop_x < 0 or crop_y < 0 or crop_x + crop_w > W or crop_y + crop_h > H:
        # 若越界，这里做安全截断（但理想情况不会走到）
        crop_x = max(0, min(crop_x, W - crop_w))
        crop_y = max(0, min(crop_y, H - crop_h))

    crop = mosaic_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
    out_objs = []

    for ob in mosaic_objs:
        dim = ob["dim"]
        kpts = ob["kpts_abs"]
        inside_count = 0
        kpts_new = []
        inside_points = []

        if dim == 3:
            for (px,py,v) in kpts:
                cxp = px - crop_x; cyp = py - crop_y
                if 0 <= cxp < crop_w and 0 <= cyp < crop_h and int(v) != 0:
                    inside_count += 1
                    inside_points.append((cxp, cyp))
                    kpts_new.append((cxp / crop_w, cyp / crop_h, int(v)))
                else:
                    kpts_new.append((0.0, 0.0, 0))
            if inside_count == 0:
                if KEEP_IF_BBOX_OVERLAP:
                    cx, cy, bw, bh = ob["bbox_abs"]
                    bx1 = cx - bw/2; by1 = cy - bh/2; bx2 = cx + bw/2; by2 = cy + bh/2
                    crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                    obj_box = (bx1, by1, bx2, by2)
                    overlap = iou_xyxy(crop_box, obj_box)
                    if overlap >= BBOX_IOU_THRESHOLD:
                        kpts_new = [(0.0,0.0,0) for _ in ob["kpts_abs"]]
                        ix1 = max(bx1, crop_x); iy1 = max(by1, crop_y); ix2 = min(bx2, crop_x+crop_w); iy2 = min(by2, crop_y+crop_h)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        bw_c = ix2 - ix1; bh_c = iy2 - iy1
                        cx_new = (ix1 + ix2) / 2.0 - crop_x
                        cy_new = (iy1 + iy2) / 2.0 - crop_y
                        out_objs.append({
                            "cls": ob["cls"],
                            "bbox": (cx_new / crop_w, cy_new / crop_h, max(1.0,bw_c)/crop_w, max(1.0,bh_c)/crop_h),
                            "kpts": kpts_new
                        })
                        continue
                continue

            xs = [p[0] for p in inside_points]; ys = [p[1] for p in inside_points]
            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            bw = max(1.0, maxx - minx); bh = max(1.0, maxy - miny)
            cx_new = (minx + maxx) / 2.0; cy_new = (miny + maxy) / 2.0
            out_objs.append({
                "cls": ob["cls"],
                "bbox": (cx_new / crop_w, cy_new / crop_h, bw / crop_w, bh / crop_h),
                "kpts": kpts_new
            })
        else:
            for (px,py) in kpts:
                cxp = px - crop_x; cyp = py - crop_y
                if 0 <= cxp < crop_w and 0 <= cyp < crop_h:
                    inside_count += 1
                    inside_points.append((cxp,cyp))
                    kpts_new.append((cxp / crop_w, cyp / crop_h))
                else:
                    kpts_new.append((0.0,0.0))
            if inside_count == 0:
                if KEEP_IF_BBOX_OVERLAP:
                    cx, cy, bw, bh = ob["bbox_abs"]
                    bx1 = cx - bw/2; by1 = cy - bh/2; bx2 = cx + bw/2; by2 = cy + bh/2
                    crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                    obj_box = (bx1, by1, bx2, by2)
                    overlap = iou_xyxy(crop_box, obj_box)
                    if overlap >= BBOX_IOU_THRESHOLD:
                        kpts_new = [(0.0,0.0) for _ in ob["kpts_abs"]]
                        ix1 = max(bx1, crop_x); iy1 = max(by1, crop_y); ix2 = min(bx2, crop_x+crop_w); iy2 = min(by2, crop_y+crop_h)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        bw_c = ix2 - ix1; bh_c = iy2 - iy1
                        cx_new = (ix1 + ix2) / 2.0 - crop_x
                        cy_new = (iy1 + iy2) / 2.0 - crop_y
                        out_objs.append({
                            "cls": ob["cls"],
                            "bbox": (cx_new / crop_w, cy_new / crop_h, max(1.0,bw_c)/crop_w, max(1.0,bh_c)/crop_h),
                            "kpts": kpts_new
                        })
                        continue
                continue

            xs = [p[0] for p in inside_points]; ys = [p[1] for p in inside_points]
            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            bw = max(1.0, maxx - minx); bh = max(1.0, maxy - miny)
            cx_new = (minx + maxx) / 2.0; cy_new = (miny + maxy) / 2.0
            out_objs.append({
                "cls": ob["cls"],
                "bbox": (cx_new / crop_w, cy_new / crop_h, bw / crop_w, bh / crop_h),
                "kpts": kpts_new
            })

    img_name = f"{out_basename}_{out_index:03d}.jpg"
    label_name = f"{out_basename}_{out_index:03d}.txt"

    if len(out_objs) == 0:
        if SAVE_EMPTY_CROPS:
            cv2.imwrite(str(out_folder_images / img_name), crop)
            with open(out_folder_labels / label_name, "w") as f:
                f.write("")
            print(f"[DEBUG] Saved empty crop {img_name} (no objects).")
            return True
        else:
            print(f"[DEBUG] Skipped empty crop {img_name} (no objects).")
            return False

    cv2.imwrite(str(out_folder_images / img_name), crop)
    sample_k = out_objs[0]["kpts"]
    out_dim = 3 if len(sample_k) > 0 and len(sample_k[0]) == 3 else 2
    write_label_file(out_folder_labels / label_name, out_objs, out_dim)
    print(f"[INFO] Saved {img_name} with {len(out_objs)} objects (dim={out_dim}).")
    return True

# ---------- 主流程 ----------
def main():
    img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
    if len(img_paths) < 4:
        raise RuntimeError("images 文件夹中图片少于 4 张，无法做 2x2 拼接。")

    selected = random.sample(img_paths, 4)
    print("[INFO] Selected images:", [p.name for p in selected])

    mosaic_img, mosaic_objs = build_mosaic(selected, max_overlap_ratio=MAX_OVERLAP_RATIO, scale_to_uniform=SCALE_TO_UNIFORM_CELL)
    print(f"[INFO] Built mosaic size {mosaic_img.shape[1]}x{mosaic_img.shape[0]}, total objects: {len(mosaic_objs)}")

    if len(mosaic_objs) == 0:
        print("[WARN] mosaic 中没有对象，可能 labels 缺失或格式不正确。请检查 labels 文件。")

    base_name = "mosaic"
    idx = 0

    # 在内容区域内随机裁 NUM_RANDOM_CROPS 张（原 mosaic）
    for i in range(NUM_RANDOM_CROPS):
        mosaic_img, mosaic_objs, cx, cy, scaled_flag = choose_random_crop_inside_content(mosaic_img, mosaic_objs, CROP_SIZE, CROP_SIZE, allow_scale_up=ALLOW_SCALE_UP, max_scale=MAX_SCALE_UP)
        if cx is None:
            print(f"[WARN] 无法生成第 {i} 个合法 crop（原 mosaic），已跳过。")
            continue
        ok = crop_and_generate_labels(mosaic_img, mosaic_objs, cx, cy, CROP_SIZE, CROP_SIZE, base_name, idx, OUT_IMAGES, OUT_LABELS)
        if ok:
            idx += 1

    # 旋转 mosaic 并变换标签
    angle = random.uniform(ROTATE_ANGLE_RANGE[0], ROTATE_ANGLE_RANGE[1])
    print(f"[INFO] Rotating mosaic by {angle:.2f} degrees.")
    M, new_w, new_h = get_rotation_matrix_and_size(mosaic_img.shape[1], mosaic_img.shape[0], angle)
    rotated = cv2.warpAffine(mosaic_img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

    # transform objs
    mosaic_objs_rot = []
    for ob in mosaic_objs:
        if ob["dim"] == 3:
            pts = [(px,py) for (px,py,v) in ob["kpts_abs"]]
            trans_pts = transform_points(pts, M)
            kpts_rot = []
            for (x,y), orig in zip(trans_pts, ob["kpts_abs"]):
                v = orig[2] if len(orig) >= 3 else 1
                kpts_rot.append((x,y,int(v)))
        else:
            pts = [(px,py) for (px,py) in ob["kpts_abs"]]
            trans_pts = transform_points(pts, M)
            kpts_rot = [(x,y) for (x,y) in trans_pts]

        cx, cy, bw, bh = ob["bbox_abs"]
        cx2, cy2 = transform_points([(cx,cy)], M)[0]
        mosaic_objs_rot.append({
            "cls": ob["cls"],
            "bbox_abs": (cx2, cy2, bw, bh),
            "kpts_abs": kpts_rot,
            "dim": ob["dim"]
        })

    mosaic_img = rotated
    mosaic_objs = mosaic_objs_rot

    # 从旋转后 mosaic 在内容区域内裁 NUM_RANDOM_CROPS 张
    for i in range(NUM_RANDOM_CROPS):
        mosaic_img, mosaic_objs, cx, cy, scaled_flag = choose_random_crop_inside_content(mosaic_img, mosaic_objs, CROP_SIZE, CROP_SIZE, allow_scale_up=ALLOW_SCALE_UP, max_scale=MAX_SCALE_UP)
        if cx is None:
            print(f"[WARN] 无法生成第 {i} 个合法 crop（旋转 mosaic），已跳过.")
            continue
        ok = crop_and_generate_labels(mosaic_img, mosaic_objs, cx, cy, CROP_SIZE, CROP_SIZE, base_name + "_rot", idx, OUT_IMAGES, OUT_LABELS)
        if ok:
            idx += 1

    print(f"[DONE] 输出目录：{OUT_IMAGES} 和 {OUT_LABELS}，共写入 {idx} 张裁剪图（含 label）。")

if __name__ == "__main__":
    main()
