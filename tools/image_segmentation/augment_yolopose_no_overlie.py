#!/usr/bin/env python3
"""
功能：
- 从 images/ 随机选 4 张图做 2x2 拼接（mosaic）
- 从拼接图随机裁出 3 张 1024x1024，保存 crop 和更新后的 labels
- 将拼接图随机旋转（随机角度），再随机裁出 3 张 1024x1024，保存 crop 和更新后的 labels
- 支持 YOLO-pose 格式 dim=2 或 dim=3（见 Ultralytics YOLO pose 格式）
- 输出到 out/images 和 out/labels

依赖：
pip install opencv-python numpy tqdm

使用：
python augment_yolopose_mosaic.py
"""
import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# ---------- 配置 ----------
IMAGES_DIR = Path("corn_datasets/images")
LABELS_DIR = Path("corn_datasets/labels")
OUT_IMAGES = Path("outCorn/images")
OUT_LABELS = Path("outCorn/labels")
CROP_SIZE = 1024
NUM_RANDOM_CROPS = 3  # 从原始 mosaic 裁剪数量
RANDOM_SEED = 42
# ------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# ---------- 工具函数 ----------
def read_label_file(label_path):
    """
    读取 YOLO-pose txt，返回 list of dict:
    {
      'cls': int,
      'bbox': (x_center_norm, y_center_norm, w_norm, h_norm),
      'kpts': [ (x_norm,y_norm) ... ]  (if dim=2)
      OR
      'kpts': [ (x_norm,y_norm,vis) ... ] (if dim=3)
      'dim': 2 or 3,
    }
    """
    objs = []
    if not label_path.exists():
        return objs
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vals = [float(x) for x in parts]
            if len(vals) < 5:
                continue
            cls = int(vals[0])
            x, y, w, h = vals[1:5]
            rest = vals[5:]
            # detect dim: if (len(rest) % 3 == 0) -> dim3 else dim2
            if len(rest) % 3 == 0:
                dim = 3
                n_kpts = len(rest) // 3
                kpts = []
                for i in range(n_kpts):
                    px = rest[3*i]
                    py = rest[3*i+1]
                    v  = int(rest[3*i+2])  # treat visibility as int
                    kpts.append((px, py, v))
            elif len(rest) % 2 == 0:
                dim = 2
                n_kpts = len(rest) // 2
                kpts = []
                for i in range(n_kpts):
                    px = rest[2*i]
                    py = rest[2*i+1]
                    kpts.append((px, py))
            else:
                # unknown -> skip
                continue
            objs.append({
                "cls": cls,
                "bbox": (x,y,w,h),
                "kpts": kpts,
                "dim": dim
            })
    return objs

def write_label_file(label_path, objs, dim):
    """
    objs: list of dict as above, but with normalized coords relative to the image to be saved
    dim: 2 or 3 (output format)
    """
    with open(label_path, "w") as f:
        for ob in objs:
            cls = int(ob["cls"])
            bx, by, bw, bh = ob["bbox"]
            line = f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
            if dim == 2:
                for (px,py) in ob["kpts"]:
                    line += f" {px:.6f} {py:.6f}"
            else:
                for (px,py,v) in ob["kpts"]:
                    line += f" {px:.6f} {py:.6f} {int(v)}"
            f.write(line + "\n")

def xywh_to_xyxy(cx,cy,w,h):
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return x1,y1,x2,y2

def xyxy_to_xywh(x1,y1,x2,y2):
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = max(0, x2-x1)
    h = max(0, y2-y1)
    return cx,cy,w,h

def ensure_min_canvas(img, min_w, min_h):
    """
    如果 img 尺寸小于要求则 pad 到至少 min_w x min_h（右下填充黑色）
    """
    h,w = img.shape[:2]
    if w >= min_w and h >= min_h:
        return img
    nw = max(w, min_w)
    nh = max(h, min_h)
    canvas = np.zeros((nh, nw, 3), dtype=img.dtype)
    canvas[0:h, 0:w] = img
    return canvas

def rotate_image_and_points(img, points, angle_deg):
    """
    旋转图片并转换点坐标。
    points: array Nx2 absolute coordinates (x,y)
    返回: rotated_img, transformed_points (Nx2), translation (dx,dy) applied due to canvas expand
    """
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # compute new bounding dims
    cos = abs(M[0,0])
    sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    # adjust rotation matrix to take into account translation
    M[0,2] += (new_w / 2) - center[0]
    M[1,2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    # apply transform to points (augment with 1)
    if len(points)==0:
        return rotated, np.zeros((0,2)), (int(M[0,2]), int(M[1,2]))
    pts = np.hstack([points, np.ones((points.shape[0],1))])
    pts_t = (M @ pts.T).T  # Nx2
    return rotated, pts_t, (M[0,2], M[1,2])

# ---------- 主要流程函数 ----------
def load_image_and_labels(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h,w = img.shape[:2]
    label_path = LABELS_DIR / (image_path.stem + ".txt")
    objs = read_label_file(label_path)
    # convert normalized coords to absolute coords (bbox center + keypoints)
    abs_objs = []
    for ob in objs:
        dim = ob["dim"]
        cxn,cyn,wn,hn = ob["bbox"]
        cx = cxn * w
        cy = cyn * h
        bw = wn * w
        bh = hn * h
        abs_kps = []
        if dim == 3:
            for (pxn,pyn,v) in ob["kpts"]:
                abs_kps.append((pxn * w, pyn * h, int(v)))
        else:
            for (pxn,pyn) in ob["kpts"]:
                abs_kps.append((pxn * w, pyn * h))
        abs_objs.append({
            "cls": ob["cls"],
            "bbox_abs": (cx,cy,bw,bh),
            "kpts_abs": abs_kps,
            "dim": dim
        })
    return img, abs_objs

def build_mosaic(selected_image_paths):
    """
    构造 2x2 mosaic，返回 mosaic_img, mosaic_objs, mosaic_dims
    mosaic placement:
     [img0 | img1]
     [img2 | img3]
    每个格子大小由各自图片实际大小决定；拼接以左上对齐（cell 内未填满的区域保持黑色）
    """
    imgs = []
    objs_all = []
    sizes = []
    for p in selected_image_paths:
        img, objs = load_image_and_labels(p)
        imgs.append(img)
        objs_all.append(objs)
        sizes.append(img.shape[:2])  # (h,w)
    # compute cell widths/heights
    top_h = max(sizes[0][0], sizes[1][0])
    bottom_h = max(sizes[2][0], sizes[3][0])
    left_w = max(sizes[0][1], sizes[2][1])
    right_w = max(sizes[1][1], sizes[3][1])
    canvas_w = left_w + right_w
    canvas_h = top_h + bottom_h
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=imgs[0].dtype)
    placements = []  # list of (dx,dy) for each of 4 images
    # place 0 (top-left)
    mosaic[0:sizes[0][0], 0:sizes[0][1]] = imgs[0]
    placements.append((0,0))
    # place 1 (top-right)
    mosaic[0:sizes[1][0], left_w:left_w+sizes[1][1]] = imgs[1]
    placements.append((left_w, 0))
    # place 2 (bottom-left)
    mosaic[top_h:top_h+sizes[2][0], 0:sizes[2][1]] = imgs[2]
    placements.append((0, top_h))
    # place 3 (bottom-right)
    mosaic[top_h:top_h+sizes[3][0], left_w:left_w+sizes[3][1]] = imgs[3]
    placements.append((left_w, top_h))

    # merge labels: for each object convert abs coords to mosaic absolute coords
    mosaic_objs = []
    for i in range(4):
        dx,dy = placements[i]
        for ob in objs_all[i]:
            cx,cy,bw,bh = ob["bbox_abs"]
            cx_m = cx + dx
            cy_m = cy + dy
            kpts_m = []
            if ob["dim"] == 3:
                for (px,py,v) in ob["kpts_abs"]:
                    kpts_m.append((px + dx, py + dy, int(v)))
            else:
                for (px,py) in ob["kpts_abs"]:
                    kpts_m.append((px + dx, py + dy))
            mosaic_objs.append({
                "cls": ob["cls"],
                "bbox_abs": (cx_m, cy_m, bw, bh),
                "kpts_abs": kpts_m,
                "dim": ob["dim"]
            })
    return mosaic, mosaic_objs

def convert_abs_objs_to_yolopose(objs_abs, img_w, img_h, kpt_dim):
    """
    将绝对坐标 objs_abs 转为 YOLO-pose 格式（normalized）；
    objs_abs: list with keys 'cls','bbox_abs','kpts_abs', 'dim'
    kpt_dim: 2 or 3 -> output format dim
    """
    out = []
    for ob in objs_abs:
        cls = ob["cls"]
        cx,cy,bw,bh = ob["bbox_abs"]
        # normalize
        bx = cx / img_w
        by = cy / img_h
        bw_n = bw / img_w
        bh_n = bh / img_h
        kpts_out = []
        if ob["dim"] == 3:
            # ob kpts are (x,y,v)
            for (px,py,v) in ob["kpts_abs"]:
                kpts_out.append((px / img_w, py / img_h, v))
        else:
            for (px,py) in ob["kpts_abs"]:
                kpts_out.append((px / img_w, py / img_h))
        # if requested output dim differs from source, adapt:
        if kpt_dim == 2 and ob["dim"] == 3:
            # drop visibility
            kpts_out = [(x,y) for (x,y,v) in kpts_out]
        if kpt_dim == 3 and ob["dim"] == 2:
            # add visibility=1 for all
            kpts_out = [(x,y,1) for (x,y) in kpts_out]
        out.append({
            "cls": cls,
            "bbox": (bx,by,bw_n,bh_n),
            "kpts": kpts_out
        })
    return out

def crop_and_generate_labels(mosaic_img, mosaic_objs, crop_x, crop_y, crop_w, crop_h, out_basename, out_index, out_folder_images, out_folder_labels):
    """
    对 mosaic_img 以 (crop_x, crop_y, crop_w, crop_h) 裁剪，生成图片与 label 文件（YOLO-pose）
    返回是否写入文件（是否有有效对象）
    处理策略简述：
      - 对每个对象，把 keypoints 转成 crop 内坐标 (cx-krop_x, cy-crop_y)
      - 如果 keypoint 在裁剪内，则保留其坐标；否则对于 dim=3 将其置为 visibility=0 并将坐标置为 0；对于 dim=2 将坐标置为 0
      - 若对象在裁剪内没有任何可见 keypoint（dim=3）或没有任何在裁剪内的 keypoint（dim=2），则丢弃该对象
      - bbox 根据裁剪后保留的 keypoint 的 min/max 计算；若只有 1 个点，bbox 设为 1x1 像素（避免 0）
    """
    H, W = mosaic_img.shape[:2]
    # ensure crop within canvas by padding if needed
    if crop_x < 0 or crop_y < 0 or crop_x + crop_w > W or crop_y + crop_h > H:
        # pad mosaic to at least crop size
        newW = max(W, crop_x + crop_w)
        newH = max(H, crop_y + crop_h)
        canvas = np.zeros((newH, newW, 3), dtype=mosaic_img.dtype)
        canvas[0:H,0:W] = mosaic_img
        mosaic_img = canvas
        H,W = mosaic_img.shape[:2]
    crop = mosaic_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
    out_objs = []
    for ob in mosaic_objs:
        dim = ob["dim"]
        kpts = ob["kpts_abs"]  # either (x,y) or (x,y,v)
        inside_count = 0
        kpts_new = []
        inside_points = []
        if dim == 3:
            for (px,py,v) in kpts:
                cxp = px - crop_x
                cyp = py - crop_y
                if 0 <= cxp < crop_w and 0 <= cyp < crop_h and int(v) != 0:
                    inside_count += 1
                    inside_points.append((cxp, cyp))
                    kpts_new.append((cxp/crop_w, cyp/crop_h, int(v)))
                else:
                    # mark invisible and coords set to 0
                    kpts_new.append((0.0, 0.0, 0))
            # require at least one visible point inside
            if inside_count == 0:
                continue
            # compute bbox from visible points
            xs = [p[0] for p in inside_points]
            ys = [p[1] for p in inside_points]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            # avoid zero size
            bw = max(1.0, maxx - minx)
            bh = max(1.0, maxy - miny)
            cx_new = (minx + maxx) / 2.0
            cy_new = (miny + maxy) / 2.0
            bx = cx_new / crop_w
            by = cy_new / crop_h
            bw_n = bw / crop_w
            bh_n = bh / crop_h
            out_objs.append({
                "cls": ob["cls"],
                "bbox": (bx,by,bw_n,bh_n),
                "kpts": kpts_new
            })
        else:
            # dim == 2
            for (px,py) in kpts:
                cxp = px - crop_x
                cyp = py - crop_y
                if 0 <= cxp < crop_w and 0 <= cyp < crop_h:
                    inside_count += 1
                    inside_points.append((cxp, cyp))
                    kpts_new.append((cxp/crop_w, cyp/crop_h))
                else:
                    # set to 0
                    kpts_new.append((0.0, 0.0))
            if inside_count == 0:
                continue
            xs = [p[0] for p in inside_points]
            ys = [p[1] for p in inside_points]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            bw = max(1.0, maxx - minx)
            bh = max(1.0, maxy - miny)
            cx_new = (minx + maxx) / 2.0
            cy_new = (miny + maxy) / 2.0
            bx = cx_new / crop_w
            by = cy_new / crop_h
            bw_n = bw / crop_w
            bh_n = bh / crop_h
            out_objs.append({
                "cls": ob["cls"],
                "bbox": (bx,by,bw_n,bh_n),
                "kpts": kpts_new
            })
    # if no objects remain, return False
    if len(out_objs) == 0:
        return False
    # write image and label
    img_name = f"{out_basename}_{out_index:03d}.jpg"
    label_name = f"{out_basename}_{out_index:03d}.txt"
    cv2.imwrite(str(out_folder_images / img_name), crop)
    # detect output kpt dim: use first obj
    sample = out_objs[0]
    sample_k = sample["kpts"]
    if len(sample_k) > 0 and len(sample_k[0]) == 3:
        out_dim = 3
    else:
        out_dim = 2
    write_label_file(out_folder_labels / label_name, out_objs, out_dim)
    return True

# ---------- 主脚本 ----------
def main():
    # list all images
    img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if len(img_paths) < 4:
        raise RuntimeError("images 文件夹中图片少于 4 张，无法做 2x2 拼接。")
    # choose 4 random images
    selected = random.sample(img_paths, 4)
    mosaic_img, mosaic_objs = build_mosaic(selected)
    H, W = mosaic_img.shape[:2]

    # pad mosaic if smaller than CROP_SIZE
    mosaic_img = ensure_min_canvas(mosaic_img, CROP_SIZE, CROP_SIZE)
    H, W = mosaic_img.shape[:2]

    base_name = "mosaic"
    idx = 0
    # 1) 从原 mosaic 随机裁出 NUM_RANDOM_CROPS 张 1024x1024
    for i in range(NUM_RANDOM_CROPS):
        # choose random top-left such that crop inside canvas
        max_x = max(0, W - CROP_SIZE)
        max_y = max(0, H - CROP_SIZE)
        cx = random.randint(0, max_x)
        cy = random.randint(0, max_y)
        ok = crop_and_generate_labels(mosaic_img, mosaic_objs, cx, cy, CROP_SIZE, CROP_SIZE, base_name, idx, OUT_IMAGES, OUT_LABELS)
        if ok:
            idx += 1

    # 2) 对 mosaic 随机旋转（角度随机）
    angle = random.uniform(-45, 45)  # 你可以根据需要更改角度范围
    rotated_img, _, _ = rotate_image_and_points(mosaic_img, np.zeros((0,2)), angle)  # we will re-calc points below
    # rotate keypoints and bboxes too
    # prepare point list per object
    all_pts = []
    obj_pts_counts = []
    for ob in mosaic_objs:
        pts = []
        if ob["dim"] == 3:
            for (x,y,v) in ob["kpts_abs"]:
                pts.append((x,y))
        else:
            for (x,y) in ob["kpts_abs"]:
                pts.append((x,y))
        obj_pts_counts.append(len(pts))
        if len(pts) > 0:
            all_pts.extend(pts)
    if len(all_pts) > 0:
        pts_arr = np.array(all_pts, dtype=np.float32)
    else:
        pts_arr = np.zeros((0,2), dtype=np.float32)
    # rotate mosaic and points (we need actual transformed points)
    rotated_img2, pts_t, trans = rotate_image_and_points(mosaic_img, pts_arr, angle)
    # rebuild mosaic_objs_rotated with transformed kpts
    mosaic_objs_rot = []
    idx_ptr = 0
    for ob in mosaic_objs:
        kcnt = obj_pts_counts[idx_ptr] if idx_ptr < len(obj_pts_counts) else 0
        klist = []
        if kcnt > 0:
            for j in range(kcnt):
                x,y = pts_t[idx_ptr]
                # original data had possibly v
                if ob["dim"] == 3:
                    v = ob["kpts_abs"][j][2]
                    klist.append((x,y,int(v)))
                else:
                    klist.append((x,y))
                idx_ptr += 1
        else:
            # no points
            pass
        # compute bbox center from original bbox absolute then rotate it as a point
        cx,cy,bw,bh = ob["bbox_abs"]
        # rotate center
        center_pts = np.array([[cx,cy]], dtype=np.float32)
        M = cv2.getRotationMatrix2D((mosaic_img.shape[1]/2, mosaic_img.shape[0]/2), angle, 1.0)
        cos = abs(M[0,0]); sin = abs(M[0,1])
        new_w = int(mosaic_img.shape[0] * sin + mosaic_img.shape[1] * cos)
        new_h = int(mosaic_img.shape[0] * cos + mosaic_img.shape[1] * sin)
        M[0,2] += (new_w / 2) - (mosaic_img.shape[1]/2)
        M[1,2] += (new_h / 2) - (mosaic_img.shape[0]/2)
        pt_aug = np.hstack([center_pts, np.ones((1,1))])
        cx2, cy2 = (M @ pt_aug.T).T[0]
        mosaic_objs_rot.append({
            "cls": ob["cls"],
            "bbox_abs": (cx2, cy2, bw, bh),  # bbox size kept (approx) same; will recompute bbox based on keypoints when cropping
            "kpts_abs": klist,
            "dim": ob["dim"]
        })
    # now rotated_img2 与 mosaic_objs_rot 对应
    rotated_img = rotated_img2
    mosaic_img = rotated_img
    mosaic_objs = mosaic_objs_rot
    H, W = mosaic_img.shape[:2]
    mosaic_img = ensure_min_canvas(mosaic_img, CROP_SIZE, CROP_SIZE)
    H, W = mosaic_img.shape[:2]

    # 3) 从旋转后的 mosaic 再裁 NUM_RANDOM_CROPS 张
    for i in range(NUM_RANDOM_CROPS):
        max_x = max(0, W - CROP_SIZE)
        max_y = max(0, H - CROP_SIZE)
        cx = random.randint(0, max_x)
        cy = random.randint(0, max_y)
        ok = crop_and_generate_labels(mosaic_img, mosaic_objs, cx, cy, CROP_SIZE, CROP_SIZE, base_name + "_rot", idx, OUT_IMAGES, OUT_LABELS)
        if ok:
            idx += 1

    print(f"完成，输出图片与 label 请查看：{OUT_IMAGES} 和 {OUT_LABELS} ，共写入 {idx} 张裁剪图（含 label）。")

if __name__ == "__main__":
    main()
