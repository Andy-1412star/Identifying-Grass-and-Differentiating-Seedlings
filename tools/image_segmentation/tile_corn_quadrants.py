# 1) 最终版切图脚本：768×768 网格，白色填充无损覆盖，基于标注的全局偏移避免切断
# 2) 使用方式：直接运行本文件（不需命令行参数）；路径与参数在顶部常量处修改
# 3) 网格尺寸：ceil(W/768) × ceil(H/768)，超出边界部分以白色像素填充
# 4) 全局偏移：穷举整数偏移（不超过 767 像素），尽量减少分割线与标注框相交（1~2 像素容差）
# 5) 标注规则：仅保留“完全落入切片”的目标，避免产生半目标标注
# 6) 标注格式：class cx cy w h kx ky v（均为相对当前切片的归一化坐标）
# 7) 命名规则：按行优先顺序，文件名追加 (1)、(2)、(3)… 数字序号
# 8) 调试信息：可选保存 meta（原图粘贴范围与位置），便于叠加可视化检查
# 9) 目录结构：输入需含 images/ 与 labels/；输出会生成 images/、labels/（以及可选 meta/）
# 10) 仅注释含中文，代码与运行时字符串均为英文，满足“除注释外不含中文”的要求

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
import json

# ===================== 顶部可配置常量（请按需修改） =====================
# 数据集根目录（包含 all、train、val 等文件夹）
ROOT_DIR = Path(".")
# 源数据目录（需包含 images/ 与 labels/ 子目录）
SRC_DIR = ROOT_DIR / "all"
# 输出目录（自动创建，包含 images/ 与 labels/ 子目录，可选 meta/ 调试子目录）
OUT_DIR = ROOT_DIR / "tiles"

# 单个切片的尺寸（像素）
TILE_SIZE = 768
# 判定“完全落入切片”的像素容差（建议 1~2 像素）
EPS_PX = 2.0
# 是否保存每个切片的调试 meta（记录原图粘贴范围、粘贴位置等）
SAVE_META = False
# 仅处理前 N 张图片（0 表示处理全部）
LIMIT = 0
# ==================================================================


@dataclass
class Obj:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    kx: float
    ky: float
    v: int

    # 派生属性（单位：像素）
    def bbox_pix(self, W: int, H: int) -> Tuple[float, float, float, float]:
        cxp, cyp = self.cx * W, self.cy * H
        wp, hp = self.w * W, self.h * H
        return cxp - wp / 2.0, cyp - hp / 2.0, cxp + wp / 2.0, cyp + hp / 2.0

    def kpt_pix(self, W: int, H: int) -> Tuple[float, float]:
        return self.kx * W, self.ky * H


def parse_label_file(path: Path) -> List[Obj]:
    objs: List[Obj] = []
    if not path.exists():
        return objs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                # 行格式不正确，跳过
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h, kx, ky = map(float, parts[1:7])
                v = int(float(parts[7]))
                objs.append(Obj(c, cx, cy, w, h, kx, ky, v))
            except Exception:
                # 容错处理：忽略该行
                continue
    return objs


def format_label_line(o: Obj) -> str:
    return f"{o.cls} {o.cx:.6f} {o.cy:.6f} {o.w:.6f} {o.h:.6f} {o.kx:.6f} {o.ky:.6f} {o.v}"


def build_forbidden_intervals(
    intervals: List[Tuple[float, float]], lower: float, upper: float, eps: float
) -> List[Tuple[float, float]]:
    # 将区间加上 eps 边距后合并，得到 [lower, upper] 范围内的“禁止放置分割线”的区间集合
    forb: List[Tuple[float, float]] = []
    for a, b in intervals:
        a2, b2 = min(a, b) - eps, max(a, b) + eps
        a2 = max(a2, lower)
        b2 = min(b2, upper)
        if a2 <= b2:
            forb.append((a2, b2))
    forb.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in forb:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged


def count_lines_in_forbidden(lines: List[float], forb: List[Tuple[float, float]]) -> int:
    # 统计有多少网格分割线落在任一禁止区间内
    cnt = 0
    for x in lines:
        for a, b in forb:
            if a <= x <= b:
                cnt += 1
                break
    return cnt


def compute_grid_offset(
    W: int,
    H: int,
    objects: List[Obj],
    tile: int,
    eps: float,
) -> Tuple[int, int, int, int]:
    # 决定全局网格偏移 (L, T) 以及网格大小 (Nx, Ny)。目标：在全覆盖的同时，尽量减少分割线穿过 bbox 的次数（带 eps 容差）。
    # 返回 (L, T, Nx, Ny)，其中 L ∈ [W - Nx*tile, 0]，T ∈ [H - Ny*tile, 0]。
    from math import ceil

    Nx = max(1, int(ceil(W / float(tile))))
    Ny = max(1, int(ceil(H / float(tile))))

    Lmin, Lmax = W - Nx * tile, 0
    Tmin, Tmax = H - Ny * tile, 0

    if not objects:
        # 无标注：无需优化，返回零偏移
        return 0, 0, Nx, Ny

    xs: List[Tuple[float, float]] = []
    ys: List[Tuple[float, float]] = []
    for o in objects:
        x0, y0, x1, y1 = o.bbox_pix(W, H)
        xs.append((x0, x1))
        ys.append((y0, y1))

    forb_x = build_forbidden_intervals(xs, 0, W, eps)
    forb_y = build_forbidden_intervals(ys, 0, H, eps)

    best_L = 0
    best_T = 0
    best_penalty = float("inf")

    # 穷举整数偏移（范围小于一个 tile，计算量可接受）
    for L in range(Lmin, Lmax + 1):
        lines_x = [L + i * tile for i in range(1, Nx)]
        px = count_lines_in_forbidden(lines_x, forb_x)
    # 简单剪枝：若当前横向罚分已不优，则跳过纵向循环
        if px > best_penalty:
            continue
        for T in range(Tmin, Tmax + 1):
            lines_y = [T + j * tile for j in range(1, Ny)]
            py = count_lines_in_forbidden(lines_y, forb_y)
            pen = px + py
            if pen < best_penalty or (pen == best_penalty and (L > best_L or (L == best_L and T > best_T))):
                best_penalty = pen
                best_L, best_T = L, T

    return best_L, best_T, Nx, Ny


def crop_with_padding(img: Image.Image, x0: int, y0: int, size: int):
    # 从 (x0, y0) 处裁剪 size×size 的区域；若超出边界则以白色填充。返回：目标图、原图有效裁剪框、粘贴位置。
    W, H = img.size
    dest = Image.new("RGB", (size, size), color=(255, 255, 255))
    # 与原图的交集区域
    src_left = max(0, x0)
    src_top = max(0, y0)
    src_right = min(W, x0 + size)
    src_bottom = min(H, y0 + size)

    dest_left = max(0, -x0)
    dest_top = max(0, -y0)

    if src_right > src_left and src_bottom > src_top:
        region = img.crop((src_left, src_top, src_right, src_bottom))
        dest.paste(region, (dest_left, dest_top))

    return dest, (src_left, src_top, src_right, src_bottom), (dest_left, dest_top)


def reproject_labels_to_tile(
    objects: List[Obj],
    W: int,
    H: int,
    tile_x: int,
    tile_y: int,
    tile: int,
    eps: float,
) -> List[Obj]:
    # 返回完全落入切片的目标，并转换为切片归一化坐标
    kept: List[Obj] = []
    x_min_t = tile_x + eps
    y_min_t = tile_y + eps
    x_max_t = tile_x + tile - eps
    y_max_t = tile_y + tile - eps

    for o in objects:
        x0, y0, x1, y1 = o.bbox_pix(W, H)
        if x0 >= x_min_t and y0 >= y_min_t and x1 <= x_max_t and y1 <= y_max_t:
            # 坐标转换到切片系
            cxp, cyp = o.cx * W, o.cy * H
            kxp, kyp = o.kx * W, o.ky * H
            new = Obj(
                cls=o.cls,
                cx=(cxp - tile_x) / tile,
                cy=(cyp - tile_y) / tile,
                w=(o.w * W) / tile,
                h=(o.h * H) / tile,
                kx=(kxp - tile_x) / tile,
                ky=(kyp - tile_y) / tile,
                v=o.v,
            )
            kept.append(new)
    return kept


    


def process_image(
    img_path: Path,
    labels_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile: int,
    eps: float,
    meta_dir: Optional[Path] = None,
):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    objects = parse_label_file(labels_path)

    # 计算全局网格偏移与网格大小
    L, T, Nx, Ny = compute_grid_offset(W, H, objects, tile, eps)

    stem = img_path.stem
    ext = img_path.suffix  # 保留原始扩展名及大小写

    # 按行优先顺序遍历切片并命名为 (1),(2),...
    idx = 1
    for j in range(Ny):
        for i in range(Nx):
            tx = L + i * tile
            ty = T + j * tile
            tile_img, src_bbox, paste_xy = crop_with_padding(img, tx, ty, tile)
            tile_objs = reproject_labels_to_tile(objects, W, H, tx, ty, tile, eps)

            idx_str = f"({idx})"

            out_img_path = out_img_dir / f"{stem}{idx_str}{ext}"
            out_lbl_path = out_lbl_dir / f"{stem}{idx_str}.txt"

            tile_img.save(out_img_path)
            with out_lbl_path.open("w", encoding="utf-8") as f:
                for o in tile_objs:
                    f.write(format_label_line(o) + "\n")

            if meta_dir is not None:
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta = {
                    "orig_w": W,
                    "orig_h": H,
                    "tile": tile,
                    "tile_x": tx,
                    "tile_y": ty,
                    "src_bbox": list(src_bbox),
                    "paste_xy": list(paste_xy),
                }
                meta_path = meta_dir / f"{stem}{idx_str}.json"
                with meta_path.open("w", encoding="utf-8") as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)

            idx += 1


def find_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}
    return sorted([p for p in images_dir.iterdir() if p.suffix in exts and p.is_file()])


def main():
    # 使用顶部常量配置路径与参数
    root = ROOT_DIR
    src_dir = SRC_DIR
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    out_dir = OUT_DIR
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_meta_dir: Optional[Path] = (out_dir / "meta") if SAVE_META else None
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return

    for idx, img_path in enumerate(images, 1):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        try:
            process_image(
                img_path,
                lbl_path,
                out_img_dir,
                out_lbl_dir,
                tile=TILE_SIZE,
                eps=EPS_PX,
                meta_dir=out_meta_dir,
            )
        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")
        if idx % 50 == 0:
            print(f"Processed {idx} images…")
        if LIMIT and idx >= LIMIT:
            break

    print("Done.")


if __name__ == "__main__":
    main()
