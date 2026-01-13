import os
from PIL import Image, ImageOps

# -------- 用户可修改的设置 --------
src_dir = r"./FMTS"   # 源图片文件夹（放 1024x1024 图片）
dst_dir = r"./tile_FMTS"     # 裁剪后图片要保存到的文件夹
target_size = (768, 768)                   # 目标宽高 (width, height)
allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
resize_if_smaller = False  # 若 True，会把小于目标的图片先按比例放大到能裁出的大小（一般不需要）
# ------------------------------------

def center_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    w, h = img.size
    if w == tw and h == th:
        return img
    left = (w - tw) // 2
    upper = (h - th) // 2
    right = left + tw
    lower = upper + th
    return img.crop((left, upper, right, lower))

os.makedirs(dst_dir, exist_ok=True)

processed = 0
skipped = 0
errors = 0

for fname in sorted(os.listdir(src_dir)):
    lower = fname.lower()
    if not lower.endswith(allowed_ext):
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    try:
        with Image.open(src_path) as im:
            # 处理相机或手机照片的旋转 EXIF 标记
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            tw, th = target_size

            if w < tw or h < th:
                if resize_if_smaller:
                    # 按比例放大到至少能裁出目标尺寸
                    scale = max(tw / w, th / h)
                    new_size = (int(w * scale + 0.9999), int(h * scale + 0.9999))
                    im = im.resize(new_size, Image.LANCZOS)
                    w, h = im.size
                else:
                    print(f"跳过（尺寸小于目标，设置 resize_if_smaller=True 可改变此行为）：{fname} -> {w}x{h}")
                    skipped += 1
                    continue

            # 中心裁剪
            cropped = center_crop(im, tw, th)

            # 尝试保留原格式（若无法获取则以 PNG 保存）
            fmt = im.format if getattr(im, "format", None) else None
            save_kwargs = {}
            # 对 JPEG 可设置 quality（可选）
            if fmt and fmt.upper() in ("JPEG", "JPG"):
                save_kwargs["quality"] = 95
                # 若图片有透明通道但目标是 JPEG，会自动丢弃透明通道
                if cropped.mode in ("RGBA", "LA"):
                    cropped = cropped.convert("RGB")

            # 保留 PNG 的透明通道（若有）
            cropped.save(dst_path, format=fmt if fmt else None, **save_kwargs)
            processed += 1

    except Exception as e:
        print(f"处理出错：{fname} -> {e}")
        errors += 1

print(f"完成。已处理: {processed} 张，跳过: {skipped} 张，错误: {errors} 张。")
