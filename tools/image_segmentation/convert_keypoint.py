"""
convert_to_yolov8_keypoint_labels.py

将 YOLO（class x_center y_center w h，归一化）标签转换为 YOLOv8 keypoint（单关键点）标签：
- 把所有 class_id 改为 0
- 关键点为边界框中心点（一个关键点）
- 将边界框大小固定为指定的像素（默认 10x10），并输出为归一化的宽高
- 默认在 keypoint triplet 中输出 visibility（默认 2 = 可见）。可选输出无 visibility（dim=2）

用法示例：
    python convert_to_yolov8_keypoint_labels.py \
        --labels-dir ./labels \
        --images-dir ./images \
        --output-dir ./labels_kp \
        --bbox-px 10 \
                    30
        --visibility 2

说明：脚本会为每个标签文件寻找同名图片（jpg/png/jpeg/tif/bmp）。如果找不到图片，你可以传入 --default-size WIDTH HEIGHT 来指定尺寸（用于归一化）。

输出的每行格式（默认包含 visibility，dim=3）：
    <class=0> <x_center_norm> <y_center_norm> <width_norm> <height_norm> <kpx_norm> <kpy_norm> <visibility>

如果使用 --no-visibility（dim=2），则输出：
    <class=0> <x_center_norm> <y_center_norm> <width_norm> <height_norm> <kpx_norm> <kpy_norm>

"""

import os
import argparse
from PIL import Image

COMMON_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def find_image_path_for_label(label_path, images_dir):
    base = os.path.splitext(os.path.basename(label_path))[0]
    for ext in COMMON_EXTS:
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_label_file(label_path, img_w, img_h, bbox_px, include_visibility=True, visibility_val=2):
    """Read a YOLO-style label file and convert each line to YOLOv8 keypoint format.

    Returns list of output lines (strings).
    """
    out_lines = []
    with open(label_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return out_lines

    new_w_norm = bbox_px / float(img_w)
    new_h_norm = bbox_px / float(img_h)

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            # skip malformed
            continue
        try:
            # original values are normalized; still we compute pixel center and then re-normalize
            xc_norm = float(parts[1])
            yc_norm = float(parts[2])
            xc_px = xc_norm * img_w
            yc_px = yc_norm * img_h

            new_xc_norm = xc_px / float(img_w)
            new_yc_norm = yc_px / float(img_h)

            kpx_norm = new_xc_norm
            kpy_norm = new_yc_norm

            if include_visibility:
                out = f"0 {new_xc_norm:.6f} {new_yc_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f} {kpx_norm:.6f} {kpy_norm:.6f} {int(visibility_val)}"
            else:
                out = f"0 {new_xc_norm:.6f} {new_yc_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f} {kpx_norm:.6f} {kpy_norm:.6f}"
            out_lines.append(out)
        except Exception:
            continue

    return out_lines


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO (xywh normalized) labels to YOLOv8 keypoint labels (single center keypoint, fixed bbox size).')
    parser.add_argument('--labels-dir', required=True, help='Directory containing original .txt label files')
    parser.add_argument('--images-dir', required=False, help='Directory containing images (searched for same basename). If omitted and --default-size not set, script will attempt to find images next to label files.')
    parser.add_argument('--output-dir', required=True, help='Directory to write converted .txt files')
    parser.add_argument('--bbox-px', type=int, default=10, help='Fixed bbox pixel size (default: 10)')
    parser.add_argument('--default-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='If provided, use this image size for all files (in pixels)')
    parser.add_argument('--visibility', type=int, default=2, help='Visibility value to write for keypoint (useful when dim=3). Default 2 (visible).')
    parser.add_argument('--no-visibility', action='store_true', help='Produce dim=2 keypoint lines (no visibility values).')
    parser.add_argument('--exts', nargs='+', default=['.txt'], help='Label file extensions to process')
    args = parser.parse_args()

    labels_dir = args.labels_dir
    images_dir = args.images_dir
    out_dir = args.output_dir
    bbox_px = args.bbox_px
    include_visibility = not args.no_visibility
    visibility_val = args.visibility

    os.makedirs(out_dir, exist_ok=True)

    label_files = []
    for root, _, files in os.walk(labels_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in args.exts:
                label_files.append(os.path.join(root, fn))

    if not label_files:
        print('No label files found in', labels_dir)
        return

    total_in = 0
    total_out = 0
    missing_images = 0

    for lab in label_files:
        basename = os.path.splitext(os.path.basename(lab))[0]

        # determine image size
        img_w = img_h = None
        if args.default_size:
            img_w, img_h = args.default_size
        else:
            # try images_dir first
            if images_dir:
                img_path = find_image_path_for_label(lab, images_dir)
                if img_path:
                    im = Image.open(img_path)
                    img_w, img_h = im.size
                else:
                    # try same folder as label or parent
                    img_path = find_image_path_for_label(lab, os.path.dirname(lab))
                    if img_path:
                        im = Image.open(img_path)
                        img_w, img_h = im.size
            else:
                # try same folder as label
                img_path = find_image_path_for_label(lab, os.path.dirname(lab))
                if img_path:
                    im = Image.open(img_path)
                    img_w, img_h = im.size

        if img_w is None or img_h is None:
            missing_images += 1
            print(f"Warning: could not find image for label '{lab}'. Use --default-size WIDTH HEIGHT or provide --images-dir. Skipping this file.")
            continue

        out_lines = process_label_file(lab, img_w, img_h, bbox_px, include_visibility, visibility_val)

        # write output label file (same name) into out_dir preserving subfolder structure
        relpath = os.path.relpath(lab, labels_dir)
        out_path = os.path.join(out_dir, relpath)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'w') as f:
            for ln in out_lines:
                f.write(ln + '\n')

        total_in += 1
        total_out += len(out_lines)

    print('Processed label files:', total_in)
    print('Total output objects:', total_out)
    if missing_images:
        print('Skipped files due to missing images:', missing_images)


if __name__ == '__main__':
    main()
