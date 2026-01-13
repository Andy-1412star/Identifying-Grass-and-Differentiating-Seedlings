#!/usr/bin/env python3
"""
convert_points_to_boxes.py

将 X-AnyLabeling／LabelMe／COCO／VIA 等常见 JSON 标注文件中以点标注的玉米苗，
转换为以该点为中心、大小为 box_size x box_size 的矩形框（标签为 'corn'）。

用法示例:
  python convert_points_to_boxes.py input.json output.json --box_size 10 --label corn
                                                                    30

支持自动识别格式（labelme / coco / via / generic），也可通过 --format 指定。

脚本会先备份原始文件为 input.json.bak。

注意：
- 坐标可能为浮点或整数，输出会保留数值类型（默认使用整数）。
- 若 JSON 中存在图像宽高信息（imageWidth/imageHeight 或 COCO images 中的 width/height），
  会将生成的 bbox 限制在图像范围内。

"""

import json
import argparse
import os
import copy
import math
from typing import Any, Dict, List, Tuple


def clamp(v, a, b):
    return max(a, min(b, v))


def detect_format(data: Dict[str, Any]) -> str:
    if isinstance(data, dict):
        if 'shapes' in data and isinstance(data['shapes'], list):
            return 'labelme'
        if 'annotations' in data and 'images' in data:
            return 'coco'
        # VIA style
        if 'regions' in data and isinstance(data['regions'], (dict, list)):
            return 'via'
    # fallback
    return 'generic'


def labelme_convert(data: Dict[str, Any], box_size: int, label: str) -> Dict[str, Any]:
    """
    For LabelMe-style JSON: do NOT remove the original point shape.
    Instead, create a new rectangle shape (box_size x box_size) centered on the point
    and insert it *before* the original point shape in the shapes list.
    """
    half = box_size / 2.0
    w = data.get('imageWidth')
    h = data.get('imageHeight')
    orig_shapes = data.get('shapes', [])
    new_shapes: List[Dict[str, Any]] = []
    for shape in orig_shapes:
        pts = shape.get('points')
        inserted = False
        if pts:
            # LabelMe point often shape_type == 'point' and points = [[x,y]]
            if shape.get('shape_type') == 'point' or (isinstance(pts, list) and len(pts) == 1):
                try:
                    x, y = pts[0]
                    x1 = x - half
                    y1 = y - half
                    x2 = x + half
                    y2 = y + half
                    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                        x1 = clamp(x1, 0, w - 1)
                        y1 = clamp(y1, 0, h - 1)
                        x2 = clamp(x2, 0, w - 1)
                        y2 = clamp(y2, 0, h - 1)
                    # create a new rectangle shape and insert it BEFORE the original point
                    rect_shape = {
                        'label': label,
                        'points': [[x1, y1], [x2, y2]],
                        'shape_type': 'rectangle',
                        # keep flags if present to be safe
                        'flags': shape.get('flags', {})
                    }
                    new_shapes.append(rect_shape)
                    inserted = True
                except Exception:
                    # if any parsing issue, skip insertion but keep original
                    inserted = False
        # always keep the original shape (so the point is not removed)
        new_shapes.append(shape)
    # replace shapes with new ordered list (rect before point)
    data['shapes'] = new_shapes
    return data


def coco_get_image_wh_map(data: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    res = {}
    for img in data.get('images', []):
        img_id = img.get('id')
        if img_id is None:
            continue
        res[img_id] = (img.get('width'), img.get('height'))
    return res


def coco_find_or_create_category(data: Dict[str, Any], label: str) -> int:
    cats = data.get('categories')
    if cats is None:
        data['categories'] = []
        cats = data['categories']
    for c in cats:
        if c.get('name') == label:
            return c['id']
    # new id: max existing id +1 or 1
    maxid = 0
    for c in cats:
        if isinstance(c.get('id'), int):
            maxid = max(maxid, c['id'])
    new_id = maxid + 1 if maxid >= 1 else 1
    cats.append({'id': new_id, 'name': label, 'supercategory': ''})
    return new_id


def coco_convert(data: Dict[str, Any], box_size: int, label: str) -> Dict[str, Any]:
    half = box_size / 2.0
    img_wh = coco_get_image_wh_map(data)
    cat_id = coco_find_or_create_category(data, label)
    for ann in data.get('annotations', []):
        # try to extract a single keypoint
        kp = ann.get('keypoints')
        x = y = None
        if kp:
            # common COCO keypoints flatten: [x1,y1,v1, x2,y2,v2, ...]
            if isinstance(kp, list) and len(kp) >= 3:
                x = kp[0]
                y = kp[1]
            # sometimes keypoints stored as list of [x,y,v]
            elif isinstance(kp, list) and len(kp) == 2:
                x, y = kp
        # fallback: some annotations may store 'point' or 'center'
        if x is None or y is None:
            # try 'point' key
            if 'point' in ann and isinstance(ann['point'], (list, tuple)) and len(ann['point']) >= 2:
                x, y = ann['point'][:2]
            elif 'x' in ann and 'y' in ann:
                x, y = ann['x'], ann['y']
        if x is None or y is None:
            # cannot find a point for this annotation; skip
            continue
        # compute bbox [x,y,w,h] with top-left
        x1 = float(x) - half
        y1 = float(y) - half
        w = float(box_size)
        h = float(box_size)
        # clamp if image size known
        img_id = ann.get('image_id')
        if img_id in img_wh and img_wh[img_id] is not None:
            iw, ih = img_wh[img_id]
            if iw is not None and ih is not None:
                x1 = clamp(x1, 0, iw - 1)
                y1 = clamp(y1, 0, ih - 1)
                # ensure width/height do not go out of bounds
                w = min(w, iw - x1)
                h = min(h, ih - y1)
        ann['bbox'] = [x1, y1, w, h]
        ann['category_id'] = cat_id
    return data


def via_convert(data: Dict[str, Any], box_size: int, label: str) -> Dict[str, Any]:
    """
    For VIA-style JSON: keep original region (point) and add a new rect region before it when possible.
    If regions are stored as a dict, add a new keyed region (key names may not preserve visual ordering in a dict).
    """
    half = box_size / 2.0
    regions = data.get('regions')
    if isinstance(regions, dict):
        # create new dict; preserve existing entries and add new keyed entries right before by building a new dict
        new_regions = {}
        for key, region in regions.items():
            sa = region.get('shape_attributes', {})
            name = sa.get('name')
            if name == 'point':
                cx = sa.get('cx') if 'cx' in sa else sa.get('x')
                cy = sa.get('cy') if 'cy' in sa else sa.get('y')
                if cx is not None and cy is not None:
                    x = float(cx) - half
                    y = float(cy) - half
                    w = float(box_size)
                    h = float(box_size)
                    # new key: original key with suffix to avoid collision
                    new_key = f"{key}_rect"
                    sa_new = {'name': 'rect', 'x': int(round(x)), 'y': int(round(y)), 'width': int(round(w)), 'height': int(round(h))}
                    new_region = {
                        'shape_attributes': sa_new,
                        'region_attributes': {'label': label}
                    }
                    # insert new region before original
                    new_regions[new_key] = new_region
            # keep original
            new_regions[key] = region
        data['regions'] = new_regions
    elif isinstance(regions, list):
        new_list = []
        for region in regions:
            sa = region.get('shape_attributes', {})
            name = sa.get('name')
            if name == 'point':
                cx = sa.get('cx') if 'cx' in sa else sa.get('x')
                cy = sa.get('cy') if 'cy' in sa else sa.get('y')
                if cx is not None and cy is not None:
                    x = float(cx) - half
                    y = float(cy) - half
                    w = float(box_size)
                    h = float(box_size)
                    sa_new = {'name': 'rect', 'x': int(round(x)), 'y': int(round(y)), 'width': int(round(w)), 'height': int(round(h))}
                    new_region = {'shape_attributes': sa_new, 'region_attributes': {'label': label}}
                    # insert rectangle region BEFORE the original point region
                    new_list.append(new_region)
            # keep original region (so the point remains)
            new_list.append(region)
        data['regions'] = new_list
    return data


def generic_recursive_convert(obj: Any, box_size: int, label: str, image_w: int = None, image_h: int = None) -> Any:
    """
    Recursively search for point annotations but DO NOT delete them. Instead, add a bbox field
    while preserving the original point fields.
    Supported point representations:
      - {'points': [[x,y]]}
      - {'x': num, 'y': num} or {'cx': num, 'cy': num}
      - {'keypoints': [x,y,...]}

    When found, add a 'bbox' key with [x_top_left, y_top_left, w, h] and keep the original keys.
    """
    half = box_size / 2.0
    if isinstance(obj, dict):
        new = dict(obj)
        # case points single: keep original 'points', add bbox_points if desired
        pts = new.get('points')
        if isinstance(pts, list) and len(pts) == 1 and isinstance(pts[0], (list, tuple)):
            x, y = pts[0][0], pts[0][1]
            x1 = float(x) - half
            y1 = float(y) - half
            x2 = float(x) + half
            y2 = float(y) + half
            if image_w and image_h:
                x1 = clamp(x1, 0, image_w - 1)
                y1 = clamp(y1, 0, image_h - 1)
                x2 = clamp(x2, 0, image_w - 1)
                y2 = clamp(y2, 0, image_h - 1)
            # add both a bbox and a rect_points field, keep original points
            new.setdefault('rect_points', [[x1, y1], [x2, y2]])
            new.setdefault('bbox', [x1, y1, float(box_size), float(box_size)])
            # do not remove original 'points'
            return new
        # case x,y or cx,cy: keep original and add bbox
        if ('x' in new and 'y' in new) or ('cx' in new and 'cy' in new):
            if 'x' in new and 'y' in new:
                x = new['x']; y = new['y']
            else:
                x = new['cx']; y = new['cy']
            x1 = float(x) - half
            y1 = float(y) - half
            w = float(box_size)
            h = float(box_size)
            if image_w and image_h:
                x1 = clamp(x1, 0, image_w - 1)
                y1 = clamp(y1, 0, image_h - 1)
                w = min(w, image_w - x1)
                h = min(h, image_h - y1)
            new.setdefault('bbox', [x1, y1, w, h])
            return new
        # case keypoints: keep original, add bbox
        if 'keypoints' in new:
            kp = new.get('keypoints')
            if isinstance(kp, list) and len(kp) >= 2:
                x = kp[0]; y = kp[1]
                x1 = float(x) - half
                y1 = float(y) - half
                new.setdefault('bbox', [x1, y1, float(box_size), float(box_size)])
                return new
        # otherwise recurse into children
        for k, v in list(new.items()):
            new[k] = generic_recursive_convert(v, box_size, label, image_w, image_h)
        return new
    elif isinstance(obj, list):
        return [generic_recursive_convert(x, box_size, label, image_w, image_h) for x in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description='Convert point annotations to small bounding boxes.')
    parser.add_argument('input', help='input json file')
    parser.add_argument('output', nargs='?', help='output json file (default: input_converted.json)')
    parser.add_argument('--box_size', type=int, default=10, help='box size in pixels (default 10)')
    parser.add_argument('--label', type=str, default='corn', help="label name to use for new boxes (default 'corn')")
    parser.add_argument('--format', type=str, choices=['auto', 'labelme', 'coco', 'via', 'generic'], default='auto', help='input json format (auto detect by default)')
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or (os.path.splitext(in_path)[0] + '_converted.json')
    box_size = args.box_size
    label = args.label
    fmt = args.format

    if not os.path.exists(in_path):
        print(f'ERROR: input file not found: {in_path}')
        return

    # backup
    bak = in_path + '.bak'
    if not os.path.exists(bak):
        with open(in_path, 'rb') as fsrc, open(bak, 'wb') as fdst:
            fdst.write(fsrc.read())
        print(f'Created backup: {bak}')

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if fmt == 'auto':
        detected = detect_format(data)
    else:
        detected = fmt
    print(f'Detected format: {detected}')

    if detected == 'labelme':
        out = labelme_convert(data, box_size, label)
    elif detected == 'coco':
        out = coco_convert(data, box_size, label)
    elif detected == 'via':
        out = via_convert(data, box_size, label)
    else:
        # try to find image size
        iw = data.get('imageWidth') or data.get('image_width') or None
        ih = data.get('imageHeight') or data.get('image_height') or None
        out = generic_recursive_convert(data, box_size, label, iw, ih)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'Wrote converted annotations to: {out_path}')


if __name__ == '__main__':
    main()
