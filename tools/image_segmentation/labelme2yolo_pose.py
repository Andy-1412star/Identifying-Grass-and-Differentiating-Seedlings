#!/usr/bin/env python3
# labelme2yolo_pose_multi.py
import os
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="LabelMe → YOLO pose (multi-plant)")
    parser.add_argument("json_dir", help="Directory containing LabelMe *.json")
    parser.add_argument("out_dir", help="Output directory for *.txt")
    parser.add_argument("--kpts", nargs="+", required=True,
                        help="List of keypoint labels in order, e.g. --kpts root stem_base stem_mid leaf_tip1 leaf_tip2")
    return parser.parse_args()


def convert_one(json_path, kpt_names):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = data["imageWidth"], data["imageHeight"]
    if img_w == 0 or img_h == 0:
        raise ValueError("imageWidth or imageHeight is zero")

    # 按 group_id 分组；没有 group_id 的框/点视为独立组
    groups = {}
    for s in data["shapes"]:
        gid = s.get("group_id")
        groups.setdefault(gid, {"boxes": [], "points": []})
        if s["shape_type"] == "rectangle":
            groups[gid]["boxes"].append(s)
        elif s["shape_type"] == "point":
            groups[gid]["points"].append(s)

    lines = []
    for gid, group in groups.items():
        boxes = group["boxes"]
        points = group["points"]
        for box in boxes:
            # 解析矩形框
            (x1, y1), (x2, y2) = box["points"]
            x_c = ((x1 + x2) / 2) / img_w
            y_c = ((y1 + y2) / 2) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            cls_id = 0

            # 收集该框对应的关键点：group_id + label 双重匹配
            pts = {p["label"]: p["points"][0] for p in points}
            kpt_line = []
            for name in kpt_names:
                if name in pts:
                    x, y = pts[name]
                    kpt_line.extend([x / img_w, y / img_h, 2])
                else:
                    kpt_line.extend([0, 0, 0])

            line = " ".join(map(str, [cls_id, x_c, y_c, w, h, *kpt_line]))
            lines.append(line)
    return lines


def main():
    args = parse_args()
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kpt_names = args.kpts

    for json_file in json_dir.rglob("*.json"):
        try:
            lines = convert_one(json_file, kpt_names)
            txt_file = out_dir / (json_file.stem + ".txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
            print(f"[OK] {json_file.name} → {len(lines)} 株")
        except Exception as e:
            print(f"[ERR] {json_file.name}: {e}")


if __name__ == "__main__":
    main()
