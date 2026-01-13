#!/usr/bin/env python3
# labelme2yolo_pose_seq.py
# python labelme2yolo_pose_seq.py ./outJson ./outTxtLabels --kpts CenterGrowthPoint
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="LabelMe → YOLO pose (sequential matching, no group_id)")
    parser.add_argument("json_dir", help="Directory containing LabelMe *.json")
    parser.add_argument("out_dir", help="Output directory for *.txt")
    parser.add_argument("--kpts", nargs="+", required=True,
                        help="List of keypoint labels in order")
    return parser.parse_args()


def convert_one(json_path, kpt_names):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = data["imageWidth"], data["imageHeight"]
    if img_w == 0 or img_h == 0:
        raise ValueError("imageWidth or imageHeight is zero")

    # 按出现顺序收集
    boxes = [s for s in data["shapes"] if s["shape_type"] == "rectangle"]
    points = [s for s in data["shapes"] if s["shape_type"] == "point"]

    n_kpt = len(kpt_names)
    if len(points) % n_kpt != 0 or len(points) // n_kpt != len(boxes):
        raise ValueError(
            f"点数 {len(points)} 不是 {n_kpt} 的整数倍，或框数 {len(boxes)} 不匹配")

    lines = []
    for i, box in enumerate(boxes):
        # 1. 矩形框
        (x1, y1), (x2, y2) = box["points"]
        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        cls_id = 0

        # 2. 取出该框对应的 n_kpt 个点
        pts = points[i * n_kpt: (i + 1) * n_kpt]

        # 建立 label → (x,y) 映射
        label2xy = {p["label"]: p["points"][0] for p in pts}

        kpt_line = []
        for name in kpt_names:
            if name in label2xy:
                x, y = label2xy[name]
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
