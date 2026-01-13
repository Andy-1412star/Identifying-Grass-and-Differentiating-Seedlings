#!/usr/bin/env python3
"""
batch_convert_folder.py

批量把文件夹内的 JSON 用 convert_points_to_boxes.py 转换（在每个点前插入 box）。
用法:
  python batch_convert_folder.py /path/to/json_folder --box_size 10 --label corn --outdir /path/to/output

默认会把转换后文件放在同一目录，文件名加后缀 .json。
要求：同目录下需有 convert_points_to_boxes.py（或修改下面的 `CONVERTER_SCRIPT` 为脚本绝对路径）。
"""
import argparse
import os
import subprocess
from pathlib import Path

CONVERTER_SCRIPT = "convert_point_box.py"  # 若不在同目录，改为绝对路径

def is_json_file(p: Path):
    return p.is_file() and p.suffix.lower() == ".json" and not p.name.endswith("_converted.json")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", help="包含 JSON 文件的文件夹路径")
    p.add_argument("--outdir", help="输出文件夹（默认放回原目录）", default=None)
    p.add_argument("--box_size", type=int, default=10, help="方框大小（像素）")
    p.add_argument("--label", default="corn", help="方框标签名称")
    p.add_argument("--format", default="auto", choices=["auto","labelme","coco","via","generic"], help="输入格式，默认 auto")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print("ERROR: input_dir 不存在或不是文件夹：", input_dir)
        return

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_dir.iterdir() if is_json_file(f)])
    if not files:
        print("没有找到 JSON 文件 (或已经都是 _converted.json)，路径：", input_dir)
        return

    ok = 0
    fail = 0
    for f in files:
        out_name = f.stem + ".json"
        out_path = (outdir / out_name) if outdir else (f.parent / out_name)
        cmd = [
            "python", CONVERTER_SCRIPT,
            str(f), str(out_path),
            "--box_size", str(args.box_size),
            "--label", args.label,
            "--format", args.format
        ]
        print("-> 处理:", f.name)
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode == 0:
                print("   OK ->", out_path.name)
                ok += 1
            else:
                print("   FAIL -> returncode", res.returncode)
                print("   STDOUT:", res.stdout)
                print("   STDERR:", res.stderr)
                fail += 1
        except Exception as e:
            print("   EXCEPTION:", e)
            fail += 1

    print(f"\n完成：总文件 {len(files)}，成功 {ok}，失败 {fail}。")
    if outdir:
        print("输出文件夹：", outdir.resolve())

if __name__ == "__main__":
    main()
