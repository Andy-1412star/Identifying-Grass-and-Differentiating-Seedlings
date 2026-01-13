#!/usr/bin/env python3
# \"\"\"yolov8_count_keypoints.py
# Run a YOLOv8 keypoint/detection model (best.pt) on a set of images and save counts per image to a CSV.
#
# Usage examples:
#   python yolov8_count_keypoints.py --weights best.pt --source tests/ --output counts.csv
#   python yolov8_count_keypoints.py --weights best.pt --source test_list.txt --output results.csv
#   python yolov8_count_keypoints.py --weights best.pt --source single_image.jpg --output out.csv
#
# Notes:
#  - Requires ultralytics (YOLOv8) installed: pip install ultralytics
#  - For large test sets you can use --batch to run prediction in batches (faster).
#  - The script will try multiple ways to count detections so it works for detection/keypoint models.
# \"\"\"
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description='Run YOLOv8 model and save detection counts per image to CSV.')
    p.add_argument('--weights', '-w', required=True, help='Path to weights file (e.g. best.pt)')
    p.add_argument('--source', '-s', required=True,
                   help='Folder of images, single image, or a text file listing image paths (one per line)')
    p.add_argument('--output', '-o', default='counts.csv', help='Output CSV path (default: counts.csv)')
    p.add_argument('--imgsz', type=int, default=640, help='Image size for inference (default: 640)')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    p.add_argument('--batch', type=int, default=8,
                   help='Batch size for prediction (default: 8). Use 1 for sequential processing.')
    p.add_argument('--device', default=None,
                   help='Device for inference (e.g. 0 or "cpu"). Default lets ultralytics choose.')
    p.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                   help='Image extensions to consider when source is a directory.')
    return p.parse_args()


def list_images_from_dir(dirpath, exts):
    p = Path(dirpath)
    imgs = []
    for e in exts:
        imgs.extend(p.rglob(f'*{e}'))
    imgs = sorted(set(imgs))
    return [str(x) for x in imgs]


def list_images_from_txt(txtfile):
    with open(txtfile, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def count_from_result(r):
    # 1) Try boxes length (most reliable for detection/keypoint models that include boxes)
    try:
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            try:
                # Boxes implements __len__ in many versions
                n = len(boxes)
                return int(n)
            except Exception:
                pass
            # fallback: try shapes on common attributes
            for attr in ('xyxy', 'xywh', 'data'):
                arr = getattr(boxes, attr, None)
                if arr is not None:
                    try:
                        import numpy as _np
                        return int(_np.asarray(arr).shape[0])
                    except Exception:
                        try:
                            return int(len(arr))
                        except Exception:
                            pass
    except Exception:
        pass

    # 2) Try keypoints (for pure keypoint outputs)
    try:
        kps = getattr(r, 'keypoints', None)
        if kps is not None:
            try:
                return int(len(kps))
            except Exception:
                arr = getattr(kps, 'xy', None)
                if arr is not None:
                    try:
                        import numpy as _np
                        return int(_np.asarray(arr).shape[0])
                    except Exception:
                        pass
    except Exception:
        pass

    # 3) Try masks (if present)
    try:
        masks = getattr(r, 'masks', None)
        if masks is not None:
            arr = getattr(masks, 'data', None)
            if arr is not None:
                try:
                    import numpy as _np
                    return int(_np.asarray(arr).shape[0])
                except Exception:
                    try:
                        return int(len(arr))
                    except Exception:
                        pass
    except Exception:
        pass

    # 4) Last-resort attempt: try attributes that may contain per-instance lists
    for attr in ('probs', 'items', 'pred', 'labels'):
        val = getattr(r, attr, None)
        if val is not None:
            try:
                return int(len(val))
            except Exception:
                pass

    # Nothing found — return 0
    return 0


def main():
    args = parse_args()

    # Check ultralytics availability
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ERROR: ultralytics package not found. Install with: pip install ultralytics', file=sys.stderr)
        raise SystemExit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f'ERROR: weights not found: {weights_path}', file=sys.stderr)
        raise SystemExit(1)

    # Build image list from source
    source = Path(args.source)
    if source.is_dir():
        image_list = list_images_from_dir(source, args.extensions)
    elif source.is_file() and source.suffix.lower() in ['.txt', '.list']:
        image_list = list_images_from_txt(source)
    elif source.is_file():
        # single image file
        image_list = [str(source)]
    else:
        # maybe a glob pattern
        import glob
        image_list = sorted(glob.glob(args.source))
        if not image_list:
            print(f'ERROR: No images found for source: {args.source}', file=sys.stderr)
            raise SystemExit(1)

    if len(image_list) == 0:
        print('No images found. Exiting.', file=sys.stderr)
        raise SystemExit(1)

    print(f'Found {len(image_list)} images. Loading model...')

    model = YOLO(str(weights_path))

    results_rows = []
    # If batch >1, use model.predict to process in batches. We'll try that path first for speed.
    use_batch_predict = args.batch and args.batch > 1

    if use_batch_predict:
        # ultralytics model.predict accepts a list of paths and returns results list
        # We'll run in chunks to avoid OOM
        chunk = args.batch
        for i in tqdm(range(0, len(image_list), chunk), desc='Batches'):
            batch_paths = image_list[i:i + chunk]
            try:
                res_list = model.predict(source=batch_paths, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                         verbose=False)
            except TypeError:
                # some ultralytics versions use predict(**kwargs) differently; fallback to calling with just source
                res_list = model.predict(batch_paths, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                         verbose=False)
            for img_path, r in zip(batch_paths, res_list):
                cnt = count_from_result(r)
                results_rows.append({'image_path': img_path, 'image_name': Path(img_path).name, 'count': cnt})
    else:
        # Sequential per-image inference (safer, but slower)
        for img_path in tqdm(image_list, desc='Images'):
            try:
                rlist = model.predict(source=[img_path], imgsz=args.imgsz, conf=args.conf, device=args.device,
                                      verbose=False)
                # predict with single image returns a list; take first element
                if isinstance(rlist, (list, tuple)) and len(rlist) > 0:
                    r = rlist[0]
                else:
                    r = rlist
            except Exception:
                # fallback: model(img_path)
                try:
                    r = model(img_path, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                except Exception as e:
                    print(f'Warning: inference failed for {img_path}: {e}', file=sys.stderr)
                    r = None
            if r is None:
                cnt = 0
            else:
                cnt = count_from_result(r)
            results_rows.append({'image_path': img_path, 'image_name': Path(img_path).name, 'count': cnt})

    # Save to CSV
    df = pd.DataFrame(results_rows, columns=['image_path', 'image_name', 'count'])
    out_path = Path(args.output)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Done. Saved counts for {len(df)} images to {out_path}')


if __name__ == '__main__':
    main()
