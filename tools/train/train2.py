import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时
os.environ['OMP_NUM_THREADS'] = '1'

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-pose.pt')  # 推荐使用预训练 pose 模型

    model.train(
        data='aug_corn_datasets/data.yaml',
        imgsz=768,
        epochs=300,
        batch=4,
        workers=0,
        optimizer='AdamW',
        lr0=1e-3,
        lrf=1e-4,
        close_mosaic=0,
        patience=50,
        device=0,
        project='runs/train',
        name='corn_exp_v2',
        single_cls=True,
        cache=False,
    )
