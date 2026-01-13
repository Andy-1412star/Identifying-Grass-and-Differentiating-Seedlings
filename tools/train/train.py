# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/v8/yolov8-pose.yaml')
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'corn_datasets/data.yaml',
                imgsz=1024,
                epochs=100,
                batch=4,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='corn_exp',
                single_cls=False,
                cache=False,
                )
