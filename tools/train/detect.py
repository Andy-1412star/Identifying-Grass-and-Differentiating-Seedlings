# -*- coding: utf-8 -*-

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时
os.environ['OMP_NUM_THREADS'] = '1'
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(model=r'runs/train/all2_corn_8s/weights/best.pt')
    model.predict(source=r'corn_datasets/test/images',
                  save=True,
                  show=False,
                  )
