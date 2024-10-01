# Ultralytics YOLO_xyz 🚀, AGPL-3.0 license

import contextlib
import urllib
from copy import copy
from pathlib import Path

import pytest
from ultralytics import RTDETR, YOLO, YOLOWorld
from ultralytics.utils import (
    ASSETS,
    ONLINE,
    WEIGHTS_DIR
)


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def _model_tune():
    """Tune YOLO_xyz model for performance improvement."""
    YOLO("yolov8n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolov8n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")

def train_yolov10():
    """Test YOLOv10 model training, validation, and prediction steps with minimal configurations."""
    model = YOLO("yolov10x.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=100, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)

def _detect_yolo_v10():
    model = YOLO(WEIGHTS_DIR / "yolov10n.pt")
    # model = YOLO_xyz("yolov10.yaml")
    results = model(ASSETS / "bus.jpg")
    results[0].show()

# 导出操作暂时只能就地
def _test_for_onnx():
    # 加载训练好的模型
    # model = YOLOWorld("yolov10n-worldv2.yaml")
    model = YOLO("yolov8n-seg.yaml")
    # 将模型转为onnx格式
    success = model.export(format='onnx')
    print(success)

def train_for_pose():
    # Load a model
    model = YOLO("yolov10n-pose.yaml")  # build a new model from YAML
    # model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov10-pose.yaml").load("yolov10n-pose.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

def test_for_pose():
    # Load a model
    model = YOLO("yolov8n-pose.pt")  # load an official model
    model = YOLO("path/to/best.pt")  # load a custom model

    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

def train_for_obb():
    # Load a model
    model = YOLO("yolov10n-obb.yaml")  # build a new model from YAML
    # model = YOLO("yolov8n-obb.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov8n-obb.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(data="dota8.yaml", epochs=100, imgsz=640)

def test_for_obb():
    # Load a model
    model = YOLO("yolov8n-obb.pt")  # load an official model
    model = YOLO("path/to/best.pt")  # load a custom model
    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

def _train_yolov10():
    """Test YOLOv10 model training, validation, and prediction steps with minimal configurations."""
    model = YOLO("yolov10x.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=100, aimgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.save(WEIGHTS_DIR / "yolov10x.pt")

if __name__ == '__main__':
    _test_for_onnx()