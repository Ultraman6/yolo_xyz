import os

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors



def crop_obj(img, results, names, classes=None):
    """
    Crop detected objects from the image and save them to the disk.

    Args:
        img (np.ndarray): Image to crop objects from.
        detections (List[Detection]): List of detected objects.
        names (List[str]): List of class names.
    """
    res_dict = {}
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(img, line_width=2, example=names)
    if boxes is not None:
        for box, cls in zip(boxes, clss):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
            crop_obj = img[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            if classes is None or cls in classes:
                if names[int(cls)] not in res_dict:
                    res_dict[names[int(cls)]] = []
                else:
                    res_dict[names[int(cls)]].append(crop_obj)
    return res_dict

if __name__ == "__main__":
    classes = ["person", "car"]  # example class names
    model = YOLO("yolov8n.pt")
    img = cv2.imread("F:/Github/YOLO/ultralytics/assets/bus.jpg")
    results = model(img)
    crop_obj(img, results, model.names, classes)