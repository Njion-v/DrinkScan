from ultralytics import YOLO
import numpy as np
import os

class DrinkModel:
    def __init__(self, config):
        self.model = YOLO(os.path.abspath(config["model_path"]))
        self.names = self.model.names
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]

    def infer(self, image):
        boxes = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0].boxes
        results = {}
        for cls, conf in zip(boxes.cls, boxes.conf):
            label = self.names[int(cls)]
            results[label] = results.get(label, 0) + 1
        return results
