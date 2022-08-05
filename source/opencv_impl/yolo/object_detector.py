import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, weights_path: str, cfg_path: str, classes_allowed: list, scale: float = 1.0, size: tuple=(832, 832)):
        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=size, scale=scale)
        self.classes_allowed = classes_allowed


    def detect_objects(self, img, threshold:float=0.4):
        # Detect Objects
        objects_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=threshold)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue

            if self.classes_allowed is None or class_id in self.classes_allowed:
                objects_boxes.append(box)

        return objects_boxes

