import os
import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector

class YoloDetector(BaseDetector):
    def load_model(self):
        if self.model_path.endswith('.xml'):
            load_path = os.path.dirname(self.model_path)
            print(f"🚀 Phát hiện OpenVINO, đang load model từ thư mục: {load_path}")
        else:
            load_path = self.model_path
            print(f"Đang load model YOLO từ: {load_path}")

        self.model = YOLO(load_path, task='detect')

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_thresh, imgsz=self.imgsz, verbose=False)
        boxes = results[0].boxes
        
        # Mảng rỗng nếu không có tàu
        if len(boxes) == 0:
            return np.empty((0, 6))
        
        # Ép kiểu dữ liệu về numpy [x1, y1, x2, y2, conf, cls]
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)

        dets = np.hstack((xyxy, conf, cls))
        return dets
    
        