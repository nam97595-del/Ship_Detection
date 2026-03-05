import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector

class YoloDetector(BaseDetector):
    def load_model(self):
        print(f"Đang load model YOLO từ: {self.model_path}")
        self.model_path = YOLO(self.model_path)

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
    
        