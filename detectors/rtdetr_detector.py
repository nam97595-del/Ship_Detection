import numpy as np
from ultralytics import RTDETR
from .base_detector import BaseDetector

class RTDETRDetector(BaseDetector):
    def load_model(self):
        print(f"Đang load model RTDETR từ: {self.model_path}")
        self.model = RTDETR(self.model_path)

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_thresh, imgsz=self.imgsz, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return np.empty((0, 0))
        
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)

        dets = np.hstack((xyxy, conf, cls))
        return dets