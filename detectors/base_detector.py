from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self, model_path, conf_thresh, imgsz):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect(self, frame):
        # Trả về định dạng array
        # [x1, y1, x2, y2, conf, cls_id]
        pass