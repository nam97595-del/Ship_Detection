from .yolo_detector import YoloDetector
from .rtdetr_detector import RTDETRDetector
def create_detector(model_type, model_path, conf_thresh, imgsz):
    if model_type == "YOLO":
        return YoloDetector(model_path, conf_thresh, imgsz)
    elif model_type == "DETR":
        return RTDETRDetector(model_path, conf_thresh, imgsz)
    else:
        raise ValueError(f"Chưa hỗ trợ mô hình: {model_type}")
    
