from .yolo_detector import YoloDetector

def create_detector(model_type, model_path, conf_thresh, imgsz):
    if model_type == "YOLO":
        return YoloDetector(model_path, conf_thresh, imgsz)
    else:
        raise ValueError(f"Chưa hỗ trợ mô hình: {model_type}")
    
