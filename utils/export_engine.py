from ultralytics import YOLO

model = YOLO('model_yolo/ShipDetect_yolo12l_16k.pt')

model.export(format='engine', half=True, device=0, imgsz=1280)