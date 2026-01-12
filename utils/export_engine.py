from ultralytics import YOLO

model = YOLO('model_yolo/ResumeYolo12l-fish-speed-pass-2.pt')

model.export(format='engine', half=True, device=0, imgsz=640)