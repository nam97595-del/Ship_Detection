from ultralytics import YOLO
from ultralytics import RTDETR

model_path = r"E:\Download\yolo12x-fishv4\yolo12x-fish\weights\best.pt"
model = YOLO(model_path)
#model = RTDETR('E:\Download\yolo12x-fishv4\yolo12x-fish\weights\best.pt')
print("Bắt đầu xuất mô hình sang TensorRT")


model.export(
    format='engine',    # Xuất ra TensorRT
    imgsz=640,          # Input Size
    half=True,          # Bật lượng tử hóa FP16
    dynamic=False,      # TẮT dynamic để tối ưu tốc độ tối đa
    simplify=False,     # True = Rút gọn đồ thị nơ-ron
    workspace=6,        # Cho phép dùng 4GB VRAM để biên dịch
    batch=1             # Chạy từng frame một
)
print("Đã xuất xong file .engine!")