from ultralytics import YOLO
from ultralytics import RTDETR

#model = YOLO('models/yolo12m-fish3.pt')
model = RTDETR('E:\Lab Nghien Cuu\Ship_Boat\model_rf\detrL-fish3(5).pt')
print("Bắt đầu xuất mô hình sang TensorRT")


model.export(
    format='engine',    # Xuất ra TensorRT
    imgsz=640,          # Input Size
    half=True,          # Bật lượng tử hóa FP16
    dynamic=False,      # TẮT dynamic để tối ưu tốc độ tối đa
    simplify=True,      # Rút gọn đồ thị nơ-ron
    workspace=4,        # Cho phép dùng 4GB VRAM để biên dịch
    batch=1             # Chạy từng frame một
)
print("Đã xuất xong file .engine!")