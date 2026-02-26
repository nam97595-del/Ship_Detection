# YOLO Model Testing Tool (Modular Version)

Công cụ kiểm thử và đánh giá hiệu năng mô hình YOLO (Object Tracking) được xây dựng bằng Python và Tkinter. Ứng dụng hỗ trợ chạy các model đã train (định dạng `.pt`, `.onnx`, `.engine`) trên video, tự động xuất báo cáo hiệu năng (FPS, thời gian xử lý) và video kết quả.

<img width="601" height="727" alt="Image" src="https://github.com/user-attachments/assets/a2b14ab3-f0c7-4a68-9db2-2541f3f61363" />

## 🚀 Tính Năng Chính

* **Giao diện đồ họa (GUI):** Dễ dàng chọn thư mục Model, Video và Output mà không cần sửa code.
* **Hỗ trợ Tracking:** Tích hợp sẵn thuật toán tracking (BoTSORT/ByteTrack) của Ultralytics để đếm đối tượng độc nhất (Unique ID).
* **Cấu hình linh hoạt:** Tùy chỉnh `Image Size`, `Skip Frame` (Stride), và `Confidence Threshold` ngay trên giao diện.
* **Báo cáo tự động:**
    * Xuất video kết quả (`.mp4`) có vẽ khung tracking.
    * Xuất file CSV chứa dữ liệu chi tiết từng frame.
    * Xuất file TXT báo cáo tổng hợp (FPS trung bình, tổng số đối tượng phát hiện, v.v.).

## 📂 Cấu Trúc Dự Án

Dự án được tổ chức theo mô hình Modular để dễ dàng bảo trì và mở rộng:

```text
project_root
├── engines/
│   ├── __init__.py
│   ├── yolo_engine.py      # Logic YOLO: load model, tracking, vẽ bounding box
│   └── ocr_engine.py       # Logic OCR: nhận diện biển số/ID tàu (PaddleOCR)
├── gui/
│   ├── __init__.py
│   ├── login.py            # Giao diện đăng nhập hệ thống
│   └── main_window.py      # Giao diện chính điều khiển video và hiển thị kết quả
├── utils/
│   ├── __init__.py
│   ├── export_engine.py    # Xử lý logic xuất dữ liệu ra Excel/Database
│   └── report_utils.py     # Các hàm hỗ trợ định dạng báo cáo (CSV, TXT)
├── sql/
│   └── lenhsql.sql         # File chứa cấu trúc bảng và câu lệnh khởi tạo Database

├── video/                  # Chứa video mẫu để test (nên để file nhẹ)
│   └── vid.mp4
├── .gitignore              # Loại bỏ các file rác, __pycache__, venv
├── main.py                 # File chạy chính (khởi động Login -> Main Window)
├── README.md               # Tài liệu hướng dẫn cài đặt và sử dụng
└── requirements.txt        # Danh sách thư viện: ultralytics, paddleocr, PyQt6/Tkinter...
```

## Yêu Cầu Hệ Thống & Cài Đặt
Yêu cầu
Python 3.8 trở lên

Khuyến nghị sử dụng GPU (NVIDIA) để đạt tốc độ xử lý tốt nhất (cần cài đặt CUDA).

Các bước cài đặt

Bước 1: Clone dự án hoặc tải về máy.

Bước 2: Cài đặt các thư viện phụ thuộc. Nên sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.
```bash
pip install -r requirements.txt
```

Bước 3: Tiến hành setup cơ sở dữ liệu
Mở SSMS
Chọn server tương ứng
Nhấn New Query
Copy toàn bộ nội dung file lenhsql.sql
Nhấn Execute
Thay servername ở máy bạn ở file yolo_engine.py

```python
self.server = '.\\SQLEXPRESS'
```
Thay servername ở máy bạn ở file main_window.py

```python
'SERVER=.\\SQLEXPRESS;'
```

## Hướng Dẫn Sử Dụng

Bước 1: Khởi chạy ứng dụng. Chạy file main.py từ terminal hoặc IDE:
```bash
python main.py
```
Bước 2: Thiết lập thông số kiểm thử trên giao diện.
- Chọn Model: Nhấn "Chọn Folder chứa Models" -> Chọn file model (.pt) từ danh sách thả xuống.
- Chọn Video: Nhấn "Chọn Folder chứa Video" -> Chọn video cần test.
- Output: Chọn thư mục để lưu kết quả.

Cấu hình:
- Image Size: Kích thước ảnh đầu vào cho model (Mặc định: 640).
- Skip Frame: Số frame bỏ qua để tăng tốc độ (Mặc định: 3 - tức là xử lý 1 frame, bỏ qua 2 frame).
- Conf Thresh: Ngưỡng tự tin để lọc kết quả.

Bước 3: Chạy và xem kết quả. Nhấn nút "CHẠY TEST NGAY".
- Cửa sổ video sẽ hiện lên với thông tin Tracking thời gian thực.
- Nhấn phím q trên cửa sổ video để dừng sớm.

Bước 4: Xem báo cáo. Sau khi chạy xong, vào thư mục Output đã chọn, trong thư mục sẽ có:
- File video .mp4: Video đã được vẽ bounding box và ID.
- File report .txt: Tổng hợp thống kê.
- File data .csv: Dữ liệu chi tiết từng frame để vẽ biểu đồ.