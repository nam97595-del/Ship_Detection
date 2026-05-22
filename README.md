# Hệ Thống Phát Hiện Và Phân Loại Tàu Thuyền  

Công cụ kiểm thử và đánh giá hiệu năng mô hình YOLO (Object Tracking) được xây dựng bằng Python và Tkinter. Ứng dụng hỗ trợ chạy các model đã train (định dạng `.pt`, `.onnx`, `.engine`) trên video, tự động xuất báo cáo hiệu năng (FPS, thời gian xử lý) và video kết quả.

## 🚀 Tính Năng Chính

* **Giao diện đồ họa:** Dễ dàng chọn thư mục Model, Video, Tracker và Output mà không cần sửa code.
* **Hỗ trợ Tracking đa dạng:** Tích hợp sẵn các thuật toán tracking để đếm đối tượng độc nhất (Unique ID).
* **OCR:** Phát hiện và nhận dạng text trên tàu (PaddleOCR).
* **Báo cáo tự động:**
    * Xuất video kết quả (`.mp4`) có vẽ khung tracking.
    * Xuất file CSV chứa dữ liệu chi tiết từng phát hiện.
    * Lưu ảnh tàu phát hiện vào `output/ship_images/`.
    * Xuất file .TXT báo cáo tổng hợp.

## 📂 Cấu Trúc Dự Án

Dự án được tổ chức theo mô hình Modular để dễ dàng bảo trì và mở rộng:
```text
project_root/
│
├── src/
│   ├── controllers/        # Xử lý logic trung gian 
│   │   ├── __init__.py
│   │   ├── log_controller.py
│   │   └── main_controller.py
│   │
│   ├── engines/            # Xử lý AI / thuật toán
│   │   ├── __init__.py
│   │   ├── yolo_engine.py
│   │   ├── ocr_engine.py
│   │   ├── speed_estimator.py
│   ├── models/             # Model AI
│   │
│   ├── trackers/           # Tracker Config files (.yaml)
│   │   ├── botsort.yaml
│   │   ├── bytetrack.yaml
│   │
│   ├── views/              # Giao diện (UI)
│   │   ├── __init__.py
│   │   ├── log_view.py
│   │   └── main_view.py
│   │
│   ├── utils/              # Tiện ích
│   │   ├── __init__.py
│   │   ├── csv_logger.py
│   │   ├── export_engine.py
│   │   └── report_utils.py
│   │
│   └── main.py             
│
├── output/                 # Kết quả
│   ├── shiplog.csv         # Log nhật ký tàu
│   └── ship_images/        # Ảnh tàu phát hiện
│
├── requirements.txt        # Thư viện Python
├── .gitignore
└── README.md
```

## Yêu Cầu Hệ Thống & Cài Đặt

### Yêu Cầu Tối Thiểu
- **Python:** 3.9+
- **GPU:** NVIDIA RTX 1050 trở lên
- **CUDA:** 12.1+ (phù hợp với PyTorch 2.1.2+cu121)
- **cuDNN:** 8.x+
- **RAM:** 8GB+
- **Disk:** 10GB+ 

### Các Bước Cài Đặt

#### Bước 1: Chuẩn Bị Môi Trường
Sử dụng Virtual Environment để tránh xung đột thư viện:
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

#### Bước 2: Nâng Cấp pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

#### Bước 3: Cài Đặt Dependencies
Tùy theo cấu hình GPU của bạn, chọn phiên bản PyTorch phù hợp:

**Cho NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Cho CPU (không GPU):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

#### Bước 4: Kiểm Tra GPU (Tùy Chọn)
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Hướng Dẫn Sử Dụng

### Khởi Chạy Ứng Dụng
Chạy file `src/main.py` từ terminal:
```bash
python src/main.py
```

### Thiết Lập Thông Số
1. **Chọn Model:** 
   - Nhấn "..." trong phần "Model" → Chọn file model (`.pt`, `.engine`)

2. **Chọn Tracker:**
   - Combobox "Tracker" tự động quét thư mục `src/trackers/`
   - Chọn tracker muốn sử dụng
   - Mặc định: `bytetrack.yaml`

3. **Chọn Video:**
   - Nhấn "..." trong phần "Video" → Chọn file video (`.mp4`, `.avi`, `.mov`, v.v.)

4. **Chọn Output:**
   - Nhấn "..." → Chọn thư mục lưu kết quả
   - Kết quả sẽ được lưu vào `output/shiplog.csv` và `output/ship_images/`

### Cấu Hình Advanced
- **Image Size:** Kích thước ảnh đầu vào cho model (Mặc định: 640)
- **Stride:** Số frame bỏ qua (Mặc định: 3 - xử lý 1/3 frame)
- **Confidence:** Ngưỡng tự tin để lọc kết quả (Slider: 0.0 - 1.0)
- **2-Stage OCR:** Bật/Tắt nhận dạng text tàu (cần chọn Text Model)

### Chạy Tracking
1. Nhấn nút **"▶ BẮT ĐẦU"**
2. Video sẽ phát với tracking thời gian thực
3. Nhấn **"⏹ DỪNG"** hoặc đóng cửa sổ để kết thúc

### Xem Kết Quả
Sau khi chạy xong, vào thư mục `output/`:
- **shiplog.csv:** Nhật ký phát hiện (track_id, class, OCR text, tốc độ, v.v.)
- **ship_images/:** Ảnh crop của tàu phát hiện
- **video.mp4:** Video tracking kết quả (nếu được xuất)

## Ghi Chú Quan Trọng

### CSV Logging
- Mỗi lần chạy được gán **Session ID** (tên video + timestamp)
- **Unique ID** được tạo từ `session_id_track_id` để tránh trùng lặp
- Dữ liệu được **tự động cập nhật** khi track_id đã tồn tại trong cùng session

### Tracker Selection
- File tracker phải có đuôi `.yaml` hoặc `.yml`
- Đặt file trong thư mục `src/trackers/`
- Ứng dụng sẽ tự động quét và hiển thị danh sách

## 📊 File CSV

File `output/shiplog.csv` có các cột sau:

| Cột | Kiểu | Mô Tả |
|-----|------|-------|
| log_id | INT | ID tự tăng |
| unique_id | STR | Định danh duy nhất (session_id_track_id) |
| track_id | INT | ID phát hiện trong session hiện tại |
| session_id | STR | ID phiên (video_name + timestamp) |
| class_name | STR | Loại đối tượng (ví dụ: "speed_boat") |
| so_hieu_ocr | STR | Số hiệu tàu (từ OCR) |
| do_tin_cay_ocr | FLOAT | Độ tin cậy OCR (0-1) |
| gio_phat_hien | DATETIME | Thời gian phát hiện |
| hinh_anh_path | STR | Đường dẫn ảnh tàu |
| video_source | STR | Tên file video |

## 📦 Cấu Hình Khuyến Nghị

Dựa trên cấu hình của bạn **(RTX 1050, CUDA 12.1, Python 3.11)**:

```
Python 3.11
PyTorch 2.1.2 (CUDA 12.1)
Ultralytics 8.4.8
PaddleOCR 2.7.3
Pandas 3.0.0
OpenCV 4.6.0
```

**Lệnh cài đặt nhanh:**
```bash
# Tạo venv
python -m venv venv
venv\Scripts\activate

# Nâng cấp pip
python -m pip install --upgrade pip setuptools wheel

# Cài PyTorch CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài dependencies
pip install -r requirements.txt
```
---

**Cập nhật lần cuối:** 7/4/2026  
