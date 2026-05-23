# Ship Detection OCR Engine
Hệ thống AI backend dùng để **phát hiện tàu biển, theo dõi đối tượng và nhận diện mã hiệu tàu bằng OCR** từ video giám sát hàng hải.
Dự án được thiết kế theo hướng **module engine độc lập**, dễ dàng tích hợp vào các hệ thống giám sát tàu hoặc ứng dụng desktop/web khác.
GUI Tkinter trong project chỉ dùng cho mục đích **kiểm thử OCR và mô phỏng vận hành**.
---

# Chức năng chính
- Phát hiện tàu bằng YOLO
- Theo dõi tàu theo `Track ID`
- Detect vùng text trên thân tàu
- OCR mã hiệu tàu bằng PaddleOCR
- Trả kết quả

# Kiến trúc hệ thống
   Ảnh đầu vào
        │
        ▼
  Phát hiện tàu
        │
        ▼
Phát hiện vùng chữ
        │
        ▼
   Cắt vùng chữ
        │
        ▼
  Nhận dạng chữ
        │
        ▼
   Kết quả OCR

# Cấu trúc thư mục
engines/
│── ocr_engine.py
│── yolo_engine.py

gui/
│── main_window.py
main.py           # Run demo

# Engine Overview
## 1. OCR Engine (`ocr_engine.py`)
Chịu trách nhiệm:
- Load PaddleOCR
- OCR vùng text tàu
- Lọc confidence
- Trả text nhận diện

## 2. YOLO Engine (`yolo_engine.py`)
Chịu trách nhiệm:
- Detect tàu
- Tracking object
- OCR theo yêu cầu
- Quản lý Track ID
- Gửi frame cho GUI/app

# Tích hợp sang ứng dụng khác
Engine được thiết kế để dễ dàng tích hợp vào:
- Desktop Application
- Maritime Surveillance Software
- Camera Monitoring System
- Web Backend
- RTSP Streaming

# GUI Test
`main_window.py` chỉ là giao diện kiểm thử giúp:
- test detect
- test OCR
- click tàu để OCR
- debug text crop
Không phải thành phần bắt buộc của hệ thống.

