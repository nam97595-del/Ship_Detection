Hệ Thống Giám Sát Tàu Biển (Ship Detection & OCR)
Dự án tích hợp mô hình YOLOv12 để phát hiện, theo dõi (tracking) tàu biển và thư viện PaddleOCR để nhận diện mã số/tên tàu từ video giám sát. Giao diện được xây dựng trên nền tảng Tkinter.
Tính năng chính
Real-time Tracking: Phát hiện và theo dõi tàu biển với độ chính xác cao bằng YOLO.
On-demand OCR: Chỉ thực hiện nhận diện chữ khi người dùng Click chuột vào đối tượng tàu, giúp tối ưu hiệu năng (không gây lag video).
Báo cáo thông minh: Tự động xuất báo cáo kết quả gồm FPS, số lượng đối tượng và nội dung OCR sau khi kết thúc video.
Giao diện trực quan: Xem video trực tiếp, xem ảnh cắt (crop) của tàu và kết quả nhận diện ngay trên màn hình.
Yêu cầu hệ thống
Python: 3.9 - 3.11
Phần cứng: Khuyến nghị có GPU NVIDIA , hoặc CPU Intel.
Cài đặt các thư viện cần thiết: pip install -r requirements.txt
Hướng dẫn sử dụng
Bước 1: Khởi chạy ứng dụng 
Chạy file main.py từ terminal hoặc IDE
Bước 2: 
Cấu hình trên giao diện
Chọn Model: Tìm đến file weight của YOLO (ví dụ: yolov12x_fish-speed-pass.pt).
Chọn Video: Chọn video tàu biển cần phân tích.
Thư mục lưu KQ: Chọn nơi sẽ lưu video kết quả và file báo cáo (Report).
Bật OCR: Tích chọn "Bật OCR" nếu bạn muốn nhận diện tên tàu.
 <img width="464" height="652" alt="image" src="https://github.com/user-attachments/assets/c6a6e8cb-159b-4162-a8f1-a89d683753b3" />

Bước 3: Thao tác khi đang chạy
Nhấn BẮT ĐẦU để chạy xử lý.
Click chuột trái trực tiếp vào khung hình (Bounding Box) của một con tàu trên video.
Hệ thống sẽ ngay lập tức cắt ảnh con tàu đó và hiển thị tại cột CHI TIẾT TÀU, sau đó trả kết quả OCR (tên tàu) sau vài giây.
 <img width="975" height="771" alt="image" src="https://github.com/user-attachments/assets/72054faa-8f37-4818-9eda-7aa158a02e75" />

Kết quả đầu ra
Sau khi nhấn DỪNG hoặc video kết thúc:
Video: Một file .mp4 có vẽ khung hình và tên tàu sẽ được lưu trong thư mục Output.
Báo cáo: File báo cáo (Excel/CSV/JSON tùy thuộc vào report_utils.py) chứa các thống kê về hiệu năng và danh sách các tàu đã nhận diện được.
 <img width="408" height="684" alt="image" src="https://github.com/user-attachments/assets/0f83273d-8106-4880-827a-3df9aa1f64d7" />

 <img width="586" height="302" alt="image" src="https://github.com/user-attachments/assets/1215e80e-d003-4ede-996d-7ab7d16368c5" />

 
 

