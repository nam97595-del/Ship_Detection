import pandas as pd
import os
import numpy as np
import datetime

def save_test_report(data, all_confs, output_folder, video_name, processed_count, total_frames, model_name, imgsz, stride, conf_thresh, tag):
    # 1. Tạo tên file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{timestamp}_{model_name}_{tag}_raw.csv"
    txt_name = f"{timestamp}_{model_name}_{tag}_REPORT.txt"
    
    csv_path = os.path.join(output_folder, csv_name)
    txt_path = os.path.join(output_folder, txt_name)

    # 2. Tạo DataFrame và lưu CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # 3. Tính toán thống kê cơ bản
    avg_fps = df["FPS"].mean()
    avg_time = df["Time_ms"].mean()
    total_detections = df["Objects_In_Frame"].sum() if "Objects_In_Frame" in df.columns else df["Objects"].sum()
    overall_conf = np.mean(all_confs) if len(all_confs) > 0 else 0.0

    # 4. Xử lý phần Tracking (MỚI)
    # Kiểm tra xem trong dữ liệu có cột "Total_Unique_Objects" không
    tracking_line = ""
    if "Total_Unique_Objects" in df.columns:
        # Lấy giá trị lớn nhất trong cột (vì số lượng tracking tăng dần)
        unique_count = df["Total_Unique_Objects"].max()
        tracking_line = f"\n   - TRACKING (Số đối tượng thực tế): {unique_count}"

    # 5. Soạn nội dung Report
    report = f"""
==========================================================
BÁO CÁO TEST TỰ ĐỘNG - {model_name}
==========================================================
Thời gian: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video Input: {video_name}
----------------------------------------------------------
1. CẤU HÌNH
   - Model: {model_name}
   - Size: {imgsz}
   - Skip Frame: {stride}
   - Confidence Thresh: {conf_thresh}

2. KẾT QUẢ HIỆU NĂNG
   - Tổng frames gốc: {total_frames}
   - Số frame đã xử lý: {processed_count}
   - FPS Trung bình: {avg_fps:.2f}
   - Thời gian xử lý/frame: {avg_time:.2f} ms

3. KẾT QUẢ NHẬN DIỆN
   - Độ tin cậy trung bình (Confidence): {overall_conf:.4f}
   - Tổng lượt phát hiện (All Detections): {total_detections}{tracking_line}
==========================================================
    """
    
    # 6. Ghi file text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # 7. Trả về đường dẫn và nội dung để hiển thị lên GUI
    return txt_path, report