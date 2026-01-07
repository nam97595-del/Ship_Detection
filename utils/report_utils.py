import pandas as pd
import os
import numpy as np
import datetime

def save_test_report(data, all_confs, output_folder, video_name, processed_count, total_frames, model_name, imgsz, stride, conf_thresh, tag):
    # Tạo tên file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{timestamp}_{model_name}_{tag}_raw.csv"
    txt_name = f"{timestamp}_{model_name}_{tag}_REPORT.txt"
    
    csv_path = os.path.join(output_folder, csv_name)
    txt_path = os.path.join(output_folder, txt_name)

    # Lưu CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Tính toán thống kê
    avg_fps = df["FPS"].mean()
    avg_time = df["Time_ms"].mean()
    total_obj = df["Objects"].sum()
    overall_conf = np.mean(all_confs) if len(all_confs) > 0 else 0.0

    # Nội dung Report
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

2. KẾT QUẢ
   - Tổng frames gốc: {total_frames}
   - Số frame xử lý: {processed_count}
   - FPS Trung bình: {avg_fps:.2f}
   - Thời gian/frame: {avg_time:.2f} ms
   - Tổng objects: {total_obj}
   - Độ tin cậy TB: {overall_conf:.4f}
==========================================================
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # --- THAY ĐỔI Ở ĐÂY: Trả về cả đường dẫn VÀ nội dung report ---
    return txt_path, report