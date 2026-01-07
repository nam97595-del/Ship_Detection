import cv2
import os
import time
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Chọn Video để Test", filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    return file_path

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Chọn Thư mục lưu kết quả")
    return folder_path

def run_yolo_test_v2(model_path):
    # 1. Setup đường dẫn
    video_path = select_file()
    if not video_path:
        print("Đã hủy chọn video.")
        return

    output_folder = select_folder()
    if not output_folder:
        print("Đã hủy chọn thư mục lưu.")
        return

    # 2. Load Model
    print(f"🔹 Đang load mô hình: {os.path.basename(model_path)}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # 3. Chuẩn bị file output
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tên các file kết quả
    output_video_name = f"{timestamp}_{model_name}_video.mp4"
    raw_csv_name = f"{timestamp}_{model_name}_raw_metrics.csv"
    report_name = f"{timestamp}_{model_name}_SUMMARY_REPORT.txt"
    
    output_video_path = os.path.join(output_folder, output_video_name)
    raw_csv_path = os.path.join(output_folder, raw_csv_name)
    report_path = os.path.join(output_folder, report_name)

    # 4. Config Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_input, (orig_width, orig_height))

    # 5. Cài đặt hiển thị (Tránh tràn màn hình)
    window_name = f"Test Result - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720) # Resize với cửa số thích hợp

    # List lưu dữ liệu thô
    frame_data = []
    
    # List lưu confidences toàn bộ video để tính trung bình
    all_confidences = []

    print(f"\n🚀 Bắt đầu test video ({total_frames_input} frames)... Nhấn 'q' để dừng sớm.")

    frame_idx = 0
    
    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_idx += 1

        # --- INFERENCE ---
        # conf=0.25: Chỉ lấy box có độ tin cậy > 25%
        results = model(frame, verbose=False, conf=0.7, iou=0.5)
        
        # Lấy thông số thời gian thực tế đo được (bao gồm preprocess + inference + postprocess)
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000
        current_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # --- XỬ LÝ KẾT QUẢ DETECT ---
        boxes = results[0].boxes
        num_objects = len(boxes)
        
        # Lấy danh sách độ tin cậy (confidence scores) trong frame này
        # Chuyển sang CPU và numpy để tính toán
        confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        if len(confs) > 0:
            avg_conf_frame = float(np.mean(confs))
            all_confidences.extend(confs) # Gom vào list tổng
        else:
            avg_conf_frame = 0.0

        # Vẽ hình
        annotated_frame = results[0].plot()

        # Hiển thị thông số lên video
        # Vẽ nền đen mờ để chữ dễ đọc hơn
        cv2.rectangle(annotated_frame, (0, 0), (400, 120), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Avg Conf: {avg_conf_frame:.2f}", (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)

        # Lưu dữ liệu thô (Bỏ qua frame đầu tiên vì thường bị lag khởi động)
        if frame_idx > 1:
            frame_data.append({
                "Frame": frame_idx,
                "FPS": round(current_fps, 2),
                "Inference_Time_ms": round(process_time_ms, 2),
                "Objects_Detected": num_objects,
                "Avg_Confidence": round(avg_conf_frame, 4)
            })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã dừng bởi người dùng.")
            break

    # 6. Dọn dẹp
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 7. TÍNH TOÁN & XUẤT BÁO CÁO (AGGREGATION)
    if frame_data:
        df = pd.DataFrame(frame_data)
        
        # Tính toán các chỉ số tổng hợp
        avg_fps = df["FPS"].mean()
        avg_time = df["Inference_Time_ms"].mean()
        min_fps = df["FPS"].min()
        max_fps = df["FPS"].max()
        
        total_objects_detected = df["Objects_Detected"].sum()
        avg_objects_per_frame = df["Objects_Detected"].mean()
        frames_with_no_detection = len(df[df["Objects_Detected"] == 0])
        
        # Tính độ tin cậy trung bình của TOÀN BỘ các box đã detect
        overall_avg_conf = np.mean(all_confidences) if len(all_confidences) > 0 else 0.0

        # Lưu file CSV chi tiết
        df.to_csv(raw_csv_path, index=False)

        # Tạo nội dung báo cáo
        report_content = f"""
==========================================================
BÁO CÁO HIỆU NĂNG MODEL: {model_name}
Ngày test: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video input: {os.path.basename(video_path)}
==========================================================

1. HIỆU SUẤT TỐC ĐỘ (SPEED)
   - FPS Trung bình (Avg FPS): {avg_fps:.2f}
   - FPS Thấp nhất (Min FPS):  {min_fps:.2f} (Drop frame)
   - FPS Cao nhất (Max FPS):   {max_fps:.2f}
   - Thời gian xử lý trung bình mỗi frame: {avg_time:.2f} ms

2. KHẢ NĂNG PHÁT HIỆN (DETECTION CAPABILITY)
   - Tổng số frame đã test: {len(df)}
   - Tổng số tàu/bè phát hiện được: {total_objects_detected}
   - Số lượng tàu trung bình/frame: {avg_objects_per_frame:.2f}
   - Độ tin cậy trung bình (Avg Confidence): {overall_avg_conf:.4f} (Max 1.0)
     -> Chỉ số này càng cao nghĩa là model càng "tự tin" với dự đoán của mình.
   - Số frame không phát hiện được gì: {frames_with_no_detection} ({frames_with_no_detection/len(df)*100:.1f}%)

3. FILE KẾT QUẢ
   - Video output: {os.path.basename(output_video_path)}
   - Data chi tiết: {os.path.basename(raw_csv_path)}
==========================================================
        """

        # Ghi file báo cáo .txt
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(report_content)
        print(f"\n✅ Đã lưu báo cáo tại: {report_path}")

    else:
        print("Không có dữ liệu để báo cáo (Video quá ngắn hoặc lỗi).")

if __name__ == "__main__":
    print("Vui lòng chọn file model .pt (YOLO)")
    root = tk.Tk()
    root.withdraw()
    selected_model = filedialog.askopenfilename(title="Chọn Model YOLO (.pt)", filetypes=[("Model files", "*.pt")])
    
    if selected_model:
        run_yolo_test_v2(selected_model)
    else:
        print("Chưa chọn model!")