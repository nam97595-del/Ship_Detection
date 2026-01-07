import cv2
import os
import time
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# ==============================================================================
# CẤU HÌNH NGƯỜI DÙNG (CHỈNH SỬA TẠI ĐÂY)
# ==============================================================================
TEST_IMGSZ = 1280      # Kích thước ảnh đầu vào (Convert size) -> Thường là 640
FRAME_STRIDE = 3      # Drop frame: Xử lý 1 frame, bỏ qua (FRAME_STRIDE - 1) frame. 
                      # = 1: Xử lý tất cả (Không drop)
                      # = 3: Video 30fps sẽ chỉ xử lý như 10fps (Giúp test nhanh hơn)
# ==============================================================================

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

def run_yolo_test_v3(model_path):
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
    
    # Tên file output ghi rõ thông số để dễ so sánh
    tag = f"sz{TEST_IMGSZ}_skip{FRAME_STRIDE}"
    output_video_name = f"{timestamp}_{model_name}_{tag}_video.mp4"
    raw_csv_name = f"{timestamp}_{model_name}_{tag}_raw.csv"
    report_name = f"{timestamp}_{model_name}_{tag}_REPORT.txt"
    
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
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính toán FPS cho video đầu ra (Vì drop frame nên video output sẽ ngắn lại hoặc tua nhanh)
    # Ở đây ta giữ nguyên FPS gốc để video output trông sẽ bị "tua nhanh" (timelapse)
    # nhưng phản ánh đúng các frame đã được detect.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (orig_width, orig_height))

    # 5. Cài đặt hiển thị
    window_name = f"Test Result (Skip {FRAME_STRIDE}) - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720)

    frame_data = []
    all_confidences = []
    processed_count = 0

    print(f"\n🚀 Bắt đầu test: Imgsz={TEST_IMGSZ} | Drop Frame={FRAME_STRIDE} (Xử lý 1/{FRAME_STRIDE} frames)")
    print("Nhấn 'q' để dừng sớm.")

    frame_idx = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_idx += 1

        # --- LOGIC DROP FRAME ---
        # Nếu frame hiện tại không chia hết cho bước nhảy thì bỏ qua
        if frame_idx % FRAME_STRIDE != 0:
            continue

        # Bắt đầu tính giờ cho frame được chọn
        processed_count += 1
        start_time = time.time()

        # --- INFERENCE (Convert size tại đây) ---
        # imgsz=TEST_IMGSZ: Resize ảnh về 640x640 (hoặc size khác) trước khi detect
        results = model(frame, verbose=False, conf=0.7, iou=0.5, imgsz=TEST_IMGSZ)
        
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000
        # FPS tức thời (Instant FPS)
        current_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # --- XỬ LÝ KẾT QUẢ ---
        boxes = results[0].boxes
        num_objects = len(boxes)
        
        confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        if len(confs) > 0:
            avg_conf_frame = float(np.mean(confs))
            all_confidences.extend(confs)
        else:
            avg_conf_frame = 0.0

        # Vẽ và hiển thị
        annotated_frame = results[0].plot()

        # Hiển thị thông tin lên hình
        cv2.rectangle(annotated_frame, (0, 0), (450, 140), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Input Size: {TEST_IMGSZ}x{TEST_IMGSZ}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Process FPS: {current_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Skipping: {FRAME_STRIDE}x (Frame {frame_idx})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)

        # Lưu dữ liệu
        if processed_count > 1: # Bỏ qua frame đầu cho ổn định
            frame_data.append({
                "Original_Frame_Index": frame_idx,
                "FPS_Instant": round(current_fps, 2),
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

    # 7. BÁO CÁO
    if frame_data:
        df = pd.DataFrame(frame_data)
        
        avg_fps = df["FPS_Instant"].mean()
        avg_time = df["Inference_Time_ms"].mean()
        total_objects = df["Objects_Detected"].sum()
        overall_conf = np.mean(all_confidences) if len(all_confidences) > 0 else 0.0
        
        df.to_csv(raw_csv_path, index=False)

        report_content = f"""
==========================================================
BÁO CÁO TEST: {model_name}
==========================================================
1. CẤU HÌNH TEST
   - Video Input: {os.path.basename(video_path)}
   - Image Size (Resize): {TEST_IMGSZ}x{TEST_IMGSZ}
   - Frame Stride (Drop Frame): {FRAME_STRIDE} (Xử lý 1 frame mỗi {FRAME_STRIDE} frame)
   - Tổng số frame gốc: {total_frames_input}
   - Số frame thực tế đã xử lý: {processed_count}

2. KẾT QUẢ HIỆU NĂNG
   - Tốc độ xử lý trung bình (FPS Process): {avg_fps:.2f}
   - Thời gian detect trung bình: {avg_time:.2f} ms/frame
   - Tổng object phát hiện: {total_objects}
   - Độ tin cậy trung bình (Confidence): {overall_conf:.4f}

3. NOTE
   - Video output sẽ tua nhanh gấp {FRAME_STRIDE} lần so với thực tế vì đã bỏ bớt frame.
==========================================================
        """

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(report_content)
        print(f"\n✅ Đã xong. File lưu tại: {output_folder}")
    else:
        print("Không có dữ liệu.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Chọn Model YOLO (.pt)...")
    selected_model = filedialog.askopenfilename( title="Chọn Model YOLO (.pt / .engine)", filetypes=[("All Model Files", "*.pt;*.engine;*.onnx")])
    
    if selected_model:
        run_yolo_test_v3(selected_model)
    else:
        print("Chưa chọn model!")