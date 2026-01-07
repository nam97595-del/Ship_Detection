import cv2
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import RTDETR
import supervision as sv

# --- CẤU HÌNH GIAO DIỆN VẼ (ĐÃ CHỈNH SỬA CHO DỄ NHÌN) ---
BOX_THICKNESS = 3           # Độ dày khung bao vật thể
TEXT_SCALE = 1.0            # Cỡ chữ (Tăng từ 0.7 -> 1.0)
TEXT_THICKNESS = 2          # Độ đậm chữ (Tăng lên 2 cho rõ)
TEXT_PADDING = 15           # Khoảng cách đệm chữ (Tăng lên để nền chữ rộng hơn)
CONFIDENCE_THRESHOLD = 0.75  # Ngưỡng lọc

# Màu sắc cho bảng thông số (BGR)
COLOR_BG_INFO = (30, 30, 30)    # Màu nền bảng thông tin (Xám đậm)
COLOR_TEXT_LABEL = (200, 200, 200) # Màu tên thông số (Xám nhạt)
COLOR_TEXT_VALUE = (0, 255, 255)   # Màu giá trị (Vàng sáng)

def select_file():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title="Chọn Model (.pt, .onnx, .engine)", 
                                    filetypes=[("Models", "*.pt;*.onnx;*.engine")])

def select_video():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title="Chọn Video Input", filetypes=[("Video", "*.mp4;*.avi;*.mkv")])

def select_folder():
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title="Chọn Thư mục lưu kết quả")

def draw_info_panel(frame, model_name, fps, num_objects, conf):
    """Hàm vẽ bảng thông số chuyên nghiệp, dễ đọc hơn"""
    # Tạo overlay nền tối
    h, w, _ = frame.shape
    panel_w = 400
    panel_h = 140
    
    # Vẽ hình chữ nhật bo góc (hoặc chữ nhật thường) làm nền
    sub_img = frame[0:panel_h, 0:panel_w]
    black_rect = np.full(sub_img.shape, COLOR_BG_INFO, dtype=np.uint8)
    
    # Blend màu nền với video (độ trong suốt 0.7) để nhìn hiện đại hơn
    res = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 1.0)
    frame[0:panel_h, 0:panel_w] = res
    
    # Cấu hình font chữ đẹp hơn (HERSHEY_COMPLEX)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.7
    thickness = 1
    line_h = 35 # Khoảng cách các dòng
    x_start = 15
    y_start = 35

    # Dòng 1: Model Name
    cv2.putText(frame, "Model:", (x_start, y_start), font, font_scale, COLOR_TEXT_LABEL, thickness, cv2.LINE_AA)
    cv2.putText(frame, model_name[:20], (x_start + 80, y_start), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Dòng 2: FPS
    cv2.putText(frame, "FPS:", (x_start, y_start + line_h), font, font_scale, COLOR_TEXT_LABEL, thickness, cv2.LINE_AA)
    # Tô màu FPS: Xanh nếu > 20, Đỏ nếu thấp
    fps_color = (0, 255, 0) if fps > 20 else (0, 0, 255)
    cv2.putText(frame, f"{fps:.1f}", (x_start + 80, y_start + line_h), font, font_scale+0.1, fps_color, 2, cv2.LINE_AA)

    # Dòng 3: Objects
    cv2.putText(frame, "Count:", (x_start, y_start + line_h*2), font, font_scale, COLOR_TEXT_LABEL, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"{num_objects}", (x_start + 80, y_start + line_h*2), font, font_scale, COLOR_TEXT_VALUE, 2, cv2.LINE_AA)

    # Dòng 4: Avg Conf
    cv2.putText(frame, "Conf:", (x_start + 160, y_start + line_h*2), font, font_scale, COLOR_TEXT_LABEL, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"{conf:.0%}", (x_start + 230, y_start + line_h*2), font, font_scale, COLOR_TEXT_VALUE, 2, cv2.LINE_AA)

    return frame

def run_full_report_test(model_path):
    # 1. SETUP
    print(f"🔹 Đang load model: {os.path.basename(model_path)}")
    try:
        model = RTDETR(model_path)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Phần cứng: {'GPU (CUDA)' if device==0 else 'CPU'}")

    video_path = select_video()
    if not video_path: return
    output_folder = select_folder()
    if not output_folder: return

    # Tên file
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_name = f"{timestamp}_{model_name}_Result.mp4"
    out_csv_name = f"{timestamp}_{model_name}_Metrics.csv"
    out_report_name = f"{timestamp}_{model_name}_FULL_REPORT.txt"

    out_path = os.path.join(output_folder, out_video_name)
    csv_path = os.path.join(output_folder, out_csv_name)
    report_path = os.path.join(output_folder, out_report_name)

    # Video Capture
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))

    # 2. SETUP SUPERVISION (CẤU HÌNH LẠI CHO ĐẸP)
    box_annotator = sv.BoxAnnotator(
        thickness=BOX_THICKNESS,
        color=sv.ColorPalette.DEFAULT 
    )
    # LabelAnnotator mới giúp chữ nằm trong box màu nền, dễ đọc hơn
    label_annotator = sv.LabelAnnotator(
        text_scale=TEXT_SCALE,          # Chữ to hơn
        text_thickness=TEXT_THICKNESS,  # Chữ đậm hơn
        text_padding=TEXT_PADDING,      # Padding rộng hơn
        text_position=sv.Position.TOP_CENTER,
        color=sv.ColorPalette.DEFAULT
    )

    # Cửa sổ hiển thị
    window_name = f"Test Result - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- BIẾN LƯU TRỮ DỮ LIỆU BÁO CÁO ---
    frame_data = []        
    all_confidences = []   

    print(f"\n🚀 Bắt đầu test video ({total_frames_input} frames)... Nhấn 'q' để dừng.")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        frame_idx += 1

        # --- INFERENCE ---
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.45)
        
        speed_dict = results[0].speed
        inference_time_ms = speed_dict['inference'] + speed_dict['postprocess'] 
        
        end_time = time.time()
        real_proc_time = end_time - start_time
        fps = 1.0 / real_proc_time if real_proc_time > 0 else 0

        # --- XỬ LÝ DATA ---
        detections = sv.Detections.from_ultralytics(results[0])
        num_objects = len(detections)

        if num_objects > 0:
            current_confs = detections.confidence.tolist()
            all_confidences.extend(current_confs)
            avg_conf_frame = np.mean(current_confs)
        else:
            avg_conf_frame = 0.0

        # --- VẼ HÌNH (ĐÃ CẢI TIẾN) ---
        # Format label: "Person 90%" thay vì "Person 0.90" -> Dễ đọc hơn
        labels = [
            f"{model.names[class_id]} {confidence:.0%}" 
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # --- VẼ INFO PANEL MỚI ---
        annotated_frame = draw_info_panel(annotated_frame, model_name, fps, num_objects, avg_conf_frame)

        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)

        # Lưu metrics
        if frame_idx > 5:
            frame_data.append({
                "Frame": frame_idx,
                "FPS": round(fps, 2),
                "Inference_Time_ms": round(inference_time_ms, 2),
                "Objects_Detected": num_objects,
                "Avg_Confidence": round(avg_conf_frame, 4)
            })

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- 3. TẠO BÁO CÁO CHI TIẾT ---
    if frame_data:
        df = pd.DataFrame(frame_data)
        
        avg_fps = df["FPS"].mean()
        min_fps = df["FPS"].min()
        max_fps = df["FPS"].max()
        avg_latency = df["Inference_Time_ms"].mean()
        
        total_objects_detected = df["Objects_Detected"].sum()
        avg_objects_per_frame = df["Objects_Detected"].mean()
        
        no_detect_frames = len(df[df["Objects_Detected"] == 0])
        detect_frames = len(df) - no_detect_frames
        
        if len(all_confidences) > 0:
            overall_avg_conf = np.mean(all_confidences)
        else:
            overall_avg_conf = 0.0

        df.to_csv(csv_path, index=False)

        report_content = f"""
==========================================================
BÁO CÁO HIỆU NĂNG MODEL: {model_name.upper()}
Thời gian test: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video input: {os.path.basename(video_path)}
==========================================================

1. HIỆU SUẤT TỐC ĐỘ (SPEED)
   - FPS Trung bình (Avg FPS): {avg_fps:.2f}
   - FPS Thấp nhất (Min FPS):  {min_fps:.2f}
   - FPS Cao nhất (Max FPS):   {max_fps:.2f}
   - Độ trễ trung bình (Latency): {avg_latency:.2f} ms

2. KHẢ NĂNG PHÁT HIỆN (DETECTION ACCURACY)
   - Tổng số frame đã test: {len(df)}
   - Độ tin cậy trung bình: {overall_avg_conf:.2%} ({overall_avg_conf:.4f})
   
   - Tổng số vật thể phát hiện: {total_objects_detected}
   - Trung bình số vật thể/frame: {avg_objects_per_frame:.2f}
   
   - Số frame có phát hiện: {detect_frames} ({detect_frames/len(df)*100:.1f}%)
   - Số frame KHÔNG phát hiện: {no_detect_frames} ({no_detect_frames/len(df)*100:.1f}%)

3. FILE KẾT QUẢ
   - Video Output: {os.path.basename(out_path)}
   - Data chi tiết (CSV): {os.path.basename(csv_path)}
==========================================================
        """

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print("\n" + "="*40)
        print("✅ HOÀN THÀNH TEST!")
        print(f"📄 Báo cáo đầy đủ đã lưu tại: {report_path}")
        print("="*40)
        print(report_content)
        
    else:
        print("❌ Không có dữ liệu để tạo báo cáo.")

if __name__ == "__main__":
    print("Chọn model RT-DETR...")
    path = select_file()
    if path:
        run_full_report_test(path)