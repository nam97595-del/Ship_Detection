import cv2
import time
import os
from ultralytics import YOLO
from tkinter import messagebox
from utils.report_utils import save_test_report  # Import hàm report từ utils

class YoloTester:
    def __init__(self, model_path, video_path, output_folder, imgsz, stride, conf, iou, stop_event):
        self.model_path = model_path
        self.video_path = video_path
        self.output_folder = output_folder
        self.imgsz = imgsz
        self.stride = stride
        self.conf = conf
        self.iou = iou
        self.stop_event = stop_event

    def run(self):
        try:
            print(f"🔹 Loading model: {self.model_path}")
            model = YOLO(self.model_path)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model:\n{e}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", f"Không mở được video:\n{self.video_path}")
            return

        # Setup Video Writer
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        tag = f"sz{self.imgsz}_skip{self.stride}"
        # ... (Tạo tên file video output - bạn có thể tùy chỉnh lại ở đây nếu cần)
        out_vid_path = os.path.join(self.output_folder, f"result_{model_name}_{tag}.mp4")
        out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (orig_w, orig_h))

        # Setup Window
        window_name = f"YOLO View - {model_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        processed_count = 0
        frame_data = []
        all_confs = []

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success: break
            
            frame_idx += 1
            if frame_idx % self.stride != 0: continue # Skip frame

            processed_count += 1
            start_t = time.time()
            
            # --- PREDICT ---
            results = model(frame, verbose=False, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            
            end_t = time.time()
            fps_curr = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0
            
            # --- VẼ GIAO DIỆN LÊN ẢNH ---
            annotated_frame = results[0].plot()
            
            # (Phần vẽ Dashboard mình giữ gọn lại để ví dụ, bạn có thể copy phần vẽ đẹp từ code cũ sang đây)
            cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            # Collect Data
            confs = results[0].boxes.conf.cpu().numpy()
            if len(confs) > 0: all_confs.extend(confs)
            
            if processed_count > 1:
                frame_data.append({
                    "Frame": frame_idx, "FPS": round(fps_curr, 2),
                    "Time_ms": round((end_t - start_t)*1000, 2),
                    "Objects": len(confs)
                })

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_data:
            # Nhận cả path và content từ hàm save
            txt_path, report_content = save_test_report(
                frame_data, all_confs, self.output_folder, os.path.basename(self.video_path), 
                processed_count, total_frames, model_name, self.imgsz, self.stride, self.conf, tag
            )
            
            # HIỂN THỊ REPORT NGAY LẬP TỨC
            # Dùng messagebox để show nội dung report
            messagebox.showinfo("KẾT QUẢ TEST CHI TIẾT", report_content)
            
        else:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu (có thể video quá ngắn hoặc không detect được gì)!")