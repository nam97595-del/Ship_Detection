import cv2
import time
import os
from ultralytics import YOLO
from tkinter import messagebox
from utils.report_utils import save_test_report

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

        input_name = os.path.splitext(os.path.basename(self.video_path))[0]
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        tag = f"sz{self.imgsz}_skip{self.stride}_TRACK" # <--- Thêm chữ TRACK vào tên file
        
        out_vid_name = f"{model_name}_vs_{input_name}_{tag}.mp4"
        out_vid_path = os.path.join(self.output_folder, out_vid_name)
        
        # Output video writer
        out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (orig_w, orig_h))

        # Setup Window
        window_name = f"YOLO Tracking - {model_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        processed_count = 0
        frame_data = []
        all_confs = []
        
        # Set lưu trữ các ID đã xuất hiện để đếm số lượng thực tế
        unique_ids = set() 

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success: break
            
            frame_idx += 1
            if frame_idx % self.stride != 0: continue 

            processed_count += 1
            start_t = time.time()
            
            # persist=True: Bắt buộc phải có để model nhớ ID qua các frame
            # tracker="bytetrack.yaml" hoặc "botsort.yaml" (mặc định là botsort)
            results = model.track(frame, persist=True, verbose=False, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
            
            end_t = time.time()
            fps_curr = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0
            
            # --- XỬ LÝ KẾT QUẢ TRACKING ---
            boxes = results[0].boxes
            
            # Lấy danh sách ID (nếu có detect được vật thể)
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist() # Lấy danh sách ID hiện tại
                for t_id in track_ids:
                    unique_ids.add(t_id) # Thêm vào set để đếm số lượng unique
            
            # --- VẼ GIAO DIỆN ---
            # Hàm plot() của Ultralytics tự động vẽ ID lên hình khi dùng mode track
            annotated_frame = results[0].plot()
            
            # Vẽ thêm thống kê Tracking lên góc trái
            total_unique_objs = len(unique_ids)
            
            # Tạo nền đen mờ để chữ dễ đọc
            cv2.rectangle(annotated_frame, (0, 0), (350, 100), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects in Frame: {len(boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Hiển thị tổng số đối tượng thực tế đã đi qua (Tracking Count)
            cv2.putText(annotated_frame, f"Total Count (Unique): {total_unique_objs}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            out.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            # Collect Data
            confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
            if len(confs) > 0: all_confs.extend(confs)
            
            if processed_count > 1:
                frame_data.append({
                    "Frame": frame_idx, "FPS": round(fps_curr, 2),
                    "Time_ms": round((end_t - start_t)*1000, 2),
                    "Objects_In_Frame": len(boxes),
                    "Total_Unique_Objects": total_unique_objs # <--- Ghi thêm vào báo cáo
                })

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_data:
            txt_path, report_content = save_test_report(
                frame_data, all_confs, self.output_folder, os.path.basename(self.video_path), 
                processed_count, total_frames, model_name, self.imgsz, self.stride, self.conf, tag
            )
            
            # Thêm dòng tổng kết tracking vào thông báo
            report_content += f"\n\n[TRACKING RESULT]\nTổng số đối tượng độc nhất phát hiện được: {len(unique_ids)}"
            
            messagebox.showinfo("KẾT QUẢ TEST", report_content)
        else:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu!")