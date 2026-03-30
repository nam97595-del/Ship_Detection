import cv2
import time
import os
import numpy as np
from tkinter import messagebox
from utils.report_utils import save_test_report
from detectors.detector_factory import create_detector
from trackers.tracker_factory import create_tracker

class GenericEngine:
    def __init__(self, model_type, model_path, tracker_type, video_path, output_folder, imgsz, stride, conf, stop_event):
        self.model_type = model_type  # Ví dụ: "YOLO"
        self.model_path = model_path
        self.tracker_type = tracker_type # Ví dụ: "ByteTrack"
        self.video_path = video_path
        self.output_folder = output_folder
        self.imgsz = imgsz
        self.stride = stride
        self.conf = conf
        self.stop_event = stop_event

    def run(self):
        DRAW_CFG = {
            "box_thick": 2, "font_scale": 0.6, "font_thick": 1, "text_bg_alpha": 0.5, "show_conf": True,
            "classes": {
                0: {"name": "F-Boat", "color": (0, 0, 255)},
                1: {"name": "P-Ship", "color": (255, 255, 0)},
                2: {"name": "S-Boat", "color": (0, 255, 0)},
                "default": {"name": "Obj", "color": (255, 255, 255)}
            }
        }

        try:
            # --- KIẾN TRÚC MODULAR: GỌI FACTORY ---
            detector = create_detector(self.model_type, self.model_path, self.conf, self.imgsz)
            tracker = create_tracker(self.tracker_type)
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Tạo", str(e))
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        input_name = os.path.splitext(os.path.basename(self.video_path))[0]
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        out_vid_path = os.path.join(self.output_folder, f"{model_name}_vs_{input_name}_{self.tracker_type}.mp4")
        out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (orig_w, orig_h))

        window_name = f"Testing: {self.model_type} + {self.tracker_type}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        processed_count = 0
        frame_data = []
        all_confs = []
        unique_ids = set()
        mot_predictions = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success: break
            
            if (frame_idx + 1) % self.stride != 0: 
                frame_idx += 1
                continue 

            processed_count += 1
            start_t = time.time()
            
            # ==========================================
            # LUỒNG DỮ LIỆU ĐÃ ĐƯỢC TÁCH RỜI (DECOUPLED)
            # Bước 1: Detector tìm Box
            boxes_tho = detector.detect(frame)
            
            # Bước 2: Tracker gán ID
            tracks = tracker.update(boxes_tho, frame)
            # ==========================================

            end_t = time.time()
            fps_curr = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0
            
            annotated_frame = frame.copy()
            overlay = frame.copy()
            
            # Vẽ Box dựa trên kết quả của Tracker
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
                x1, y1, x2, y2, track_id, cls_id = int(x1), int(y1), int(x2), int(y2), int(track_id), int(cls_id)
                
                unique_ids.add(track_id)
                all_confs.append(conf)

                # Ghi dữ liệu MOTA
                w, h = x2 - x1, y2 - y1
                mot_line = f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf:.4f},{cls_id},-1,-1"
                mot_predictions.append(mot_line)

                # Vẽ GUI (giữ nguyên logic cũ của bạn)
                class_info = DRAW_CFG["classes"].get(cls_id, DRAW_CFG["classes"]["default"])
                color, label_name = class_info["color"], class_info["name"]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, DRAW_CFG["box_thick"])
                label = f"#{track_id} {label_name} {conf:.2f}"
                (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, DRAW_CFG["font_scale"], DRAW_CFG["font_thick"])
                text_y = y1 - 5 if y1 - h_text - 5 > 0 else y1 + h_text + 5
                cv2.rectangle(overlay, (x1, text_y - h_text - 5), (x1 + w_text, text_y + baseline), color, -1) 
                cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, DRAW_CFG["font_scale"], (255, 255, 255), DRAW_CFG["font_thick"], cv2.LINE_AA)

            alpha = 1 - DRAW_CFG["text_bg_alpha"]
            annotated_frame = cv2.addWeighted(overlay, 1 - alpha, annotated_frame, alpha, 0)

            # Vẽ UI
            cv2.rectangle(annotated_frame, (5, 5), (250, 85), (0, 0, 0), -1) 
            cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Objs Current: {len(tracks)}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"Total Count: {len(unique_ids)}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            
            out.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            if processed_count > 1:
                frame_data.append({
                    "Frame": frame_idx, "FPS": round(fps_curr, 2), "Time_ms": round((end_t - start_t)*1000, 2),
                    "Objects_In_Frame": len(tracks), "Total_Unique_Objects": len(unique_ids)
                })

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Lưu file pred.txt
        pred_path = os.path.join(self.output_folder, f"pred_{input_name}_{timestamp}.txt")
        with open(pred_path, 'w') as f:
            f.write("\n".join(mot_predictions))
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_data:
            txt_path, report_content = save_test_report(frame_data, all_confs, self.output_folder, input_name, processed_count, total_frames, self.model_type, self.imgsz, self.stride, self.conf, self.tracker_type)
            messagebox.showinfo("KẾT QUẢ TEST", report_content)