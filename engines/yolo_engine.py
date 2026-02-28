import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from tkinter import messagebox
from utils.report_utils import save_test_report

timestamp = time.strftime("%Y%m%d_%H%M%S")
class YoloTester:
    def __init__(self, model_path, video_path, output_folder, imgsz, stride, conf, iou, stop_event, tracker_type="bytetrack.yaml"):
        self.model_path = model_path
        self.video_path = video_path
        self.output_folder = output_folder
        self.imgsz = imgsz
        self.stride = stride
        self.conf = conf
        self.iou = iou
        self.stop_event = stop_event
        self.tracker_type = tracker_type

    def run(self):
        # --- CẤU HÌNH GIAO DIỆN ---
        DRAW_CFG = {
            "box_thick": 2,          # Độ dày khung hình chữ nhật
            "font_scale": 0.6,       # Cỡ chữ
            "font_thick": 1,         # Độ đậm của chữ
            "text_bg_alpha": 0.5,    # Độ trong suốt nền chữ (0-1)
            "show_conf": True,       # Hiện độ tin cậy
            
            # Mapping ID class sang tên hiển thị và màu sắc (BGR)
            "classes": {
                0: {"name": "F-Boat", "color": (0, 0, 255)},    # Đỏ (Fishing)
                1: {"name": "P-Ship", "color": (255, 255, 0)},  # Xanh ngọc (Passenger)
                2: {"name": "S-Boat", "color": (0, 255, 0)},    # Xanh lá (Speed)
                # Thêm mặc định nếu có class lạ
                "default": {"name": "Obj", "color": (255, 255, 255)} 
            }
        }

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
        tracker_tag = self.tracker_type.replace('.yaml', '').upper()
        tag = f"sz{self.imgsz}_skip{self.stride}_{tracker_tag}_CUSTOM"
        
        out_vid_name = f"{model_name}_vs_{input_name}_{tag}.mp4"
        out_vid_path = os.path.join(self.output_folder, out_vid_name)
        out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (orig_w, orig_h))

        # Setup Window
        window_name = f"YOLO Custom Draw - {model_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0 # KHỞI TẠO TỪ 0 ĐỂ KHỚP VỚI DARKLABEL
        processed_count = 0
        frame_data = []
        all_confs = []
        unique_ids = set()

        # Mảng lưu kết quả MOT
        mot_predictions = []

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success: break
            
            # --- SỬA LỖI ĐẾM FRAME ĐỂ KHÔNG BỊ LỆCH PHA VỚI GT ---
            if (frame_idx + 1) % self.stride != 0: 
                frame_idx += 1
                continue 

            processed_count += 1
            start_t = time.time()
            
            # Tracking
            results = model.track(frame, persist=True, verbose=False, conf=self.conf, iou=self.iou, imgsz=self.imgsz, tracker=self.tracker_type)
            
            end_t = time.time()
            fps_curr = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0
            
            # --- LOGIC VẼ LABEL ---
            annotated_frame = frame.copy() # Copy ảnh gốc để vẽ
            overlay = frame.copy()         # Layer để vẽ độ trong suốt (transparency)
            
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # Lấy dữ liệu dạng array để xử lý nhanh hơn
                xyxys = boxes.xyxy.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1]*len(boxes)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                
                # Lưu ID để thống kê
                for t_id in ids:
                    if t_id != -1: unique_ids.add(t_id)
                    
                # Vòng lặp vẽ từng box
                for box, track_id, cls_id, conf in zip(xyxys, ids, clss, confs):
                    x1, y1, x2, y2 = box

                    # Lưu mảng MOT
                    w = x2 - x1
                    h = y2 - y1

                    # Thông tin tracking (10 cột)
                    if track_id != -1:
                        mot_line = f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf:.4f},{cls_id},-1,-1"
                        mot_predictions.append(mot_line)
                        
                    # 1. Lấy thông tin class từ Config
                    class_info = DRAW_CFG["classes"].get(cls_id, DRAW_CFG["classes"]["default"])
                    color = class_info["color"]
                    label_name = class_info["name"]
                    
                    # 2. Vẽ Box (Khung hình chữ nhật)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, DRAW_CFG["box_thick"])
                    
                    # 3. Tạo Label text
                    id_text = f"#{track_id}" if track_id != -1 else ""
                    conf_text = f"{conf:.2f}" if DRAW_CFG["show_conf"] else ""
                    label = f"{id_text} {label_name} {conf_text}".strip()
                    
                    # 4. Tính toán kích thước nền chữ
                    (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, DRAW_CFG["font_scale"], DRAW_CFG["font_thick"])
                    
                    # Đặt vị trí chữ
                    text_y = y1 - 5 if y1 - h_text - 5 > 0 else y1 + h_text + 5
                    
                    # 5. Vẽ nền chữ TRONG SUỐT
                    cv2.rectangle(overlay, 
                                  (x1, text_y - h_text - 5), 
                                  (x1 + w_text, text_y + baseline), 
                                  color, -1) 
                    
                    # Vẽ lên frame chính
                    cv2.putText(annotated_frame, label, (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, DRAW_CFG["font_scale"], (255, 255, 255), DRAW_CFG["font_thick"], cv2.LINE_AA)

            # --- GỘP LAYER TRONG SUỐT ---
            alpha = 1 - DRAW_CFG["text_bg_alpha"]
            annotated_frame = cv2.addWeighted(overlay, 1 - alpha, annotated_frame, alpha, 0)

            # --- VẼ THỐNG KÊ (UI) ---
            cv2.rectangle(annotated_frame, (5, 5), (250, 85), (0, 0, 0), -1) 
            cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Objs Current: {len(boxes)}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"Total Count: {len(unique_ids)}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            
            out.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            # Collect Data
            if len(boxes) > 0: all_confs.extend(confs)
            if processed_count > 1:
                frame_data.append({
                    "Frame": frame_idx, "FPS": round(fps_curr, 2),
                    "Time_ms": round((end_t - start_t)*1000, 2),
                    "Objects_In_Frame": len(boxes),
                    "Total_Unique_Objects": len(unique_ids)
                })

            # TĂNG FRAME_IDX Ở CUỐI VÒNG LẶP
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Lưu file pred.txt
        pred_file = f"pred_{input_name}_{timestamp}.txt"
        pred_path = os.path.join(self.output_folder, pred_file)
        
        with open(pred_path, 'w') as f:
            f.write("\n".join(mot_predictions))
        print(f"✅ Đã lưu kết quả dự đoán (MOT format) tại: {pred_path}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_data:
            txt_path, report_content = save_test_report(
                frame_data, all_confs, self.output_folder, os.path.basename(self.video_path), 
                processed_count, total_frames, model_name, self.imgsz, self.stride, self.conf, tag
            )
            report_content += f"\n\n[TRACKING RESULT]\nTổng số đối tượng độc nhất: {len(unique_ids)}"
            messagebox.showinfo("KẾT QUẢ TEST", report_content)
        else:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu!")
