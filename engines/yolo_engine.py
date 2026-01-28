import cv2
import time
import os
import threading
import queue
from ultralytics import YOLO
from engines.ocr_engine import ShipOCR
from utils.report_utils import save_test_report

class YoloTester:
    def __init__(self, model_path, input_source, output_folder,
                 conf, imgsz, stride, use_ocr=False):

        self.model_path = model_path
        self.input_source = input_source
        self.output_folder = output_folder
        self.conf = conf
        self.imgsz = imgsz
        self.stride = stride
        self.use_ocr = use_ocr

        self.stop_event = False

        print(f">> Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.ocr_engine = None
        self.ocr_queue = queue.Queue()
        
        if use_ocr:
            try:
                self.ocr_engine = ShipOCR()
                self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
                self.ocr_thread.start()
            except Exception as e:
                print(f"Không thể khởi tạo OCR: {e}")
                self.use_ocr = False

        self.ocr_cache = {}

        self.current_objects = {}

    def ocr_worker(self):
        """
        Luồng riêng để xử lý OCR nặng nhọc
        """
        print(">> Luồng OCR đã khởi động và đang chờ ảnh...")
        while not self.stop_event:
            try:

                req = self.ocr_queue.get(timeout=1)
                track_id, crop_img = req
                

                res = self.ocr_engine.ocr_image(crop_img)
                
                if res:
                    text = res[0]["text"]

                    print(f">> OCR Success [ID {track_id}]: {text}")


                    if track_id in self.ocr_cache:
                        self.ocr_cache[track_id]["texts"].append(text)
                        
                        texts = self.ocr_cache[track_id]["texts"]
                        if len(texts) >= 1:
                            most_common = max(set(texts), key=texts.count)
                            self.ocr_cache[track_id]["final"] = most_common
                else:

                    pass
                
                self.ocr_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong luồng OCR: {e}")

    def run(self, update_gui_callback):
        cap = cv2.VideoCapture(self.input_source)
        if not cap.isOpened():
            print("Không thể mở video!")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        all_confs = []
        data_report = []

        print(">> Bắt đầu xử lý video...")

        while cap.isOpened() and not self.stop_event:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.stride != 0:
                continue

            start_time = time.time()


            results = self.model.track(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                persist=True,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            current_ids = []

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                all_confs.extend(confs)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    current_ids.append(track_id)

                    if track_id not in self.ocr_cache:
                        self.ocr_cache[track_id] = {
                            "texts": [],
                            "final": None
                        }

                    ocr_text = self.ocr_cache[track_id]["final"]


                    if self.use_ocr and self.ocr_engine:

                        if ocr_text is None and frame_count % 5 == 0:
                            if self.ocr_queue.qsize() < 10: 
                                h, w, _ = frame.shape
                                cy1 = max(0, y1)
                                cy2 = min(h, y2)
                                cx1 = max(0, x1)
                                cx2 = min(w, x2)
                                
                                crop_img = frame[cy1:cy2, cx1:cx2].copy()
                                self.ocr_queue.put((track_id, crop_img))

                    show_text = ocr_text if ocr_text else "..."

                    self.current_objects[track_id] = {
                        "bbox": (x1, y1, x2, y2),
                        "ocr": show_text,
                        "crop": frame[y1:y2, x1:x2].copy()
                    }

                    if ocr_text:
                        cv2.putText(
                            annotated_frame,
                            f"{ocr_text}",
                            (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2
                        )

            end_time = time.time()
            process_time_ms = (end_time - start_time) * 1000
            fps_cur = 1000.0 / process_time_ms if process_time_ms > 0 else 0

            update_gui_callback(annotated_frame, fps_cur)

            data_report.append({
                "Frame": frame_count,
                "FPS": fps_cur,
                "Time_ms": process_time_ms,
                "Objects_In_Frame": len(current_ids),
                "Total_Unique_Objects": len(self.ocr_cache)
            })

        cap.release()


        if data_report:
            try:
                save_test_report(
                    data=data_report,
                    all_confs=all_confs,
                    output_folder=self.output_folder,
                    video_name=os.path.basename(self.input_source),
                    processed_count=frame_count,
                    total_frames=total_frames,
                    model_name=os.path.basename(self.model_path),
                    imgsz=self.imgsz,
                    stride=self.stride,
                    conf_thresh=self.conf,
                    tag="integrate_test",
                    ocr_data={
                        tid: v["final"] if v["final"] else "Unknown"
                        for tid, v in self.ocr_cache.items()
                    }
                )
                print(f">> Đã lưu báo cáo tại: {self.output_folder}")
            except Exception as e:
                print(f"Lỗi khi lưu báo cáo: {e}")

    def stop(self):
        self.stop_event = True