import cv2
import time
from ultralytics import YOLO
from engines.ocr_engine import ShipOCR


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

        print(f">> Loading ship detector: {model_path}")
        self.model = YOLO(model_path)

        print(">> Loading text detector: best (8).pt")
        self.text_model = YOLO("best (8).pt")

        if use_ocr:
            print(">> Loading OCR engine")
            self.ocr_engine = ShipOCR()
        else:
            self.ocr_engine = None

        self.current_objects = {}

    #OCR khi click
    def request_ocr(self, track_id):

        if not self.ocr_engine:
            return

        if track_id not in self.current_objects:
            return

        crop = self.current_objects[track_id]["crop"]

        if crop is None:
            return

        h = crop.shape[0]

        #Detect Text
        results = self.text_model(crop, conf=0.5, verbose=False)

        text_crop = None

        for r in results:

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes
            conf = boxes.conf.cpu().numpy()
            idx = conf.argmax()

            box = boxes.xyxy[idx].cpu().numpy().astype(int)

            x1,x2,y1,y2=box
            pad = 5

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(crop.shape[1], x2 + pad)
            y2 = min(crop.shape[0], y2 + pad)

            text_crop = crop[y1:y2, x1:x2]

            break

        if text_crop is None:

            print("Không detect được vùng chữ")

            self.current_objects[track_id]["text_crop"] = None
            return
        cv2.imshow(f"text_crop_{track_id}", text_crop)
        cv2.waitKey(1)

        #Lưu vùng chữ để GUI hiển thị debug
        self.current_objects[track_id]["text_crop"] = text_crop
        res = self.ocr_engine.ocr_image(text_crop)

        if res:

            text = " ".join(r.get("text", "") for r in res)

            print(f"OCR RESULT [ID {track_id}]: {text}")

            self.current_objects[track_id]["ocr"] = text

        else:

            print("OCR không đọc được")

    #Vòng lặp chính
    def run(self, update_gui_callback):

        cap = cv2.VideoCapture(self.input_source)

        if not cap.isOpened():
            print("Không mở được video")
            return

        frame_count = 0

        print(">> Start processing video")

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

            if result.boxes and result.boxes.id is not None:

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):

                    x1, y1, x2, y2 = box

                    if track_id not in self.current_objects:

                        self.current_objects[track_id] = {
                            "bbox": (x1, y1, x2, y2),
                            "ocr": None,
                            "crop": None,
                            "text_crop": None
                        }

                    crop = frame[y1:y2, x1:x2]

                    self.current_objects[track_id]["bbox"] = (x1, y1, x2, y2)
                    self.current_objects[track_id]["crop"] = crop

                    ocr_text = self.current_objects[track_id]["ocr"]

                    if ocr_text:

                        cv2.putText(
                            annotated_frame,
                            ocr_text,
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

            end_time = time.time()

            process_time = (end_time - start_time) * 1000

            fps = 1000.0 / process_time if process_time > 0 else 0

            update_gui_callback(annotated_frame, fps)

        cap.release()

    def stop(self):
        self.stop_event = True

