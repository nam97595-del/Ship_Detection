import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

from ocr_module import ShipOCR

MODEL_PATH = r"D:/TauThuyen/best.pt/yolov12x_fish-speed-pass.pt"
CONF = 0.7
IOU = 0.5


class ShipDetectOCR:
    def __init__(self, root):
        self.root = root
        root.title("Ship Detect + OCR")
        root.geometry("1500x850")

        self.model = YOLO(MODEL_PATH).to("cuda")
        self.ocr_reader = ShipOCR(lang="en")

        self.last_boxes = []
        self.last_frame = None
        self.scale = 1.0

        self.stop_video = False
        self.video_thread = None

        self.build_ui()

    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas_left = tk.Canvas(main, bg="black")
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_left.bind("<Button-1>", self.on_click)

        right = tk.Frame(main, width=450)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(right, text="📦 BOX ĐƯỢC CHỌN", font=("Arial", 12, "bold")).pack()
        self.canvas_box = tk.Canvas(right, width=420, height=260, bg="gray")
        self.canvas_box.pack(padx=5, pady=5)

        tk.Label(right, text="OCR RESULT", font=("Arial", 12, "bold")).pack()
        self.txt = tk.Text(right, height=20, font=("Consolas", 13))
        self.txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X)

        tk.Button(bottom, text="Ảnh", width=15, command=self.open_image)\
            .pack(side=tk.LEFT, padx=5)

        tk.Button(bottom, text="Video", width=15, command=self.open_video)\
            .pack(side=tk.LEFT, padx=5)

    def open_image(self):
        self.stop_video = True
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1)

        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return

        frame = cv2.imread(path)
        self.process_frame(frame)

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mkv")]
        )
        if not path:
            return

        self.stop_video = True
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1)

        self.stop_video = False
        self.video_thread = threading.Thread(
            target=self.play_video,
            args=(path,),
            daemon=True
        )
        self.video_thread.start()

    def play_video(self, path):
        cap = cv2.VideoCapture(path)

        while cap.isOpened() and not self.stop_video:
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(frame)
            time.sleep(0.03)

        cap.release()

    def process_frame(self, frame):
        result = self.model.predict(
            frame, conf=CONF, iou=IOU, verbose=False
        )[0]

        self.last_boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes else []
        )
        self.last_frame = frame.copy()

        frame_draw = result.plot()
        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        cw = self.canvas_left.winfo_width()
        ch = self.canvas_left.winfo_height()
        iw, ih = pil.size

        self.scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * self.scale), int(ih * self.scale)

        pil = pil.resize((nw, nh), Image.LANCZOS)
        self.tk_left = ImageTk.PhotoImage(pil)

        self.canvas_left.delete("all")
        self.canvas_left.create_image(
            (cw - nw) // 2,
            (ch - nh) // 2,
            anchor=tk.NW,
            image=self.tk_left
        )

    def on_click(self, event):
        if self.last_frame is None:
            return

        ox = (self.canvas_left.winfo_width() - self.tk_left.width()) // 2
        oy = (self.canvas_left.winfo_height() - self.tk_left.height()) // 2

        x = int((event.x - ox) / self.scale)
        y = int((event.y - oy) / self.scale)

        for box in self.last_boxes:
            x1, y1, x2, y2 = map(int, box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                crop = self.last_frame[y1:y2, x1:x2]
                self.show_crop_and_ocr(crop)
                break

    def show_crop_and_ocr(self, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((420, 260))
        self.tk_crop = ImageTk.PhotoImage(pil)

        self.canvas_box.delete("all")
        self.canvas_box.create_image(
            0, 0, anchor=tk.NW, image=self.tk_crop
        )
        

        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Đang OCR vùng đã chọn...\n\n")
        self.txt.update()

        results = self.ocr_reader.ocr_image(crop)

        if not results:
            self.txt.insert(
                tk.END,
                "Không phát hiện chữ trong box này.\n"
            )
            return

        self.txt.insert(
            tk.END,
            f"Phát hiện {len(results)} vùng chữ:\n\n"
        )

        for i, r in enumerate(results, 1):
            text = r["text"]
            score = r["score"]
            self.txt.insert(
                tk.END,
                f"{i}. {text} ({int(score * 100)}%)\n"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = ShipDetectOCR(root)
    root.mainloop()
