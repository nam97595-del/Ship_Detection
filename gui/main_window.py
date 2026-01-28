import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os

from engines.yolo_engine import YoloTester
from utils.report_utils import save_test_report

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Giám sát Tàu biển - YOLOv12 & OCR")
        self.root.geometry("1400x850")
        self.output_dir = tk.StringVar()
        # Cấu hình
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.conf_val = tk.DoubleVar(value=0.5)
        self.stride_val = tk.IntVar(value=2)
        self.use_ocr_var = tk.BooleanVar(value=False) 

        self.setup_ui()
        
    def setup_ui(self):
        
        control_frame = tk.Frame(self.root, bg="#f0f0f0", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(control_frame, text="CẤU HÌNH", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Nút chọn Model
        tk.Button(control_frame, text="Chọn Model (.pt/.engine)", command=self.choose_model).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.model_path, fg="blue", wraplength=280).pack()
        
        # Nút chọn Video
        tk.Button(control_frame, text="Chọn Video Input", command=self.choose_video).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.video_path, fg="blue", wraplength=280).pack()
        
        # Slider Confidence
        tk.Label(control_frame, text="Độ tin cậy (Conf)").pack(pady=(20,0))
        tk.Scale(control_frame, variable=self.conf_val, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Slider Stride
        tk.Label(control_frame, text="Nhảy Frame (Stride)").pack(pady=(10,0))
        tk.Scale(control_frame, variable=self.stride_val, from_=1, to=10, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # CHECKBOX OCR
        tk.Checkbutton(control_frame, text="Bật Nhận diện Mã hiệu (OCR)", 
                       variable=self.use_ocr_var, font=("Arial", 11, "bold"), fg="red").pack(pady=20)
        
        # Nút Chạy
        tk.Button(control_frame, text="▶ BẮT ĐẦU CHẠY", bg="green", fg="white", font=("Arial", 12, "bold"), 
                  height=2, command=self.start_process).pack(fill=tk.X, pady=20)
                  
        # Nút Dừng
        tk.Button(control_frame, text="⏹ DỪNG", bg="red", fg="white", command=self.stop_process).pack(fill=tk.X)

        # Nút lưu KQ
        tk.Button(control_frame, text="Chọn thư mục lưu kết quả", command=self.choose_output_folder).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.output_dir, fg="green", wraplength=280).pack()
        # Panel hiển thị Video (Canvas)
        self.canvas_video = tk.Canvas(self.root, bg="black")
        self.canvas_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_video.bind("<Button-1>", self.on_canvas_click)

        self.detail_frame = tk.Frame(self.root, width=300, bg="#ffffff")
        self.detail_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.detail_frame, text="CHI TIẾT TÀU", font=("Arial", 12, "bold")).pack(pady=10)

        self.detail_canvas = tk.Canvas(self.detail_frame, width=280, height=200, bg="gray")
        self.detail_canvas.pack(pady=10)

        self.detail_text = tk.Label(self.detail_frame, text="", wraplength=280, justify=tk.LEFT)
        self.detail_text.pack()

    def choose_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt *.engine")])
        if path: self.model_path.set(path)

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")])
        if path: self.video_path.set(path)

    def start_process(self):
        if not self.model_path.get() or not self.video_path.get():
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn Model và Video trước!")
            return

        if not self.output_dir.get():
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn thư mục lưu kết quả!")
            return
        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        self.engine = YoloTester(
            model_path=self.model_path.get(),
            input_source=self.video_path.get(),
            output_folder=output_dir,
            conf=self.conf_val.get(),
            imgsz=640,
            stride=self.stride_val.get(),
            use_ocr=self.use_ocr_var.get()  # Truyền giá trị bật/tắt OCR
        )
        
        self.thread = threading.Thread(target=self.engine.run, args=(self.update_frame,))
        self.thread.start()

    def stop_process(self):
        if hasattr(self, 'engine'):
            self.engine.stop()

    def update_frame(self, frame_cv, fps):
        h, w = frame_cv.shape[:2]
        canvas_h = self.canvas_video.winfo_height()
        canvas_w = self.canvas_video.winfo_width()

        if canvas_w > 0 and canvas_h > 0:
            scale = min(canvas_w / w, canvas_h / h)

            new_w, new_h = int(w * scale), int(h * scale)

            self.last_scale = scale
            self.last_offset = (
                (canvas_w - new_w) // 2,
                (canvas_h - new_h) // 2
            )

            frame_resized = cv2.resize(frame_cv, (new_w, new_h))

            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.canvas_video.create_image(
                canvas_w // 2,
                canvas_h // 2,
                anchor=tk.CENTER,
                image=img_tk
            )
            self.canvas_video.image = img_tk

        self.root.title(f"Hệ thống Giám sát - FPS: {fps:.2f}")
    def on_closing(self):
        if messagebox.askokcancel("Thoát", "Bạn có muốn dừng chương trình và thoát?"):
            self.stop_process()
            self.root.destroy()
    def choose_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def on_canvas_click(self, event):
        if not hasattr(self, 'engine'):
            return

        if not hasattr(self, 'last_scale') or not hasattr(self, 'last_offset'):
            return

        x = (event.x - self.last_offset[0]) / self.last_scale
        y = (event.y - self.last_offset[1]) / self.last_scale

        for tid, obj in self.engine.current_objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.show_ship_detail(tid, obj)
                break
    
    def show_ship_detail(self, track_id, obj):
        if obj["crop"] is not None:
            img = cv2.cvtColor(obj["crop"], cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (280, 200))
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.detail_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.detail_canvas.image = img_tk

        text = obj["ocr"] if obj["ocr"] else "Chưa nhận diện"
        self.detail_text.config(
            text=f"Track ID: {track_id}\nOCR: {text}"
        )
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()