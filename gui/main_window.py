import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
from engines.yolo_engine import YoloTester

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Giám sát Tàu biển - YOLO & OCR")
        self.root.geometry("1400x850")
        
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.conf_val = tk.DoubleVar(value=0.5)
        self.stride_val = tk.IntVar(value=2)
        self.use_ocr_var = tk.BooleanVar(value=True) 
        
        self.selected_track_id = None 
        self.setup_ui()
        
    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#f0f0f0", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(control_frame, text="CẤU HÌNH", font=("Arial", 14, "bold")).pack(pady=10)
        
        tk.Button(control_frame, text="Chọn Model", command=self.choose_model).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.model_path, fg="blue", wraplength=280).pack()
        
        tk.Button(control_frame, text="Chọn Video", command=self.choose_video).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.video_path, fg="blue", wraplength=280).pack()
        
        tk.Button(control_frame, text="Thư mục lưu KQ", command=self.choose_output_folder).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.output_dir, fg="green", wraplength=280).pack()
        
        tk.Checkbutton(control_frame, text="Bật OCR", variable=self.use_ocr_var, fg="red", font=("Arial", 11, "bold")).pack(pady=20)
        
        tk.Button(control_frame, text="▶ BẮT ĐẦU", bg="green", fg="white", font=("Arial", 12, "bold"), height=2, command=self.start_process).pack(fill=tk.X, pady=10)
        tk.Button(control_frame, text="⏹ DỪNG", bg="red", fg="white", command=self.stop_process).pack(fill=tk.X)

        self.detail_frame = tk.Frame(self.root, width=300, bg="white")
        self.detail_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        tk.Label(self.detail_frame, text="CHI TIẾT TÀU", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.detail_canvas = tk.Canvas(self.detail_frame, width=280, height=200, bg="gray")
        self.detail_canvas.pack(pady=5)
        self.detail_text = tk.Label(self.detail_frame, text="Click vào box trên video...", font=("Arial", 11), wraplength=280, justify=tk.LEFT)
        self.detail_text.pack()

        self.canvas_video = tk.Canvas(self.root, bg="black")
        self.canvas_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_video.bind("<Button-1>", self.on_canvas_click)

    def choose_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pt *.engine")])
        if p: self.model_path.set(p)
    def choose_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if p: self.video_path.set(p)
    def choose_output_folder(self):
        p = filedialog.askdirectory()
        if p: self.output_dir.set(p)

    def start_process(self):
        if not all([self.model_path.get(), self.video_path.get(), self.output_dir.get()]):
            messagebox.showwarning("Thiếu thông tin", "Chọn đủ Model, Video và Output Folder!")
            return
        
        self.selected_track_id = None
        os.makedirs(self.output_dir.get(), exist_ok=True)
        
        self.engine = YoloTester(
            model_path=self.model_path.get(),
            input_source=self.video_path.get(),
            output_folder=self.output_dir.get(),
            conf=self.conf_val.get(), imgsz=640, stride=self.stride_val.get(),
            use_ocr=self.use_ocr_var.get()
        )
        self.thread = threading.Thread(target=self.engine.run, args=(self.update_frame,))
        self.thread.start()

    def stop_process(self):
        if hasattr(self, 'engine'): self.engine.stop()

    def update_frame(self, frame, fps):
        h, w = frame.shape[:2]
        ch = self.canvas_video.winfo_height()
        cw = self.canvas_video.winfo_width()
        
        if ch > 0 and cw > 0:
            scale = min(cw/w, ch/h)
            nw, nh = int(w*scale), int(h*scale)
            
            self.last_scale = scale
            self.last_offset = ((cw-nw)//2, (ch-nh)//2)
            
            img = cv2.resize(frame, (nw, nh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
            
            self.canvas_video.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.tk_img)
        
        self.root.title(f"FPS: {fps:.2f}")

        if self.selected_track_id is not None and hasattr(self, 'engine'):
            if self.selected_track_id in self.engine.current_objects:
                ocr_txt = self.engine.current_objects[self.selected_track_id]["ocr"]
                if ocr_txt and ocr_txt != "...":
                    self.detail_text.config(text=f"ID: {self.selected_track_id}\n\nOCR: {ocr_txt}")

    def on_canvas_click(self, event):
        if not hasattr(self, 'engine') or not hasattr(self, 'last_scale'): return
        
        x_click = (event.x - self.last_offset[0]) / self.last_scale
        y_click = (event.y - self.last_offset[1]) / self.last_scale
        
        found = False
        for tid, obj in self.engine.current_objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            if x1 <= x_click <= x2 and y1 <= y_click <= y2:
                self.selected_track_id = tid
                found = True
                
                self.show_crop(obj["crop"])
                self.detail_text.config(text=f"ID: {tid}\n\nĐang đọc OCR...")
                
                self.engine.request_manual_ocr(tid)
                break
        
        if not found:
            self.selected_track_id = None
            self.detail_text.config(text="Đã bỏ chọn.")

    def show_crop(self, img_cv):
        if img_cv is None or img_cv.size == 0: return
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (280, 200))
        self.tk_crop = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.detail_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_crop)

    def on_closing(self):
        self.stop_process()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()