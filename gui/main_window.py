import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os

from engines.generic_engine import GenericEngine

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Model Testing Tool - Modular Version")
        self.geometry("600x750")
        self.resizable(False, False)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.folder_model = tk.StringVar()
        self.folder_video = tk.StringVar()
        self.folder_output = tk.StringVar()
        
        self.selected_model = tk.StringVar()
        self.selected_video = tk.StringVar()
        
        self.var_model_type = tk.StringVar(value="YOLO")
        self.var_tracker_type = tk.StringVar(value="ByteTrack")

        self.var_imgsz = tk.IntVar(value=640)
        self.var_stride = tk.IntVar(value=3)
        self.var_conf = tk.DoubleVar(value=0.5)
        
        self.stop_event = threading.Event()

        self.create_widgets()

    def create_widgets(self):
        lbl_title = tk.Label(self, text="CÔNG CỤ TEST MODEL (MODULAR ARCHITECTURE)", font=("Arial", 14, "bold"), fg="blue")
        lbl_title.pack(pady=10)

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        frame_arch = tk.LabelFrame(main_frame, text="1. Chọn Kiến Trúc (Architecture)", font=("Arial", 10, "bold"))
        frame_arch.pack(fill=tk.X, pady=5, ipadx=5, ipady=5)

        tk.Label(frame_arch, text="Loại Model:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        cb_model_type = ttk.Combobox(frame_arch, textvariable=self.var_model_type, values=["YOLO", "Faster R-CNN", "DETR"], state="readonly", width=15)
        cb_model_type.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(frame_arch, text="Loại Tracker:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        cb_tracker_type = ttk.Combobox(frame_arch, textvariable=self.var_tracker_type, values=["ByteTrack", "DeepSORT", "BoT-SORT"], state="readonly", width=15)
        cb_tracker_type.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        frame_file = tk.LabelFrame(main_frame, text="2. Cấu Hình File & Thư Mục", font=("Arial", 10, "bold"))
        frame_file.pack(fill=tk.X, pady=5, ipadx=5, ipady=5)

        tk.Button(frame_file, text="Chọn Thư Mục Model", command=self.browse_model_folder, width=20).grid(row=0, column=0, padx=5, pady=5)
        self.cb_models = ttk.Combobox(frame_file, textvariable=self.selected_model, state="readonly", width=35)
        self.cb_models.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(frame_file, text="Chọn Thư Mục Video", command=self.browse_video_folder, width=20).grid(row=1, column=0, padx=5, pady=5)
        self.cb_videos = ttk.Combobox(frame_file, textvariable=self.selected_video, state="readonly", width=35)
        self.cb_videos.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(frame_file, text="Chọn Thư Mục Lưu Output", command=self.browse_output_folder, width=20).grid(row=2, column=0, padx=5, pady=5)
        tk.Entry(frame_file, textvariable=self.folder_output, state="readonly", width=38).grid(row=2, column=1, padx=5, pady=5)

        frame_param = tk.LabelFrame(main_frame, text="3. Tham Số Test", font=("Arial", 10, "bold"))
        frame_param.pack(fill=tk.X, pady=5, ipadx=5, ipady=5)

        tk.Label(frame_param, text="Kích thước ảnh (imgsz):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_param, textvariable=self.var_imgsz, width=10).grid(row=0, column=1, sticky="w")

        tk.Label(frame_param, text="Bỏ qua khung hình (stride):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_param, textvariable=self.var_stride, width=10).grid(row=1, column=1, sticky="w")

        tk.Label(frame_param, text="Ngưỡng độ tin cậy (conf):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_param, textvariable=self.var_conf, width=10).grid(row=2, column=1, sticky="w")

        frame_btn = tk.Frame(main_frame)
        frame_btn.pack(pady=20)

        tk.Button(frame_btn, text="BẮT ĐẦU TEST", command=self.start_thread, bg="green", fg="white", font=("Arial", 12, "bold"), width=15).grid(row=0, column=0, padx=10)
        tk.Button(frame_btn, text="DỪNG LẠI", command=self.stop_testing, bg="red", fg="white", font=("Arial", 12, "bold"), width=15).grid(row=0, column=1, padx=10)

    def browse_model_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_model.set(path)
            files = [f for f in os.listdir(path) if f.endswith(('.pt', '.pth', '.engine', '.onnx'))]
            self.cb_models['values'] = files
            if files: self.cb_models.current(0)

    def browse_video_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_video.set(path)
            files = [f for f in os.listdir(path) if f.endswith(('.mp4', '.avi', '.mkv'))]
            self.cb_videos['values'] = files
            if files: self.cb_videos.current(0)

    def browse_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_output.set(path)

    def start_thread(self):
        if not self.folder_model.get() or not self.selected_model.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Model!")
            return
        if not self.folder_video.get() or not self.selected_video.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Video!")
            return
        if not self.folder_output.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Output Folder!")
            return

        model_path = os.path.join(self.folder_model.get(), self.selected_model.get())
        video_path = os.path.join(self.folder_video.get(), self.selected_video.get())

        self.stop_event.clear()

        engine = GenericEngine(
            model_type=self.var_model_type.get(),
            model_path=model_path,
            tracker_type=self.var_tracker_type.get(),
            video_path=video_path,
            output_folder=self.folder_output.get(),
            imgsz=self.var_imgsz.get(),
            stride=self.var_stride.get(),
            conf=self.var_conf.get(),
            stop_event=self.stop_event
        )

        threading.Thread(target=engine.run, daemon=True).start()

    def stop_testing(self):
        self.stop_event.set()

    def on_closing(self):
        self.stop_event.set()
        self.destroy()