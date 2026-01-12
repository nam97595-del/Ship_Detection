import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
# Import Engine YOLO từ folder engines
from engines.yolo_engine import YoloTester
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Model Testing Tool - Modular Version")
        self.geometry("600x700")
        self.resizable(False, False)
        
        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Biến lưu trữ dữ liệu ---
        self.folder_model = tk.StringVar()
        self.folder_video = tk.StringVar()
        self.folder_output = tk.StringVar()
        
        self.selected_model = tk.StringVar()
        self.selected_video = tk.StringVar()
        
        # Tham số cấu hình mặc định
        self.var_imgsz = tk.IntVar(value=640)
        self.var_stride = tk.IntVar(value=3)
        self.var_conf = tk.DoubleVar(value=0.5)
        
        # Sự kiện dừng thread
        self.stop_event = threading.Event()

        # Tạo giao diện
        self.create_widgets()

    def create_widgets(self):
        # --- HEADER ---
        lbl_title = tk.Label(self, text="CÔNG CỤ TEST MODEL YOLO (MODULAR)", font=("Arial", 16, "bold"), fg="#2c3e50")
        lbl_title.pack(pady=15)

        # --- CONTAINER CHÍNH ---
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. KHUNG CHỌN MODEL
        grp_model = ttk.LabelFrame(main_frame, text="1. Chọn Model (YOLO)", padding=10)
        grp_model.pack(fill=tk.X, pady=5)
        
        btn_browse_model = ttk.Button(grp_model, text="📂 Chọn Folder chứa Models", command=self.browse_model_folder)
        btn_browse_model.pack(fill=tk.X)
        
        self.cb_models = ttk.Combobox(grp_model, textvariable=self.selected_model, state="readonly")
        self.cb_models.pack(fill=tk.X, pady=5)
        self.cb_models.set("<- Hãy chọn folder trước")

        # 2. KHUNG CHỌN VIDEO
        grp_video = ttk.LabelFrame(main_frame, text="2. Chọn Video Test", padding=10)
        grp_video.pack(fill=tk.X, pady=5)

        btn_browse_video = ttk.Button(grp_video, text="📂 Chọn Folder chứa Video", command=self.browse_video_folder)
        btn_browse_video.pack(fill=tk.X)

        self.cb_videos = ttk.Combobox(grp_video, textvariable=self.selected_video, state="readonly")
        self.cb_videos.pack(fill=tk.X, pady=5)
        self.cb_videos.set("<- Hãy chọn folder trước")

        # 3. KHUNG CHỌN OUTPUT
        grp_out = ttk.LabelFrame(main_frame, text="3. Thư mục lưu kết quả", padding=10)
        grp_out.pack(fill=tk.X, pady=5)
        
        btn_browse_out = ttk.Button(grp_out, text="📂 Chọn Output Folder", command=self.browse_output_folder)
        btn_browse_out.pack(side=tk.LEFT)
        
        entry_out_path = tk.Entry(grp_out, textvariable=self.folder_output, state="readonly")
        entry_out_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 4. CẤU HÌNH (SETTINGS)
        grp_setting = ttk.LabelFrame(main_frame, text="4. Cấu hình tham số", padding=10)
        grp_setting.pack(fill=tk.X, pady=5)

        # Grid Layout cho phần setting
        ttk.Label(grp_setting, text="Image Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grp_setting, textvariable=self.var_imgsz, width=10).grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(grp_setting, text="Skip Frame:").grid(row=0, column=2, sticky="w")
        ttk.Entry(grp_setting, textvariable=self.var_stride, width=10).grid(row=0, column=3, sticky="w", padx=10)

        ttk.Label(grp_setting, text="Conf Thresh:").grid(row=1, column=0, sticky="w", pady=10)
        scl_conf = ttk.Scale(grp_setting, from_=0.1, to=1.0, variable=self.var_conf, command=lambda v: self.var_conf.set(round(float(v), 2)))
        scl_conf.grid(row=1, column=1, columnspan=3, sticky="we", padx=10)
        tk.Label(grp_setting, textvariable=self.var_conf).grid(row=1, column=4)

        # 5. NÚT CHẠY
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)

        self.btn_run = tk.Button(btn_frame, text="▶ CHẠY TEST NGAY", font=("Arial", 12, "bold"), bg="#27ae60", fg="white", height=2, command=self.start_thread)
        self.btn_run.pack(fill=tk.X)

        lbl_note = tk.Label(main_frame, text="*Nhấn 'q' trên cửa sổ video hoặc tắt tool để dừng.", fg="gray", font=("Arial", 9, "italic"))
        lbl_note.pack()

    # --- CÁC HÀM XỬ LÝ LOGIC ---

    def browse_model_folder(self):
        path = filedialog.askdirectory(title="Chọn thư mục chứa Model (.pt, .engine)")
        if path:
            self.folder_model.set(path)
            # Lọc lấy file .pt, .onnx, .engine
            files = [f for f in os.listdir(path) if f.endswith(('.pt', '.pth', '.engine', '.onnx'))]
            if files:
                self.cb_models['values'] = files
                self.cb_models.current(0)
            else:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy file model nào trong thư mục này!")

    def browse_video_folder(self):
        path = filedialog.askdirectory(title="Chọn thư mục chứa Video")
        if path:
            self.folder_video.set(path)
            # Lọc lấy file video
            files = [f for f in os.listdir(path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.MP4'))]
            if files:
                self.cb_videos['values'] = files
                self.cb_videos.current(0)
            else:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy video nào!")

    def browse_output_folder(self):
        path = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if path:
            self.folder_output.set(path)

    def start_thread(self):
        # Kiểm tra dữ liệu đầu vào
        if not self.folder_model.get() or not self.selected_model.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Model!")
            return
        if not self.folder_video.get() or not self.selected_video.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Video!")
            return
        if not self.folder_output.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn Output Folder!")
            return

        # Tạo đường dẫn đầy đủ
        model_path = os.path.join(self.folder_model.get(), self.selected_model.get())
        video_path = os.path.join(self.folder_video.get(), self.selected_video.get())

        # Reset cờ dừng
        self.stop_event.clear()

        # Gọi Engine YOLO từ file yolo_engine.py
        tester = YoloTester(
            model_path=model_path,
            video_path=video_path,
            output_folder=self.folder_output.get(),
            imgsz=self.var_imgsz.get(),
            stride=self.var_stride.get(),
            conf=self.var_conf.get(),
            iou=0.45,
            stop_event=self.stop_event
        )
        
        # Chạy trong luồng riêng để không treo giao diện
        t = threading.Thread(target=tester.run)
        t.start()

    def on_closing(self):
        self.stop_event.set()
        self.destroy()