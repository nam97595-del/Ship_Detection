import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import threading
import os
import pyodbc
from engines.yolo_engine import YoloTester

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Giám sát Tàu biển - YOLO & OCR")
        self.root.geometry("1400x900")
        
        # --- Biến cấu hình ---
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.conf_val = tk.DoubleVar(value=0.5)
        self.stride_val = tk.IntVar(value=2)
        self.use_ocr_var = tk.BooleanVar(value=True) 
        self.selected_track_id = None 

        self.setup_navbar()
        
        # Container chứa các trang
        self.container = tk.Frame(self.root)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Khởi tạo các trang (Frames)
        self.frames = {}
        self.setup_monitoring_page()
        self.setup_database_page()

        # Hiển thị trang mặc định
        self.show_frame("monitoring")

    def setup_navbar(self):
        """Tạo thanh điều hướng phía trên"""
        navbar = tk.Frame(self.root, bg="#2c3e50", height=50)
        navbar.pack(side=tk.TOP, fill=tk.X)

        # Style cho nút Navbar
        nav_style = {"bg": "#2c3e50", "fg": "white", "font": ("Arial", 11, "bold"), 
                     "relief": "flat", "activebackground": "#34495e", "activeforeground": "white", "padx": 20}

        tk.Button(navbar, text="🏠 Hệ thống giám sát", **nav_style, 
                  command=lambda: self.show_frame("monitoring")).pack(side=tk.LEFT)
        
        tk.Button(navbar, text="📊 Cơ sở dữ liệu", **nav_style, 
                  command=lambda: self.show_frame("database")).pack(side=tk.LEFT)
        
        tk.Button(navbar, text="🚪 Đăng xuất", **nav_style, 
                  command=self.logout).pack(side=tk.RIGHT)

    def show_frame(self, page_name):
        """Chuyển đổi giữa các trang"""
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name == "database":
            self.load_database_data()

    # ================= TRANG GIÁM SÁT =================
    def setup_monitoring_page(self):
        page = tk.Frame(self.container, bg="white")
        self.frames["monitoring"] = page
        page.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Tách cấu trúc UI cũ vào đây
        control_frame = tk.Frame(page, bg="#f0f0f0", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(control_frame, text="CẤU HÌNH", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
        tk.Button(control_frame, text="Chọn Model", command=self.choose_model).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.model_path, fg="blue", wraplength=280, bg="#f0f0f0").pack()
        tk.Button(control_frame, text="Chọn Video", command=self.choose_video).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.video_path, fg="blue", wraplength=280, bg="#f0f0f0").pack()
        tk.Button(control_frame, text="Thư mục lưu KQ", command=self.choose_output_folder).pack(fill=tk.X, pady=5)
        tk.Label(control_frame, textvariable=self.output_dir, fg="green", wraplength=280, bg="#f0f0f0").pack()
        tk.Checkbutton(control_frame, text="Bật OCR", variable=self.use_ocr_var, fg="red", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(pady=20)
        tk.Button(control_frame, text="▶ BẮT ĐẦU", bg="green", fg="white", font=("Arial", 12, "bold"), height=2, command=self.start_process).pack(fill=tk.X, pady=10)
        tk.Button(control_frame, text="⏹ DỪNG", bg="red", fg="white", command=self.stop_process).pack(fill=tk.X)

        self.detail_frame = tk.Frame(page, width=300, bg="white")
        self.detail_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        tk.Label(self.detail_frame, text="CHI TIẾT TÀU", font=("Arial", 12, "bold"), bg="white").pack(pady=10)
        self.detail_canvas = tk.Canvas(self.detail_frame, width=280, height=200, bg="gray")
        self.detail_canvas.pack(pady=5)
        self.detail_text = tk.Label(self.detail_frame, text="Click vào box trên video...", font=("Arial", 11), wraplength=280, justify=tk.LEFT, bg="white")
        self.detail_text.pack()

        self.canvas_video = tk.Canvas(page, bg="black")
        self.canvas_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_video.bind("<Button-1>", self.on_canvas_click)

    # ================= TRANG CƠ SỞ DỮ LIỆU =================
    def setup_database_page(self):
        page = tk.Frame(self.container, bg="#ecf0f1")
        self.frames["database"] = page
        page.grid(row=0, column=0, sticky="nsew")

        tk.Label(page, text="NHẬT KÝ SHIPLOG", font=("Arial", 18, "bold"), bg="#ecf0f1").pack(pady=20)

        # Tạo bảng hiển thị
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

        self.tree = ttk.Treeview(page, columns=("ID", "Class", "SoHieu", "Gio"), show='headings')
        self.tree.heading("ID", text="ID Tracking")
        self.tree.heading("Class", text="Loại tàu")
        self.tree.heading("SoHieu", text="Số hiệu")
        self.tree.heading("Gio", text="Giờ phát hiện")
        
        self.tree.column("ID", width=100, anchor=tk.CENTER)
        self.tree.column("Class", width=200)
        self.tree.column("SoHieu", width=200)
        self.tree.column("Gio", width=250)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)

        btn_refresh = tk.Button(page, text="🔄 Làm mới dữ liệu", command=self.load_database_data, bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10)
        btn_refresh.pack(pady=10)

    def load_database_data(self):
        """Truy vấn dữ liệu từ SQL Server"""
        # Xóa dữ liệu cũ trên bảng UI
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=.\\SQLEXPRESS;'
                'DATABASE=shipdb;'
                'Trusted_Connection=yes;'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT track_id, class_name, ISNULL(so_hieu, 'Unknown'), gio_phat_hien FROM shiplog ORDER BY gio_phat_hien DESC")
            
            for row in cursor.fetchall():
                self.tree.insert("", tk.END, values=(row[0], row[1], row[2], row[3]))
            conn.close()
        except Exception as e:
            print(f"Lỗi DB: {e}")

    # ================= CÁC HÀM XỬ LÝ KHÁC =================
    def logout(self):
        if messagebox.askyesno("Đăng xuất", "Bạn có chắc chắn muốn đăng xuất?"):
            self.stop_process()
            self.root.destroy()
            # Ở đây bạn có thể gọi lại màn hình login.py nếu cần
            os.system('python main.py') 

    # --- Các hàm logic YOLO giữ nguyên như cũ ---
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
        if self.selected_track_id is not None and hasattr(self, 'engine'):
            if self.selected_track_id in self.engine.current_objects:
                ocr_txt = self.engine.current_objects[self.selected_track_id]["ocr"]
                if ocr_txt and ocr_txt != "...":
                    self.detail_text.config(text=f"ID: {self.selected_track_id}\n\nOCR: {ocr_txt}")

    def on_canvas_click(self, event):
        if not hasattr(self, 'engine') or not hasattr(self, 'last_scale'): return
        x_click = (event.x - self.last_offset[0]) / self.last_scale
        y_click = (event.y - self.last_offset[1]) / self.last_scale
        for tid, obj in self.engine.current_objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            if x1 <= x_click <= x2 and y1 <= y_click <= y2:
                self.selected_track_id = tid
                self.show_crop(obj["crop"])
                self.detail_text.config(text=f"ID: {tid}\n\nĐang đọc OCR...")
                self.engine.request_manual_ocr(tid)
                break

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