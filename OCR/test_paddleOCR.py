import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import threading
from paddleocr import PaddleOCR
import logging

# Tắt log rác
logging.getLogger("ppocr").setLevel(logging.WARNING)

class AutoOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto OCR - Final Fix (PIL Drawing Strategy)")
        self.root.geometry("1300x750")

        print("⏳ Đang tải PaddleOCR...")
        try:
            # use_angle_cls=True: Tự động xoay chiều chữ
            # Bỏ show_log=False để tương thích mọi phiên bản
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("✅ PaddleOCR sẵn sàng")
        except Exception as e:
            messagebox.showerror("Lỗi Init", f"Không thể tải model: {e}")

        self.original_image_cv = None
        self.current_pil_image = None # Ảnh đang hiển thị (đã vẽ hoặc chưa)

        # --- GIAO DIỆN ---
        top_frame = tk.Frame(root, pady=5, bg="#f0f0f0")
        top_frame.pack(fill=tk.X)

        tk.Button(top_frame, text="📂 Chọn ảnh", command=self.select_image,
                  font=("Arial", 11, "bold"), bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=10)

        tk.Button(top_frame, text="▶ Quét chữ", command=self.auto_detect_and_ocr,
                  font=("Arial", 11, "bold"), bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)

        tk.Label(top_frame, text="✅ Sử dụng công nghệ vẽ PIL Draw chính xác tuyệt đối.", bg="#f0f0f0", font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=20)

        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas hiển thị ảnh
        self.canvas = tk.Canvas(main_pane, bg="gray")
        main_pane.add(self.canvas, minsize=850)

        # Khung kết quả text
        right_frame = tk.Frame(main_pane, bg="white", bd=2, relief=tk.GROOVE)
        main_pane.add(right_frame, minsize=400)

        tk.Label(right_frame, text="📄 Kết quả OCR", font=("Arial", 11, "bold"), bg="white").pack(anchor="w", padx=5, pady=(10, 0))

        self.txt_output = scrolledtext.ScrolledText(right_frame, width=40, height=30, font=("Consolas", 13))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg;*.bmp;*.jfif")])
        if not path: return

        # 1. Đọc ảnh bằng OpenCV chế độ MÀU CHUẨN (Bỏ kênh Alpha nếu có)
        # Sử dụng imdecode để đọc được đường dẫn tiếng Việt
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.original_image_cv = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

        # 2. Chuyển sang RGB ngay lập tức để đồng bộ màu sắc
        img_rgb = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2RGB)
        self.current_pil_image = Image.fromarray(img_rgb)
        
        # 3. Hiển thị
        self.display_image(self.current_pil_image)

        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "📂 Đã tải ảnh xong.\n")

    def display_image(self, image_pil):
        """Hàm hiển thị ảnh lên Canvas (Resize vừa khung)"""
        canvas_w = int(self.root.winfo_width() * 0.6)
        canvas_h = self.root.winfo_height() - 100
        canvas_w = max(canvas_w, 800)
        canvas_h = max(canvas_h, 600)

        w, h = image_pil.size
        # Tính tỉ lệ scale giữ nguyên khung hình
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        # Căn giữa ảnh
        x_center = (canvas_w - new_w) // 2
        y_center = (canvas_h - new_h) // 2
        
        self.canvas.config(width=canvas_w, height=canvas_h)
        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.tk_image)

    def auto_detect_and_ocr(self):
        if self.original_image_cv is None:
            messagebox.showwarning("Lỗi", "Chưa chọn ảnh!")
            return

        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "🔍 Đang quét...\n")
        threading.Thread(target=self.run_auto_ocr).start()

    def run_auto_ocr(self):
        try:
            # Gọi OCR. 
            # Lưu ý: PaddleOCR nhận input là numpy array (RGB hoặc BGR đều được, nhưng RGB tốt hơn)
            # Ta lấy ảnh từ PIL Image đã convert sang RGB lúc nãy để đảm bảo đồng nhất
            img_input = np.array(self.current_pil_image)
            
            result = self.ocr.ocr(img_input)

            if not result or result[0] is None:
                self.root.after(0, lambda: self.update_text("❌ Không tìm thấy chữ."))
                return

            # --- XỬ LÝ KẾT QUẢ ---
            res = result[0]
            data_items = []

            # Logic tương thích cả 2 phiên bản (Dict hoặc List)
            if isinstance(res, dict): 
                boxes = res.get("dt_polys", [])
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                data_items = zip(boxes, texts, scores)
            else:
                for line in res:
                    # line = [[x1,y1, x2,y2...], ("text", 0.99)]
                    data_items.append((line[0], line[1][0], line[1][1]))

            # --- VẼ TRỰC TIẾP LÊN ẢNH PIL (CHÍNH XÁC HƠN OPENCV) ---
            # Tạo đối tượng vẽ trên ảnh PIL hiện tại
            draw_img = self.current_pil_image.copy()
            draw = ImageDraw.Draw(draw_img)
            output_text = ""

            for box, txt, score in data_items:
                # box là list các điểm [[x,y], [x,y]...]
                # Convert sang list phẳng [x1, y1, x2, y2...] để PIL vẽ
                flatten_box = []
                for point in box:
                    flatten_box.append((point[0], point[1]))
                
                # Vẽ khung đỏ (outline), độ dày 3
                draw.polygon(flatten_box, outline="red", width=3)

                output_text += f"► {txt}\n(Độ tin cậy: {int(score*100)}%)\n\n"

            # Cập nhật UI
            self.root.after(0, lambda: self.update_text(output_text))
            # Hiển thị ảnh ĐÃ VẼ KHUNG
            self.root.after(0, lambda: self.display_image(draw_img))

        except Exception as e:
            err = f"Lỗi OCR: {e}"
            print(err)
            self.root.after(0, lambda msg=err: self.update_text(msg))

    def update_text(self, text):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoOCRApp(root)
    root.mainloop()