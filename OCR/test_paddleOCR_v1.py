import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from paddleocr import PaddleOCR
import os
import logging

# Tắt log hệ thống
logging.getLogger("ppocr").setLevel(logging.WARNING)

class InteractiveOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Tương Tác: Khoanh vùng & Đọc (Final V3 - Padding Fix)")
        self.root.geometry("1300x750")

        # --- LOAD MODEL ---
        print("--- Đang tải model PaddleOCR... ---")
        try:
            # Init chuẩn cho version mới: Cấu hình ngay tại đây
            # use_angle_cls=True: Cho phép nhận diện chữ xoay ngược/nghiêng
            self.ocr = PaddleOCR(use_angle_cls=True, lang='vi')
            print("--- Đã tải xong! ---")
        except Exception as e:
            messagebox.showerror("Lỗi Init", f"Không thể tải model: {e}")

        # Biến lưu trữ
        self.original_image_cv = None  
        self.original_image_pil = None 
        self.scale_factor = 1.0        
        
        # Biến vẽ chuột
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id = None

        # --- GIAO DIỆN ---
        top_frame = tk.Frame(root, bg="#f0f0f0", pady=5)
        top_frame.pack(fill=tk.X)
        
        tk.Button(top_frame, text="📂 Chọn Ảnh Tàu", command=self.select_image, font=("Arial", 11, "bold"), bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=20)
        tk.Label(top_frame, text="💡 Dùng chuột KÉO và THẢ khung chữ nhật bao quanh biển số.", font=("Arial", 10, "italic"), bg="#f0f0f0").pack(side=tk.LEFT)

        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # TRÁI: Canvas vẽ
        self.canvas = tk.Canvas(main_pane, bg="gray", cursor="cross")
        main_pane.add(self.canvas, minsize=800)

        # PHẢI: Kết quả
        right_frame = tk.Frame(main_pane, bg="white", bd=2, relief=tk.GROOVE)
        main_pane.add(right_frame, minsize=400)

        tk.Label(right_frame, text="Vùng ảnh phát hiện được:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w", pady=(10,0), padx=5)
        self.lbl_preview = tk.Label(right_frame, bg="#dddddd", text="[Chưa chọn vùng]")
        self.lbl_preview.pack(pady=5)

        tk.Label(right_frame, text="Kết quả đọc được:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w", padx=5)
        self.txt_output = scrolledtext.ScrolledText(right_frame, width=40, height=20, font=("Consolas", 14))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Binding chuột
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg;*.bmp")])
        if not path: return

        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.original_image_cv = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        self.original_image_pil = Image.open(path).convert("RGB")
        self.display_image()

    def display_image(self):
        canvas_width = self.root.winfo_width() * 0.6 
        canvas_height = self.root.winfo_height() - 100
        if canvas_width < 100: canvas_width = 800 
        if canvas_height < 100: canvas_height = 600

        w, h = self.original_image_pil.size
        self.scale_factor = min(canvas_width/w, canvas_height/h)
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        img_resized = self.original_image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_mouse_down(self, event):
        self.rect_start_x = self.canvas.canvasx(event.x)
        self.rect_start_y = self.canvas.canvasy(event.y)
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.rect_start_x, self.rect_start_y, self.rect_start_x, self.rect_start_y, 
            outline="red", width=2
        )

    def on_mouse_drag(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.rect_start_x, self.rect_start_y, cur_x, cur_y)

    def on_mouse_up(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x1 = min(self.rect_start_x, cur_x) / self.scale_factor
        y1 = min(self.rect_start_y, cur_y) / self.scale_factor
        x2 = max(self.rect_start_x, cur_x) / self.scale_factor
        y2 = max(self.rect_start_y, cur_y) / self.scale_factor

        if (x2 - x1) < 10 or (y2 - y1) < 10: return
        self.crop_and_process(int(x1), int(y1), int(x2), int(y2))

    def crop_and_process(self, x1, y1, x2, y2):
        if self.original_image_cv is None: return
        
        cropped_img = self.original_image_cv[y1:y2, x1:x2]
        if cropped_img.size == 0: return

        # Xử lý ảnh: Chuyển xám + Làm nét + THÊM VIỀN
        processed_img = self.preprocess_image(cropped_img)

        # Preview
        preview_pil = Image.fromarray(processed_img) 
        preview_tk = ImageTk.PhotoImage(preview_pil)
        self.lbl_preview.config(image=preview_tk)
        self.lbl_preview.image = preview_tk 

        # Chạy OCR
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "Đang đọc...\n")
        thread = threading.Thread(target=self.run_ocr, args=(processed_img,))
        thread.start()

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        padded = cv2.copyMakeBorder(
            sharpened,
            50, 50, 50, 50,
            cv2.BORDER_CONSTANT,
            value=255
        )

        # 🔴 QUAN TRỌNG: CHUYỂN LẠI SANG 3 KÊNH
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        return padded_bgr 

    def run_ocr(self, img_input):
        try:
            result = self.ocr.ocr(img_input)

            if not result or result[0] is None:
                text_result = "Không tìm thấy chữ."
            else:
                res = result[0]
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])

                if len(texts) == 0:
                    text_result = "Không đọc được (Ảnh mờ hoặc nhiễu)."
                else:
                    text_result = ""
                    for text, score in zip(texts, scores):
                        text_result += f"{text}\n(Độ tin cậy: {int(score*100)}%)\n\n"

            self.root.after(0, lambda: self.update_ui_text(text_result))

        except Exception as e:
            err = f"Lỗi xử lý: {e}"
            print(err)
            self.root.after(0, lambda: self.update_ui_text(err))


    def update_ui_text(self, text):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveOCRApp(root)
    root.mainloop()