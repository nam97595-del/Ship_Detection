import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from paddleocr import PaddleOCR
import logging

logging.getLogger("ppocr").setLevel(logging.WARNING)


class AutoOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto OCR + Preprocessing (Ship ID)")
        self.root.geometry("1350x780")

        print("⏳ Đang tải PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        print("✅ PaddleOCR sẵn sàng")

        self.original_image_cv = None
        self.original_image_pil = None
        self.scale_factor = 1.0

        # ================= TOP BAR =================
        top_frame = tk.Frame(root, pady=5, bg="#f0f0f0")
        top_frame.pack(fill=tk.X)

        tk.Button(
            top_frame, text="📂 Chọn ảnh",
            command=self.select_image,
            font=("Arial", 11, "bold"),
            bg="#2196F3", fg="white"
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            top_frame, text="🔍 OCR tự động",
            command=self.auto_detect_and_ocr,
            font=("Arial", 11, "bold"),
            bg="#4CAF50", fg="white"
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            top_frame,
            text="Auto detect + OCR với tiền xử lý ảnh",
            bg="#f0f0f0",
            font=("Arial", 10, "italic")
        ).pack(side=tk.LEFT, padx=20)

        # ================= MAIN =================
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ===== LEFT: IMAGE =====
        self.canvas = tk.Canvas(main_pane, bg="gray")
        main_pane.add(self.canvas, minsize=850)

        # ===== RIGHT =====
        right_frame = tk.Frame(main_pane, bg="white", bd=2, relief=tk.GROOVE)
        main_pane.add(right_frame, minsize=450)

        # OCR TEXT
        tk.Label(
            right_frame, text="📄 Kết quả OCR",
            font=("Arial", 11, "bold"), bg="white"
        ).pack(anchor="w", padx=5, pady=(10, 0))

        self.txt_output = scrolledtext.ScrolledText(
            right_frame, width=42, height=18,
            font=("Consolas", 13)
        )
        self.txt_output.pack(fill=tk.X, padx=5, pady=5)

        # PREPROCESS IMAGE
        tk.Label(
            right_frame, text="🧪 Ảnh sau tiền xử lý",
            font=("Arial", 10, "bold"), bg="white"
        ).pack(anchor="w", padx=5, pady=(10, 0))

        self.preprocess_label = tk.Label(
            right_frame, bg="black", width=420, height=240
        )
        self.preprocess_label.pack(padx=5, pady=5)

    # ================= IMAGE =================
    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.png;*.jpeg;*.bmp;*.jfif")]
        )
        if not path:
            return

        self.original_image_cv = cv2.imread(path)
        self.original_image_pil = Image.open(path).convert("RGB")
        self.display_image()

        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "📂 Ảnh đã load\n")

    def display_image(self):
        canvas_w = int(self.root.winfo_width() * 0.6)
        canvas_h = self.root.winfo_height() - 120
        canvas_w = max(canvas_w, 800)
        canvas_h = max(canvas_h, 600)

        w, h = self.original_image_pil.size
        self.scale_factor = min(canvas_w / w, canvas_h / h)

        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)

        resized = self.original_image_pil.resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")

        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        self.canvas.config(width=canvas_w, height=canvas_h)
        self.canvas.create_image(
            self.offset_x, self.offset_y,
            anchor=tk.NW, image=self.tk_image
        )

    # ================= PREPROCESS =================
    def preprocess_for_ocr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpen = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)

        _, binary = cv2.threshold(
            sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def show_preprocessed_image(self, img):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((420, 240), Image.Resampling.LANCZOS)
        self.tk_pre_img = ImageTk.PhotoImage(img_pil)
        self.preprocess_label.config(image=self.tk_pre_img)

    # ================= OCR =================
    def auto_detect_and_ocr(self):
        if self.original_image_cv is None:
            messagebox.showwarning("Chưa có ảnh", "Vui lòng chọn ảnh trước.")
            return

        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "🔍 Đang OCR...\n")

        threading.Thread(target=self.run_auto_ocr).start()

    def run_auto_ocr(self):
        try:
            pre_img = self.preprocess_for_ocr(self.original_image_cv)
            self.root.after(
                0, lambda img=pre_img: self.show_preprocessed_image(img)
            )

            result = self.ocr.ocr(self.original_image_cv)

            if not result or result[0] is None:
                self.root.after(
                    0, lambda: self.update_text("❌ Không phát hiện chữ")
                )
                return

            res = result[0]
            boxes = res["dt_polys"]
            texts = res["rec_texts"]
            scores = res["rec_scores"]

            output = ""
            for txt, score in zip(texts, scores):
                output += f"{txt} ({int(score * 100)}%)\n"

            self.root.after(0, lambda: self.update_text(output))
            self.root.after(0, lambda: self.draw_boxes(boxes))

        except Exception as e:
            self.root.after(
                0, lambda msg=str(e): self.update_text(f"Lỗi OCR: {msg}")
            )

    def draw_boxes(self, boxes):
        self.canvas.delete("ocr_box")
        for box in boxes:
            pts = []
            for x, y in box:
                cx = x * self.scale_factor + self.offset_x
                cy = y * self.scale_factor + self.offset_y
                pts.extend([cx, cy])

            self.canvas.create_polygon(
                pts, outline="red", width=2,
                fill="", tags="ocr_box"
            )

    def update_text(self, text):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, text)


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoOCRApp(root)
    root.mainloop()
