import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR

# Tắt log của Paddle
logging.getLogger("ppocr").setLevel(logging.WARNING)

class ShipOCR:
    def __init__(self, lang="en", use_angle_cls=True):
        print(">> Đang khởi tạo PaddleOCR (có thể mất vài giây)...")
        # Khởi tạo model
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        print(">> PaddleOCR đã sẵn sàng!")

    def preprocess_for_ocr(self, img):
        """
        Tiền xử lý ảnh: Tăng độ tương phản và làm nét chữ
        giúp đọc tốt hơn trong điều kiện sương mù hoặc lóa nắng.
        """
        try:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Cân bằng histogram cục bộ (CLAHE) để xử lý chói sáng
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast = clahe.apply(gray)
            
            # Làm nét ảnh (Sharpening)
            blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
            sharpen = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)
            
            # Trả về ảnh sharpen (xám) hoặc binary tùy điều kiện
            # Trả về ảnh sharpen để OCR tự xử lý threshold
            return sharpen
        except Exception as e:
            print(f"Lỗi Preprocess: {e}")
            return img

    def ocr_image(self, image_cv):
        results = []
        try:
            processed_img = self.preprocess_for_ocr(image_cv)

            if len(processed_img.shape) == 2:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

            ocr_result = self.ocr.ocr(processed_img)

            if not ocr_result or ocr_result[0] is None:
                return results

            data = ocr_result[0]

            for line in data:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue

                box = line[0]
                rec = line[1]

                if isinstance(rec, (list, tuple)):
                    if len(rec) == 2:
                        text, score = rec
                    elif len(rec) == 1:
                        text = rec[0]
                        score = 0.0
                    else:
                        continue
                else:
                    text = str(rec)
                    score = 0.0

                if score > 0.6:
                    results.append({
                        "text": text,
                        "score": float(score),
                        "box": box
                    })

        except Exception as e:
            print(f"Lỗi OCR: {e}")

        return results
