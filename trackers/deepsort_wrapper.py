import numpy as np
from boxmot import DeepOcSort
from .base_tracker import BaseTracker
from pathlib import Path

class DeepSortWrapper(BaseTracker):
    def __init__(self):
        print("🔹 Đang khởi tạo DeepOcSort (Bản nâng cấp của DeepSORT)...")
        self.tracker = DeepOcSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'), # Tự động tải model ReID
            device='cuda:0', 
            half=True # F16 tiết kiệm vram GPU
        )

    def update(self, bboxes_tho, frame):
        if len(bboxes_tho) == 0:
            return np.empty((0, 7))
        
        tracks = self.tracker.update(bboxes_tho, frame)
        return tracks