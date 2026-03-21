import numpy as np
from boxmot import OcSort
from .base_tracker import BaseTracker
from pathlib import Path

class OcSortWrapper(BaseTracker):
    def __init__(self):
        print("🔹 Đang khởi tạo BoostTrack (Thuật toán có chỉ số IDF1 cực cao)...")
        self.tracker = OcSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device='cuda:0',
            half=True
        )

    def update(self, bboxes_tho, frame):
        if len(bboxes_tho) == 0:
            return np.empty((0, 7))
        
        tracks = self.tracker.update(bboxes_tho, frame)
        return tracks