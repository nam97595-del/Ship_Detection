import numpy as np
from boxmot import StrongSort
from .base_tracker import BaseTracker
from pathlib import Path

class StrongSortWrapper(BaseTracker):
    def __init__(self):
        print("🔹 Đang khởi tạo StrongSORT (Khả năng nhớ ID dài hạn cực tốt)...")
        self.tracker = StrongSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device='cuda:0',
            half=True
        )

    def update(self, bboxes_tho, frame):
        if len(bboxes_tho) == 0:
            return np.empty((0, 7))
        
        tracks = self.tracker.update(bboxes_tho, frame)
        return tracks