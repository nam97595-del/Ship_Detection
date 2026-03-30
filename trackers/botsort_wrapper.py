import numpy as np
from boxmot import BotSort
from .base_tracker import BaseTracker
from pathlib import Path

class BotSortWrapper(BaseTracker):
    def __init__(self):
        print("🔹 Đang khởi tạo BotSort (Tối ưu cho camera di chuyển/rung lắc)...")
        self.tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device="cuda:0",
            half=True
        )

    def update(self, bboxes_tho, frame):
        if len(bboxes_tho) == 0:
            return np.empty((0, 7))
        tracks = self.tracker.update(bboxes_tho, frame)
        return tracks