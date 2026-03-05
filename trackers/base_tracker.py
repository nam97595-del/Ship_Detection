from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def update(self, bboxes_tho, frame):
        # Input: mảng [x1, y1, x2, y2, conf, cls]
        # Output: mảng [x1, y1, x2, y2, track_id, conf, cls]
        pass