import numpy as np
from boxmot import ByteTrack
from .base_tracker import BaseTracker

class ByteTrackWrapper(BaseTracker):
    def __init__(self):
        # Bytetrack
        self.tracker = ByteTrack(
            track_thresh=0.2,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30
        )

    def update(self, bboxes_tho, frame):
        if len(bboxes_tho) == 0:
            return np.empty((0, 7))
        
        tracks = self.tracker.update(bboxes_tho, frame)
        return tracks