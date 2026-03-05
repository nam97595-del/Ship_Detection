from .bytetrack_wrapper import ByteTrackWrapper

def create_tracker(tracker_type):
    if tracker_type == "ByteTrack":
        return ByteTrackWrapper()
    else:
        raise ValueError(f"Chưa hỗ trợ tracker: {tracker_type}")
    