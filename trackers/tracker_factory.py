from .bytetrack_wrapper import ByteTrackWrapper
from .deepsort_wrapper import DeepSortWrapper
from .botsort_wrapper import BotSortWrapper
def create_tracker(tracker_type):
    if tracker_type == "ByteTrack":
        return ByteTrackWrapper()
    elif tracker_type == "DeepOcSort":
        return DeepSortWrapper()
    elif tracker_type == "BoT-SORT":
        return BotSortWrapper()
    else:
        raise ValueError(f"Chưa hỗ trợ tracker: {tracker_type}")
    
    