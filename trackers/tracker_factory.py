from .bytetrack_wrapper import ByteTrackWrapper
from .deepsort_wrapper import DeepSortWrapper
from .botsort_wrapper import BotSortWrapper
from .strongsort_wrapper import StrongSortWrapper
from .boostrack_wrapper import BoostTrackWrapper
from .hybridsort_wrapper import HybridSortWrapper
def create_tracker(tracker_type):
    if tracker_type == "ByteTrack":
        return ByteTrackWrapper()
    elif tracker_type == "DeepOcSort":
        return DeepSortWrapper()
    elif tracker_type == "BoT-SORT":
        return BotSortWrapper()
    elif tracker_type == "StrongSORT":
        return StrongSortWrapper()
    elif tracker_type == "BoostTrack":
        return BoostTrackWrapper()
    elif tracker_type == "HybridSort":
        return HybridSortWrapper()
    else:
        raise ValueError(f"Chưa hỗ trợ tracker: {tracker_type}")
    
    