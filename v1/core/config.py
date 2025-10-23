from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfig:
    pose_model: str = "lite"
    camera_source: int | str = "media/video.mp4"
    frame_width: int = 640
    frame_height: int = 480
    min_confidence: float = 0.5
    num_poses: int = 6
    max_num_hands: int = 12
    gesture_score_threshold: float = 0.3
    smoothing_dead_zone: int = 15
    smoothing_factor: float = 0.3
    window_title: str = "Person Tracker"
