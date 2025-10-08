from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameStats:
    frame_index: int
    people: int
    hands: int
    gestures: int
    pose_model: str
    show_hands: bool
    show_poses: bool
