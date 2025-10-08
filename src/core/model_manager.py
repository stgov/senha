from __future__ import annotations

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple


class ModelManager:
    def __init__(
        self,
        pose_model: str = "lite",
        num_poses: int = 6,
        max_num_hands: int = 12,
        min_confidence: float = 0.5
    ):
        self.pose_model = pose_model
        self.num_poses = num_poses
        self.max_num_hands = max_num_hands
        self.min_confidence = min_confidence
        
        self.gesture_model_path = "models/gesture_recognizer.task"
        self.pose_model_path = self._get_pose_model_path(pose_model)
        
        self.gesture_recognizer = self._create_gesture_recognizer()
        self.pose_landmarker = self._create_pose_landmarker()
    
    def _get_pose_model_path(self, pose_model: str) -> str:
        if pose_model == "lite":
            return "models/pose_landmarker_lite.task"
        elif pose_model == "full":
            return "models/pose_landmarker_full.task"
        else:
            raise ValueError("pose_model must be 'lite' or 'full'")
    
    def _create_gesture_recognizer(self) -> vision.GestureRecognizer:
        base_options = python.BaseOptions(model_asset_path=self.gesture_model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            num_hands=self.max_num_hands,
        )
        return vision.GestureRecognizer.create_from_options(options)
    
    def _create_pose_landmarker(self) -> vision.PoseLandmarker:
        base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            num_poses=self.num_poses,
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def process_frame(
        self,
        mp_image: mp.Image
    ) -> Tuple[vision.GestureRecognizerResult, vision.PoseLandmarkerResult]:
        gesture_result = self.gesture_recognizer.recognize(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        return gesture_result, pose_result
