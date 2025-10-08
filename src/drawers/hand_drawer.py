from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from typing import List, Sequence, Tuple


class HandDrawer:
    def __init__(self, gesture_threshold: float = 0.5):
        self.gesture_threshold = gesture_threshold
        self.colors: List[Tuple[int, int, int]] = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128)
        ]
    
    def draw(self, frame: np.ndarray, gesture_result: vision.GestureRecognizerResult) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if not gesture_result.hand_landmarks:
            return annotated_frame
        
        for idx, hand_landmarks in enumerate(gesture_result.hand_landmarks):
            color = self.colors[idx % len(self.colors)]
            self._draw_landmarks(annotated_frame, hand_landmarks, color)
            self._draw_connections(annotated_frame, hand_landmarks, color)
            self._draw_gesture_label(annotated_frame, hand_landmarks, gesture_result, idx, color)
        
        return annotated_frame
    
    def _draw_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
        color: Tuple[int, int, int]
    ) -> None:
        h, w = frame.shape[:2]
        
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, color, -1)
    
    def _draw_connections(
        self,
        frame: np.ndarray,
        hand_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
        color: Tuple[int, int, int]
    ) -> None:
        h, w = frame.shape[:2]
        connections = mp.solutions.hands.HAND_CONNECTIONS
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (
                int(hand_landmarks[start_idx].x * w),
                int(hand_landmarks[start_idx].y * h)
            )
            end_point = (
                int(hand_landmarks[end_idx].x * w),
                int(hand_landmarks[end_idx].y * h)
            )
            cv2.line(frame, start_point, end_point, color, 2)
    
    def _draw_gesture_label(
        self,
        frame: np.ndarray,
        hand_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
        gesture_result: vision.GestureRecognizerResult,
        idx: int,
        color: Tuple[int, int, int]
    ) -> None:
        if not gesture_result.gestures or idx >= len(gesture_result.gestures):
            return
        
        gesture_list = gesture_result.gestures[idx]
        if not gesture_list:
            return
        
        top_gesture = gesture_list[0]
        if top_gesture.score < self.gesture_threshold:
            return
        
        h, w = frame.shape[:2]
        wrist_x = int(hand_landmarks[0].x * w)
        wrist_y = int(hand_landmarks[0].y * h)
        label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        
        cv2.putText(
            frame,
            label,
            (wrist_x, wrist_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
