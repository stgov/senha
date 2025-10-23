"""
Módulo de reconocimiento de gestos con MediaPipe.
"""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List

from ..core.config import TrackingConfig


class GestureRecognizer:
    """Reconocedor de gestos usando MediaPipe."""
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        
        # MediaPipe Gesture Recognizer
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerClass = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=config.gesture_model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=config.gesture_confidence
        )
        self.recognizer = GestureRecognizerClass.create_from_options(options)
    
    def recognize_gestures(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[List, str, float, Tuple[int, int]]]:
        """
        Reconocer gestos en una región del frame.
        
        Returns:
            (hand_landmarks, gesture_name, confidence, offset) o None
        """
        x1, y1, x2, y2 = bbox
        
        # Recortar región
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        # Convertir a RGB
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        
        try:
            gesture_result = self.recognizer.recognize(mp_image)
            
            if gesture_result.gestures and gesture_result.hand_landmarks:
                for gesture_list, hand_landmarks in zip(
                    gesture_result.gestures,
                    gesture_result.hand_landmarks
                ):
                    if not gesture_list:
                        continue
                    
                    top_gesture = gesture_list[0]
                    
                    return (
                        hand_landmarks,
                        top_gesture.category_name,
                        top_gesture.score,
                        (x1, y1)
                    )
        except Exception:
            pass
        
        return None
    
    def is_marking_gesture(self, gesture_name: str, confidence: float) -> bool:
        """Verificar si el gesto es Closed_Fist con suficiente confianza."""
        return (
            gesture_name == "Closed_Fist" and
            confidence >= self.config.gesture_confidence
        )
