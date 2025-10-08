from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from typing import Tuple

from src.core.config import TrackerConfig
from src.core.model_manager import ModelManager
from src.core.stats import FrameStats
from src.drawers.hand_drawer import HandDrawer
from src.drawers.pose_drawer import PoseDrawer
from src.smoothing.bbox_smoother import BBoxSmoother


class TrackerApp:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.bbox_smoother = BBoxSmoother(config.smoothing_dead_zone, config.smoothing_factor)
        self.pose_drawer = PoseDrawer(self.bbox_smoother)
        self.hand_drawer = HandDrawer(config.gesture_score_threshold)
        self.model_manager = ModelManager(
            pose_model=config.pose_model,
            num_poses=config.num_poses,
            max_num_hands=config.max_num_hands,
            min_confidence=config.min_confidence
        )
        self.show_hands = True
        self.show_poses = True
        self.frame_index = 0

    def run(self) -> None:
        capture = self._open_capture()
        if not capture.isOpened():
            raise RuntimeError("No se pudo abrir la fuente de video")
        self._print_startup_banner()
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                self.frame_index += 1
                display_frame, stats = self._process_frame(frame)
                cv2.imshow(self.config.window_title, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.config.camera_source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return capture

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FrameStats]:
        mp_image = self._to_mp_image(frame)
        gesture_result, pose_result = self.model_manager.process_frame(mp_image)
        display_frame = frame.copy()
        if self.show_poses:
            display_frame = self.pose_drawer.draw(display_frame, pose_result)
        if self.show_hands:
            display_frame = self.hand_drawer.draw(display_frame, gesture_result)
        stats = self._build_stats(gesture_result, pose_result)
        self._render_overlay(display_frame, stats)
        return display_frame, stats

    def _to_mp_image(self, frame: np.ndarray) -> mp.Image:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _build_stats(
        self,
        gesture_result: vision.GestureRecognizerResult,
        pose_result: vision.PoseLandmarkerResult
    ) -> FrameStats:
        people = len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0
        hands = len(gesture_result.hand_landmarks) if gesture_result.hand_landmarks else 0
        gestures = 0
        if gesture_result.gestures:
            for gesture_list in gesture_result.gestures:
                if not gesture_list:
                    continue
                top_gesture = gesture_list[0]
                if top_gesture.score >= self.config.gesture_score_threshold:
                    gestures += 1
        return FrameStats(
            frame_index=self.frame_index,
            people=people,
            hands=hands,
            gestures=gestures,
            pose_model=self.config.pose_model,
            show_hands=self.show_hands,
            show_poses=self.show_poses
        )

    def _render_overlay(self, frame: np.ndarray, stats: FrameStats) -> None:
        y = 30
        cv2.putText(
            frame,
            f"Personas: {stats.people} | Manos: {stats.hands} | Gestos: {stats.gestures}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        y += 30
        cv2.putText(
            frame,
            f"Modelo: {stats.pose_model.upper()} | Frame: {stats.frame_index}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y += 25
        hands_state = "ON" if stats.show_hands else "OFF"
        poses_state = "ON" if stats.show_poses else "OFF"
        cv2.putText(
            frame,
            f"[H]ands: {hands_state} | [P]oses: {poses_state} | [Q]uit",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return True
        if key == ord("h"):
            self.show_hands = not self.show_hands
            print(f"Visualización de manos: {'ON' if self.show_hands else 'OFF'}")
        if key == ord("p"):
            self.show_poses = not self.show_poses
            print(f"Visualización de poses: {'ON' if self.show_poses else 'OFF'}")
        return False

    def _print_startup_banner(self) -> None:
        print("🎬 Sistema de seguimiento iniciado")
        print(f"📦 Modelo de pose: {self.config.pose_model.upper()}")
        print(f"👥 Detección: Hasta {self.config.num_poses} personas y {self.config.max_num_hands} manos")
        print(
            f"🎯 Suavizado: Dead zone {self.config.smoothing_dead_zone}px, "
            f"Factor {self.config.smoothing_factor}"
        )
        print("📋 Controles: [H] manos, [P] poses, [Q] salir")
