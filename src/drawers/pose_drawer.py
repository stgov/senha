from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision

from src.smoothing.bbox_smoother import BBoxSmoother


class PoseDrawer:
    def __init__(self, bbox_smoother: BBoxSmoother):
        self.bbox_smoother = bbox_smoother
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def draw(
        self,
        frame: np.ndarray,
        pose_result: vision.PoseLandmarkerResult,
        person_ids: Dict[int, str] = None,
        marked_persons: set = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Dibujar poses y retornar bboxes.

        Args:
            frame: Frame a anotar
            pose_result: Resultado de detección de poses
            person_ids: Mapeo de índice de pose a ID de REID
            marked_persons: Set de IDs de personas marcadas

        Returns:
            Tuple de (frame anotado, lista de bboxes)
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        bboxes = []

        if not pose_result.pose_landmarks:
            return annotated_frame, bboxes

        if person_ids is None:
            person_ids = {}
        if marked_persons is None:
            marked_persons = set()

        for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
            self._draw_skeleton(annotated_frame, pose_landmarks)
            bbox = self._draw_bbox(
                annotated_frame,
                pose_landmarks,
                idx,
                w,
                h,
                person_ids.get(idx),
                person_ids.get(idx) in marked_persons,
            )
            if bbox:
                bboxes.append(bbox)

        return annotated_frame, bboxes

    def _draw_skeleton(
        self, frame: np.ndarray, pose_landmarks: Sequence[landmark_pb2.NormalizedLandmark]
    ) -> None:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        for landmark in pose_landmarks:
            landmark_proto = pose_landmarks_proto.landmark.add()
            landmark_proto.x = landmark.x
            landmark_proto.y = landmark.y
            landmark_proto.z = landmark.z
            landmark_proto.visibility = landmark.visibility

        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks_proto,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    def _draw_bbox(
        self,
        frame: np.ndarray,
        pose_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
        idx: int,
        w: int,
        h: int,
        person_id: str = None,
        is_marked: bool = False,
    ) -> Tuple[int, int, int, int] | None:
        """
        Dibujar bounding box y retornar coordenadas.

        Returns:
            Tuple (x_min, y_min, x_max, y_max) o None si no hay landmarks visibles
        """
        x_coords = [int(lm.x * w) for lm in pose_landmarks if lm.visibility > 0.5]
        y_coords = [int(lm.y * h) for lm in pose_landmarks if lm.visibility > 0.5]

        if not x_coords or not y_coords:
            return None

        padding = 20
        x_min_raw = max(0, min(x_coords) - padding)
        y_min_raw = max(0, min(y_coords) - padding)
        x_max_raw = min(w, max(x_coords) + padding)
        y_max_raw = min(h, max(y_coords) + padding)

        raw_bbox = (x_min_raw, y_min_raw, x_max_raw, y_max_raw)
        x_min, y_min, x_max, y_max = self.bbox_smoother.smooth(idx, raw_bbox)

        # Color basado en estado de marcado
        color = (0, 255, 255) if is_marked else (0, 255, 0)  # Amarillo si marcado, verde si no
        thickness = 3 if is_marked else 2

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

        cx, cy = self.bbox_smoother.get_centroid_from_bbox((x_min, y_min, x_max, y_max))
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Mostrar ID de REID o índice temporal
        label = person_id if person_id else f"Person {idx + 1}"
        if is_marked:
            label += " [MARCADO]"

        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return (x_min, y_min, x_max, y_max)
