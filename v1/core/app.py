from __future__ import annotations

from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from v1.core.config import TrackerConfig
from v1.core.model_manager import ModelManager
from v1.core.person_reid import PersonReID
from v1.core.stats import FrameStats
from v1.drawers.hand_drawer import HandDrawer
from v1.drawers.pose_drawer import PoseDrawer
from v1.smoothing.bbox_smoother import BBoxSmoother


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
            min_confidence=config.min_confidence,
        )
        # Sistema de Re-Identificación
        self.person_reid = PersonReID(similarity_threshold=0.75, max_absent_frames=500)
        self.show_hands = True
        self.show_poses = True
        self.frame_index = 0
        # Mapeo de índice de pose a ID de REID
        self.pose_to_reid: Dict[int, str] = {}

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

        # Procesar Re-ID para cada persona detectada
        self._process_reid(frame, pose_result)

        # Detectar gesto Closed_Fist y marcar personas
        self._process_closed_fist_gesture(gesture_result, pose_result, frame)

        # Actualizar personas ausentes
        self.person_reid.update_absent_persons()

        display_frame = frame.copy()
        if self.show_poses:
            display_frame, bboxes = self.pose_drawer.draw(
                display_frame,
                pose_result,
                person_ids=self.pose_to_reid,
                marked_persons=self.person_reid.marked_ids,
            )
        if self.show_hands:
            display_frame = self.hand_drawer.draw(display_frame, gesture_result)
        stats = self._build_stats(gesture_result, pose_result)
        self._render_overlay(display_frame, stats)
        return display_frame, stats

    def _to_mp_image(self, frame: np.ndarray) -> mp.Image:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _process_reid(self, frame: np.ndarray, pose_result: vision.PoseLandmarkerResult) -> None:
        """
        Procesar Re-ID para cada persona detectada.
        Genera embeddings y asigna IDs persistentes.
        """
        self.pose_to_reid.clear()

        if not pose_result.pose_landmarks:
            return

        h, w = frame.shape[:2]

        for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
            # Calcular bounding box
            x_coords = [int(lm.x * w) for lm in pose_landmarks if lm.visibility > 0.5]
            y_coords = [int(lm.y * h) for lm in pose_landmarks if lm.visibility > 0.5]

            if not x_coords or not y_coords:
                continue

            padding = 20
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_max = min(h, max(y_coords) + padding)

            bbox = (x_min, y_min, x_max, y_max)

            # Buscar o crear persona con Re-ID
            person_id = self.person_reid.find_or_create_person(frame, bbox)
            self.pose_to_reid[idx] = person_id

    def _process_closed_fist_gesture(
        self,
        gesture_result: vision.GestureRecognizerResult,
        pose_result: vision.PoseLandmarkerResult,
        frame: np.ndarray,
    ) -> None:
        """
        Detectar gesto Closed_Fist y marcar la persona correspondiente permanentemente.
        """
        if not gesture_result.gestures or not gesture_result.hand_landmarks:
            return

        if not pose_result.pose_landmarks:
            return

        h, w = frame.shape[:2]

        # Procesar cada mano detectada
        for hand_idx, (gesture_list, hand_landmarks) in enumerate(
            zip(gesture_result.gestures, gesture_result.hand_landmarks)
        ):
            if not gesture_list:
                continue

            # Obtener gesto de mayor confianza
            top_gesture = gesture_list[0]

            # Verificar si es Closed_Fist con suficiente confianza
            if (
                top_gesture.category_name == "Closed_Fist"
                and top_gesture.score >= self.config.gesture_score_threshold
            ):

                # Encontrar persona más cercana a esta mano
                person_idx = self._find_closest_person_to_hand(
                    hand_landmarks, pose_result.pose_landmarks, w, h
                )

                if person_idx is not None and person_idx in self.pose_to_reid:
                    person_id = self.pose_to_reid[person_idx]
                    self.person_reid.mark_person(person_id)
                    print(f"Persona {person_id} marcada con Closed_Fist")

    def _find_closest_person_to_hand(
        self, hand_landmarks, pose_landmarks_list, w: int, h: int
    ) -> int | None:
        """
        Encontrar la persona más cercana a una mano detectada.
        Usa el centroide de la mano y lo compara con los torsos de las personas.
        """
        # Calcular centroide de la mano (promedio de todos los puntos)
        hand_x = np.mean([lm.x for lm in hand_landmarks]) * w
        hand_y = np.mean([lm.y for lm in hand_landmarks]) * h

        min_distance = float("inf")
        closest_person_idx = None

        for person_idx, pose_landmarks in enumerate(pose_landmarks_list):
            # Usar landmarks del torso superior (hombros y caderas)
            # Indices MediaPipe: 11-12 (hombros), 23-24 (caderas)
            torso_landmarks = [
                pose_landmarks[11],  # Hombro izquierdo
                pose_landmarks[12],  # Hombro derecho
                pose_landmarks[23],  # Cadera izquierda
                pose_landmarks[24],  # Cadera derecha
            ]

            # Filtrar landmarks visibles
            visible_torso = [lm for lm in torso_landmarks if lm.visibility > 0.5]

            if not visible_torso:
                continue

            # Calcular centroide del torso
            torso_x = np.mean([lm.x for lm in visible_torso]) * w
            torso_y = np.mean([lm.y for lm in visible_torso]) * h

            # Calcular distancia euclidiana
            distance = np.sqrt((hand_x - torso_x) ** 2 + (hand_y - torso_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_person_idx = person_idx

        # Solo retornar si la distancia es razonable (menos de 200 píxeles)
        if min_distance < 200:
            return closest_person_idx

        return None

    def _build_stats(
        self,
        gesture_result: vision.GestureRecognizerResult,
        pose_result: vision.PoseLandmarkerResult,
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
            show_poses=self.show_poses,
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
            2,
        )
        y += 30

        # Estadísticas de Re-ID
        reid_stats = self.person_reid.get_stats()
        cv2.putText(
            frame,
            f"Re-ID: {reid_stats['total_persons']} total | "
            f"{reid_stats['active_persons']} activas | "
            f"{reid_stats['marked_persons']} marcadas",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y += 25

        cv2.putText(
            frame,
            f"Modelo: {stats.pose_model.upper()} | Frame: {stats.frame_index}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
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
            1,
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
        print(
            f"👥 Detección: Hasta {self.config.num_poses} personas y {self.config.max_num_hands} manos"
        )
        print(
            f"🎯 Suavizado: Dead zone {self.config.smoothing_dead_zone}px, "
            f"Factor {self.config.smoothing_factor}"
        )
        print("📋 Controles: [H] manos, [P] poses, [Q] salir")
