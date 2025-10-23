"""
Sistema de Re-ID con tracking - Versión 2 (Modular)
Integra YOLO, BotSort, ChromaDB, MediaPipe y Kalman Filters
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import cv2

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from v2.core.config import TrackingConfig
from v2.core.embedding_extractor import EmbeddingExtractor
from v2.core.person_reid import PersonReID
from v2.gestures.recognizer import GestureRecognizer
from v2.smoothing.bbox_smoother import BBoxSmoother
from v2.tracking.tracker import PersonTracker
from v2.visualization.renderer import Visualizer


class ReIDSystem:
    """Sistema completo de Re-ID con tracking."""

    def __init__(
        self,
        video_source: str | int = 0,
        config: Optional[TrackingConfig] = None,
    ):
        # Configuración
        self.config = config or TrackingConfig()
        self.config.video_source = video_source

        # Video
        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None

        # Componentes
        self.tracker = PersonTracker(self.config)
        self.embedding_extractor = EmbeddingExtractor()
        self.person_reid = PersonReID(self.config, self.embedding_extractor)
        self.gesture_recognizer = GestureRecognizer(self.config)
        self.visualizer = Visualizer(self.config)
        self.bbox_smoother = BBoxSmoother(
            dead_zone=self.config.bbox_dead_zone, smoothing_factor=self.config.bbox_smoothing_factor
        )

        # Estado
        self.frame_index = 0
        self.debug_mode = False
        self.running = False

        # Mapeo tracker_id -> person_id para optimización Re-ID
        self.tracker_to_person: Dict[int, str] = {}

        # FPS
        self.fps = 0.0
        self.fps_samples = []

    def setup_video(self) -> bool:
        """Inicializar captura de video."""
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print(f"❌ Error: no se pudo abrir video '{self.video_source}'")
            return False

        # Configurar
        if isinstance(self.video_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS")
        print(f"🔍 Detector: {self.config.yolo_model_path}")
        print(
            f"🧠 Re-ID: umbral={self.config.similarity_threshold:.2f}, "
            f"optimizado={self.config.optimize_reid}"
        )
        print(f"👋 Gestos: {self.config.gesture_model_path}")
        print(
            f"📦 Kalman: process_noise={self.config.kalman_embedding_process_noise}, "
            f"measurement_noise={self.config.kalman_embedding_measurement_noise}"
        )

        return True

    def process_frame(self, frame):
        """Procesar un frame completo."""
        start_time = time.time()

        # 1. Tracking (YOLO + BotSort)
        detections, yolo_raw = self.tracker.detect_and_track(frame)

        if self.debug_mode and yolo_raw is not None:
            # Mostrar TODAS las detecciones YOLO
            debug_frame = self.visualizer.draw_debug_detections(frame, yolo_raw)
            cv2.imshow("Debug YOLO", debug_frame)

        # Preparar datos para visualización
        detections_normal = []
        detections_marked = []
        labels_normal = []
        labels_marked = []

        # 2. Procesar cada detección
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[0])
            tracker_id = int(detection[4]) if detection[4] is not None else None

            if tracker_id is None:
                continue

            # Suavizar bbox
            x1, y1, x2, y2 = self.bbox_smoother.smooth(tracker_id, (x1, y1, x2, y2))
            bbox = (x1, y1, x2, y2)

            # Re-ID: solo procesar si es nuevo tracker_id (optimización)
            if self.config.optimize_reid and tracker_id in self.tracker_to_person:
                person_id = self.tracker_to_person[tracker_id]
            else:
                # Calcular Re-ID
                person_id = self.person_reid.find_or_create_person(frame, bbox)

                if self.config.optimize_reid:
                    self.tracker_to_person[tracker_id] = person_id

            person_record = self.person_reid.persons[person_id]

            # 3. Reconocimiento de gestos en ROI
            gesture_result = self.gesture_recognizer.recognize_gestures(frame, bbox)

            # Verificar gesto de marcado y dibujar mano
            if gesture_result:
                hand_landmarks, gesture_name, confidence, _ = gesture_result

                # Dibujar keypoints de la mano
                self.visualizer.draw_hand_keypoints(
                    frame, hand_landmarks, bbox, person_record.marked
                )

                # Dibujar etiqueta del gesto
                self.visualizer.draw_gesture_label(frame, gesture_name, confidence, (x1, y2))

                # Marcar persona si detectamos Closed_Fist
                if self.gesture_recognizer.is_marking_gesture(gesture_name, confidence):
                    if not person_record.marked:
                        self.person_reid.mark_person(person_id)
                        print(f"🎯 Persona {person_id} marcada por gesto!")

            # Preparar para visualización
            det_data = (detection[0], detection[1], detection[2], detection[3], detection[4])
            label = f"ID: {person_id} (T:{tracker_id})"

            if person_record.marked:
                detections_marked.append(det_data)
                labels_marked.append(label)
            else:
                detections_normal.append(det_data)
                labels_normal.append(label)

        # 3.5. Actualizar personas ausentes
        self.person_reid.update_absent_persons()

        # 4. Visualización
        annotated = self.visualizer.draw_detections(
            frame, detections_normal, detections_marked, labels_normal, labels_marked
        )

        # Overlay de estadísticas
        stats = self.person_reid.get_stats()
        annotated = self.visualizer.draw_overlay(
            annotated, stats, self.frame_index, self.fps, len(detections)
        )

        # Calcular FPS
        elapsed = time.time() - start_time
        self.fps_samples.append(1.0 / elapsed if elapsed > 0 else 0.0)
        if len(self.fps_samples) > 30:
            self.fps_samples.pop(0)
        self.fps = sum(self.fps_samples) / len(self.fps_samples)

        return annotated

    def run(self):
        """Ejecutar el sistema."""
        if not self.setup_video():
            return

        self.running = True
        print("\n▶️  Sistema iniciado. Controles:")
        print("  [Q] Salir")
        print("  [R] Reset Re-ID")
        print("  [D] Toggle debug YOLO")
        print()

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Fin del video")
                    break

                self.frame_index += 1

                # Procesar frame
                annotated = self.process_frame(frame)

                # Mostrar
                cv2.imshow("Re-ID System v2", annotated)

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("🛑 Saliendo...")
                    break
                elif key == ord("r"):
                    print("🔄 Reseteando Re-ID...")
                    self.person_reid.reset()
                    self.tracker_to_person.clear()
                    self.bbox_smoother.reset()
                elif key == ord("d"):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    if not self.debug_mode:
                        cv2.destroyWindow("Debug YOLO")

        finally:
            self.cleanup()

    def cleanup(self):
        """Limpiar recursos."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Mostrar estadísticas finales
        stats = self.person_reid.get_stats()
        print("\n📊 Estadísticas finales:")
        print(f"  Frames procesados: {self.frame_index}")
        print(f"  Personas totales: {stats['total']}")
        print(f"  Personas activas: {stats['active']}")
        print(f"  Personas marcadas: {stats['marked']}")
        print(f"  FPS promedio: {self.fps:.1f}")


def main():
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description="Sistema Re-ID v2")
    parser.add_argument(
        "--source", type=str, default="0", help="Fuente de video (0=webcam, path=archivo)"
    )
    parser.add_argument(
        "--model", type=str, default="models/yolov8n.pt", help="Ruta al modelo YOLO"
    )
    parser.add_argument("--threshold", type=float, default=0.70, help="Umbral de similitud Re-ID")
    parser.add_argument("--no-optimize", action="store_true", help="Desactivar optimización Re-ID")
    parser.add_argument(
        "--conf", type=float, default=0.3, help="Confidence threshold para detección"
    )
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold para NMS")

    args = parser.parse_args()

    # Parsear source
    video_source = args.source
    if video_source.isdigit():
        video_source = int(video_source)

    # Configuraciones
    config = TrackingConfig(
        yolo_model_path=args.model,
        reid_model_path="models/osnet_x0_25_msmt17.pt",  # Asegúrate que este path es correcto
        similarity_threshold=args.threshold,
        optimize_reid=not args.no_optimize,
    )

    # Ejecutar sistema
    system = ReIDSystem(
        video_source=video_source,
        config=config,
    )

    system.run()


if __name__ == "__main__":
    main()
