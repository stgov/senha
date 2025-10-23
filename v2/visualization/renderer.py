"""
Módulo de visualización para el sistema de tracking.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional

from ..core.person_reid import PersonReID, PersonRecord
from ..core.config import TrackingConfig


class Visualizer:
    """Clase para renderizar visualizaciones del sistema de tracking."""
    
    # Conexiones de la mano para MediaPipe
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
        (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
        (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
        (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
        (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
        (5, 9), (9, 13), (13, 17)  # Palma
    ]
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        
        # Anotadores de Supervision
        self.box_annotator_green = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
        self.box_annotator_red = sv.BoxAnnotator(thickness=4, color=sv.Color.RED)
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_scale=0.6,
            text_thickness=2,
        )
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections_normal: List,
        detections_marked: List,
        labels_normal: List[str],
        labels_marked: List[str],
    ) -> np.ndarray:
        """Dibujar detecciones normales y marcadas."""
        annotated = frame.copy()
        
        # Personas normales (verde)
        if len(detections_normal) > 0:
            detections_obj = sv.Detections(
                xyxy=np.array([d[0] for d in detections_normal]),
                confidence=np.array([d[1] for d in detections_normal]) if detections_normal[0][1] is not None else None,
                class_id=np.array([d[2] for d in detections_normal], dtype=np.int32) if detections_normal[0][2] is not None else None,
                tracker_id=np.array([int(d[4]) for d in detections_normal], dtype=np.int32)
            )
            annotated = self.box_annotator_green.annotate(annotated, detections_obj)
            annotated = self.label_annotator.annotate(annotated, detections_obj, labels_normal)
        
        # Personas marcadas (rojo)
        if len(detections_marked) > 0:
            detections_obj = sv.Detections(
                xyxy=np.array([d[0] for d in detections_marked]),
                confidence=np.array([d[1] for d in detections_marked]) if detections_marked[0][1] is not None else None,
                class_id=np.array([d[2] for d in detections_marked], dtype=np.int32) if detections_marked[0][2] is not None else None,
                tracker_id=np.array([int(d[4]) for d in detections_marked], dtype=np.int32)
            )
            annotated = self.box_annotator_red.annotate(annotated, detections_obj)
            annotated = self.label_annotator.annotate(annotated, detections_obj, labels_marked)
        
        return annotated
    
    def draw_hand_keypoints(
        self,
        frame: np.ndarray,
        hand_landmarks,
        bbox: Tuple[int, int, int, int],
        is_marked: bool,
    ) -> np.ndarray:
        """Dibujar keypoints de la mano."""
        x_offset, y_offset = bbox[:2]
        crop_w = bbox[2] - bbox[0]
        crop_h = bbox[3] - bbox[1]
        
        line_color = (0, 0, 255) if is_marked else (255, 200, 0)
        
        # Dibujar conexiones
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_x = int(start.x * crop_w + x_offset)
            start_y = int(start.y * crop_h + y_offset)
            end_x = int(end.x * crop_w + x_offset)
            end_y = int(end.y * crop_h + y_offset)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, 2)
        
        # Dibujar keypoints
        for landmark in hand_landmarks:
            x = int(landmark.x * crop_w + x_offset)
            y = int(landmark.y * crop_h + y_offset)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 3, line_color, -1)
        
        return frame
    
    def draw_gesture_label(
        self,
        frame: np.ndarray,
        gesture_name: str,
        confidence: float,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Dibujar etiqueta del gesto detectado."""
        text = f"{gesture_name} ({confidence:.2f})"
        text_x, text_y = position
        text_y -= 10
        
        # Fondo para el texto
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, 
                     (text_x, text_y - text_h - 5), 
                     (text_x + text_w + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        stats: Dict[str, int],
        frame_index: int,
        fps: float,
        num_detections: int,
    ) -> np.ndarray:
        """Dibujar overlay de estadísticas."""
        y = 30
        
        # Línea 1: Detecciones actuales
        cv2.putText(
            frame,
            f"Detecciones: {num_detections} | FPS: {fps:.1f} | Frame: {frame_index}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        y += 30
        # Línea 2: Re-ID stats
        cv2.putText(
            frame,
            f"Re-ID: {stats['total']} total | {stats['active']} activas | {stats['marked']} marcadas",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        y += 30
        # Línea 3: Configuración
        reid_mode = "Optimizado (entrada/salida)" if self.config.optimize_reid else "Completo (cada frame)"
        cv2.putText(
            frame,
            f"Umbral: {self.config.similarity_threshold:.2f} | Re-ID: {reid_mode}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        y += 25
        # Línea 4: Controles
        cv2.putText(
            frame,
            "[Q] Salir | [R] Reset | [D] Debug",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return frame
    
    def draw_debug_detections(
        self,
        frame: np.ndarray,
        yolo_detections: np.ndarray,
    ) -> np.ndarray:
        """Dibujar TODAS las detecciones de YOLO para debug."""
        debug_frame = frame.copy()
        
        # COCO class names (abreviado)
        coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant']
        
        for detection in yolo_detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            class_id = int(detection[5]) if len(detection) > 5 else -1
            conf = detection[4] if len(detection) > 4 else 0.0
            
            class_name = coco_names[class_id] if class_id < len(coco_names) else f"clase_{class_id}"
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_frame, f"{class_name} {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return debug_frame
