import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any

# Conexiones de la mano
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

class Annotator:
    """Se encarga de dibujar todas las anotaciones en el frame."""
    
    def __init__(self):
        # Colores (BGR)
        self.color_bbox = (0, 255, 0)       # Verde
        self.color_bbox_marked = (0, 0, 255) # Rojo
        self.color_text = (255, 255, 255)    # Blanco
        self.color_gesture_text = (0, 255, 255) # Amarillo
        self.color_hand_kp = (255, 255, 0)   # Cyan
        self.color_hand_conn = (200, 200, 200) # Gris claro
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_all(self, frame: np.ndarray, frame_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibuja todas las anotaciones para un frame.
        (Modificado para no dibujar pose landmarks).
        """
        annotated_frame = frame.copy()
        
        for data in frame_data:
            person_id = data["person_id"]
            bbox = data["bbox"]
            is_marked = data["is_marked"]
            gesture_name = data["gesture_name"]
            hand_landmarks = data["hand_landmarks"]
            roi_offset = data["roi_offset"]
            roi_dims = data["roi_dims"]

            color = self.color_bbox_marked if is_marked else self.color_bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # --- 1. Dibujar Bounding Box ---
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # --- 2. Dibujar Texto (ID y Gesto) ---
            label = f"ID: {person_id}"
            if is_marked and gesture_name == "Closed_Fist":
                label += " (GESTO: CLOSED)"
            
            # Fondo para el texto
            (text_w, text_h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            # Texto
            cv2.putText(annotated_frame, label, (x1, y1 - 5), self.font, 0.6, self.color_text, 2)
            
            # --- 3. Dibujar Landmarks de Manos ---
            if hand_landmarks:
                self._draw_hand_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    roi_offset,
                    roi_dims
                )

        return annotated_frame

    def _draw_hand_landmarks(self, frame, hands_landmarks_list, roi_offset, roi_dims):
        """Dibuja los landmarks para una o más manos."""
        offset_x, offset_y = roi_offset
        roi_w, roi_h = roi_dims
        
        for hand_landmarks in hands_landmarks_list:
            # Dibujar conexiones
            if HAND_CONNECTIONS:
                for conn in HAND_CONNECTIONS:
                    start_idx = conn[0]
                    end_idx = conn[1]
                    
                    if 0 <= start_idx < len(hand_landmarks) and 0 <= end_idx < len(hand_landmarks):
                        start_lm = hand_landmarks[start_idx]
                        end_lm = hand_landmarks[end_idx]
                        
                        start_pt = (int(start_lm.x * roi_w) + offset_x, int(start_lm.y * roi_h) + offset_y)
                        end_pt = (int(end_lm.x * roi_w) + offset_x, int(end_lm.y * roi_h) + offset_y)
                        
                        cv2.line(frame, start_pt, end_pt, self.color_hand_conn, 2)

            # Dibujar puntos
            for landmark in hand_landmarks:
                x = int(landmark.x * roi_w) + offset_x
                y = int(landmark.y * roi_h) + offset_y
                cv2.circle(frame, (x, y), 4, self.color_hand_kp, -1)

    def draw_fps(self, frame, fps):
        """Dibuja el contador de FPS."""
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), self.font, 1, (0, 255, 0), 2)

