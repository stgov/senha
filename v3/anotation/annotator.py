import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple, Optional

# Definir conexiones de la mano (constante)
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

class Annotator:
    def __init__(self):
        self.colors = {
            "bbox_normal": (0, 255, 0),  # Verde
            "bbox_marked": (0, 0, 255),  # Rojo
            "text_normal": (255, 255, 255), # Blanco
            "text_marked": (0, 0, 255),  # Rojo
            "hand_landmarks": (255, 0, 0), # Azul
            "hand_connections": (0, 255, 0) # Verde
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.debug_draw_hands = False # Estado de depuración

    def draw_person_box(self, 
                        frame: np.ndarray, 
                        bbox: Tuple[int, int, int, int], 
                        person_id: str, 
                        is_marked: bool, 
                        gesture_name: Optional[str]) -> np.ndarray:
        """Dibuja el bounding box y la etiqueta de la persona."""
        x1, y1, x2, y2 = bbox
        
        color = self.colors["bbox_marked"] if is_marked else self.colors["bbox_normal"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID: {person_id}"
        if is_marked:
            label += " [MARCADO]"
        
        # Solo mostrar el gesto si no es "None" y las manos están visibles (debug)
        if self.debug_draw_hands and gesture_name and gesture_name != "None":
            label += f" Gesto: {gesture_name}"

        (w, h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
        
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        cv2.putText(frame, label, (x1, y1 - 5), self.font, 0.6, self.colors["text_normal"], 2)
        
        return frame

    def draw_hand_landmarks(self, 
                            frame: np.ndarray, 
                            hand_landmarks: List[Any], # Lista de NormalizedLandmark
                            roi_offset: Tuple[int, int], 
                            roi_dims: Tuple[int, int]) -> np.ndarray:
        """Dibuja los landmarks y conexiones de UNA mano."""
        roi_w, roi_h = roi_dims
        offset_x, offset_y = roi_offset
        
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * roi_w + offset_x)
            y = int(landmark.y * roi_h + offset_y)
            points.append((x, y))
            
            cv2.circle(frame, (x, y), 3, self.colors["hand_landmarks"], -1)

        if HAND_CONNECTIONS:
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], self.colors["hand_connections"], 2)
        
        return frame

    def draw_all(self, 
                 frame: np.ndarray, 
                 frame_data: List[Dict[str, Any]], 
                 debug_draw_hands: bool = False) -> np.ndarray:
        """Dibuja todas las anotaciones para un frame."""
        annotated_frame = frame.copy()
        self.debug_draw_hands = debug_draw_hands # Actualizar estado de depuración
        
        for data in frame_data:
            bbox = data.get("bbox")
            person_id = data.get("person_id", "??")
            is_marked = data.get("is_marked", False)
            gesture_name = data.get("gesture_name")
            
            if bbox:
                annotated_frame = self.draw_person_box(
                    annotated_frame, bbox, person_id, is_marked, gesture_name
                )
            
            # --- MODIFICACIÓN: Dibujar Landmarks solo si el debug está activo ---
            if self.debug_draw_hands:
                hand_landmarks_list = data.get("hand_landmarks")
                
                if hand_landmarks_list:
                    roi_offset = data.get("roi_offset", (0, 0))
                    roi_dims = data.get("roi_dims", (frame.shape[1], frame.shape[0]))
                    
                    for hand_landmarks in hand_landmarks_list:
                        annotated_frame = self.draw_hand_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            roi_offset,
                            roi_dims
                        )
                
        return annotated_frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Dibuja el contador de FPS y estado de depuración."""
        fps_label = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_label, (10, 30), self.font, 1, (0, 255, 0), 2)
        
        # Mostrar estado de depuración
        debug_label = f"Debug Manos: {'ON' if self.debug_draw_hands else 'OFF'} [D]"
        cv2.putText(frame, debug_label, (10, 60), self.font, 0.6, (255, 255, 0), 2)
        
        reset_label = "Reset: [R]"
        cv2.putText(frame, reset_label, (10, 85), self.font, 0.6, (255, 255, 0), 2)
        
        return frame

