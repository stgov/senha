import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch

class PersonTracker:
    def __init__(self, device='cpu'):
        """
        Inicializa el detector YOLO y el tracker ByteTrack.
        
        Args:
            device (str): 'cuda' o 'cpu'.
        """
        self.device = device
        self.model = YOLO("yolov8n.pt").to(device)
        self.reset() # Inicializar el tracker

    def reset(self):
        """Resetea el tracker a su estado inicial."""
        print("[Tracker] Reseteando tracker...")
        # frame_rate=30 es un valor común, ajustar si el video es muy lento/rápido
        self.tracker = sv.ByteTrack(frame_rate=30)

    def track(self, frame_rgb: np.ndarray) -> list[tuple[int, list]]:
        """
        Realiza la detección y el tracking en un frame.
        
        Args:
            frame_rgb: El frame en formato RGB.
            
        Returns:
            Una lista de tuplas (tracker_id, [x1, y1, x2, y2])
        """
        results = self.model.track(
            source=frame_rgb,
            persist=True,
            classes=[0], # Clase 0 es 'person' en COCO
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml"
        )[0]
        
        # Convertir resultados a supervision.Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Actualizar el tracker
        tracked_detections = self.tracker.update_with_detections(detections)
        
        output = []
        for det in tracked_detections:
            # det es (xyxy, mask, confidence, class_id, tracker_id)
            xyxy = det[0].astype(int)
            tracker_id = det[4]
            
            if tracker_id is None:
                continue
                
            output.append((int(tracker_id), [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]))
            
        return output

