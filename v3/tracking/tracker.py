from ultralytics import YOLO
import numpy as np
import supervision as sv

class PersonTracker:
    """
    Combina YOLOv8 para detección y ByteTrack para tracking.
    
    Se enfoca solo en la clase 'persona' (clase 0 en COCO).
    """
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("YOLO model loaded.")
        
        # Inicializar ByteTrack
        # NOTA: Se eliminó 'track_thresh=0.25' para compatibilidad
        # con versiones de 'supervision' que no aceptan ese argumento.
        self.tracker = sv.ByteTrack(
            frame_rate=30 # Ajusta si tu video es más lento/rápido
        )
        # Clase de COCO para 'persona' es 0
        self.target_class_id = 0

    def update(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Detecta y trackea personas en un frame.

        Args:
            frame_rgb: El frame de video en formato RGB.

        Returns:
            Un array de numpy con [x1, y1, x2, y2, tracker_id] para cada persona.
        """
        # 1. Detección con YOLO
        # verbose=False para silenciar logs
        results = self.model(frame_rgb, classes=[self.target_class_id], verbose=False)
        
        # 2. Convertir a formato Supervision Detections
        # results[0].boxes.cpu().numpy() contiene todos los datos
        detections = sv.Detections.from_ultralytics(results[0])

        # 3. Actualizar el Tracker
        # El tracker (ByteTrack) maneja la lógica de movimiento y asigna IDs
        tracked_detections = self.tracker.update_with_detections(detections)

        # 4. Formatear salida
        # Queremos [x1, y1, x2, y2, tracker_id]
        output = []
        for det in tracked_detections:
            # det es (xyxy, mask, confidence, class_id, tracker_id)
            xyxy = det[0]
            tracker_id = det[4]
            
            if tracker_id is not None:
                output.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], tracker_id])

        return np.array(output)

