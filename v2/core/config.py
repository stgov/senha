"""
Configuración del sistema de tracking con Re-ID.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingConfig:
    """Configuración para el sistema de tracking."""
    
    # Video
    video_source: int | str = 0  # 0 para webcam, o ruta de archivo
    
    # Re-ID
    similarity_threshold: float = 0.65  # Umbral para considerar misma persona
    max_absent_frames: int = 150  # ~5s a 30fps antes de olvidar persona
    optimize_reid: bool = True  # Solo ejecutar Re-ID cuando sea necesario
    
    # Gestos
    gesture_confidence: float = 0.5  # Confianza mínima para gestos
    gesture_model_path: str = './models/gesture_recognizer.task'
    
    # BBoxSmoother
    bbox_dead_zone: int = 15  # Píxeles de zona muerta para evitar jitter
    bbox_smoothing_factor: float = 0.3  # Factor de suavizado para interpolación
    
    # Kalman Filter para Embeddings
    use_kalman_embeddings: bool = True  # Usar Kalman para embeddings temporales
    kalman_embedding_process_noise: float = 1e-3  # Ruido de proceso
    kalman_embedding_measurement_noise: float = 1e-2  # Ruido de medición
    
    # Modelos YOLO
    yolo_model_path: str = './models/yolov8n.pt'
    reid_model_path: str = './models/osnet_x0_25_msmt17.pt'
    
    # Debug
    debug_mode: bool = True
    show_all_detections: bool = True
    
    # Visualización
    window_title: str = "Tracking con Re-ID Avanzado v2"
