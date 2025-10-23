"""
Gestor de modelos para el sistema v2.
Maneja la carga de YOLO, BotSort y MediaPipe Gesture Recognizer.
"""
from __future__ import annotations

import mediapipe as mp
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from typing import Optional
import sys
import os

# Agregar el directorio scripts al path para importar ModelDownloader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from model_downloader import ModelDownloader


class ModelManager:
    """Gestor centralizado de modelos de ML."""
    
    def __init__(
        self,
        yolo_model_path: str = "models/yolov8n.pt",
        reid_model_path: str = "models/osnet_x0_25_msmt17.pt",
        gesture_model_path: str = "models/gesture_recognizer.task",
        device: str = "cpu"
    ):
        """
        Inicializar el gestor de modelos.
        
        Args:
            yolo_model_path: Ruta al modelo YOLO
            reid_model_path: Ruta al modelo OSNet para Re-ID
            gesture_model_path: Ruta al modelo de gestos de MediaPipe
            device: 'cpu' o 'cuda'
        """
        self.yolo_model_path = Path(yolo_model_path)
        self.reid_model_path = Path(reid_model_path)
        self.gesture_model_path = Path(gesture_model_path)
        self.device = device
        
        # Verificar modelos
        self._ensure_models_available()
        
        # Inicializar modelos
        self.yolo_model: Optional[YOLO] = None
        self.tracker: Optional[BotSort] = None
        self.gesture_recognizer = None
    
    def _ensure_models_available(self) -> None:
        """Verificar y descargar modelos si no existen."""
        print("🔍 Verificando modelos...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Inicializar descargador
        downloader = ModelDownloader(models_dir="models")
        
        # Verificar que existan los modelos necesarios
        if not self.yolo_model_path.exists():
            print(f"⚠️ Modelo YOLO no encontrado en {self.yolo_model_path}")
        
        if not self.reid_model_path.exists():
            print(f"⚠️ Modelo ReID no encontrado en {self.reid_model_path}")
        
        if not self.gesture_model_path.exists():
            print(f"⚠️ Modelo de gestos no encontrado en {self.gesture_model_path}")
            print("📥 Descargando modelo de gestos...")
            # Aquí podrías agregar lógica de descarga si es necesario
    
    def load_yolo(self) -> YOLO:
        """Cargar modelo YOLO para detección de personas."""
        if self.yolo_model is None:
            print("  ⏳ Cargando YOLO...")
            self.yolo_model = YOLO(str(self.yolo_model_path))
            print("  ✅ YOLO cargado correctamente")
        return self.yolo_model
    
    def load_tracker(self) -> BotSort:
        """Cargar BotSort tracker (YOLO + Kalman + OSNet)."""
        if self.tracker is None:
            print("  ⏳ Cargando BotSort tracker...")
            try:
                self.tracker = BotSort(
                    model_weights=self.yolo_model_path,
                    device=self.device,
                    reid_weights=self.reid_model_path,
                    half=False
                )
                print("  ✅ BotSort cargado correctamente")
            except Exception as e:
                print(f"  ❌ Error cargando BotSort: {e}")
                print("  ℹ️ Usando solo YOLO sin tracker")
                self.tracker = None
        return self.tracker
    
    def load_gesture_recognizer(
        self,
        num_hands: int = 2,
        min_confidence: float = 0.5
    ):
        """
        Cargar MediaPipe Gesture Recognizer.
        
        Args:
            num_hands: Número máximo de manos a detectar
            min_confidence: Confianza mínima para detección
        """
        if self.gesture_recognizer is None:
            print("  ⏳ Cargando reconocedor de gestos...")
            
            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(self.gesture_model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_confidence
            )
            
            self.gesture_recognizer = GestureRecognizer.create_from_options(options)
            print("  ✅ Reconocedor de gestos cargado")
        
        return self.gesture_recognizer
    
    def load_all(self, num_hands: int = 2, gesture_confidence: float = 0.5):
        """
        Cargar todos los modelos necesarios.
        
        Args:
            num_hands: Número de manos para el reconocedor de gestos
            gesture_confidence: Confianza mínima para gestos
        
        Returns:
            tuple: (yolo_model, tracker, gesture_recognizer)
        """
        yolo = self.load_yolo()
        tracker = self.load_tracker()
        gesture = self.load_gesture_recognizer(num_hands, gesture_confidence)
        
        return yolo, tracker, gesture
