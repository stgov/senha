"""
Módulo de tracking con BotSort (YOLO + Kalman + OSNet).
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from boxmot import BotSort
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Tuple

from ..core.config import TrackingConfig


class PersonTracker:
    """
    Tracker de personas usando BotSort.
    Combina YOLO para detección + Kalman interno + OSNet para Re-ID.
    """
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        
        # YOLO para detección
        print("  ⏳ Cargando YOLO...")
        self.yolo_model = YOLO(config.yolo_model_path)
        print("  ✅ YOLO cargado")
        
        # BotSort para tracking
        print("  ⏳ Cargando BotSort tracker...")
        try:
            self.tracker = BotSort(
                model_weights=Path(config.yolo_model_path),
                device='cpu',
                reid_weights=Path(config.reid_model_path),
                half=False
            )
            print("  ✅ BotSort cargado")
        except Exception as e:
            print(f"  ❌ Error cargando BotSort: {e}")
            print("  ℹ️ Usando solo YOLO sin tracker")
            self.tracker = None
    
    def detect_and_track(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[sv.Detections], Optional[np.ndarray]]:
        """
        Detectar y trackear personas en el frame.
        
        Returns:
            (detections, yolo_results): Detecciones filtradas de personas y resultados raw de YOLO
        """
        try:
            # 1. Obtener detecciones de YOLO
            yolo_results = self.yolo_model(frame, verbose=False)
            
            if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
                if self.config.debug_mode and np.random.rand() < 0.033:  # ~1/30 frames
                    print("⚠️ YOLO no detectó ningún objeto")
                return sv.Detections.empty(), None
            
            boxes = yolo_results[0].boxes
            
            if self.config.debug_mode and np.random.rand() < 0.033:
                total_detections = len(boxes)
                person_detections = sum(1 for box in boxes if box.cls[0].cpu().numpy() == 0)
                print(f"🔍 YOLO detectó: {total_detections} objetos ({person_detections} personas)")
            
            # 2. Convertir a formato [x1, y1, x2, y2, conf, class_id]
            yolo_detections = []
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                yolo_detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
            yolo_detections = np.array(yolo_detections)
            
            # 3. Pasar detecciones a BotSort para tracking
            if self.tracker is not None:
                results = self.tracker.update(yolo_detections, frame)
            else:
                # Sin tracker: usar YOLO directamente con IDs temporales
                results = []
                for i, det in enumerate(yolo_detections):
                    results.append([det[0], det[1], det[2], det[3], i, det[4], det[5]])
                results = np.array(results) if len(results) > 0 else None
            
            if results is None or len(results) == 0:
                if self.config.debug_mode and np.random.rand() < 0.033:
                    print("⚠️ BotSort no detectó nada")
                return sv.Detections.empty(), yolo_detections
            
            # 4. Convertir a Supervision Detections
            xyxy = results[:, :4]
            confidence = results[:, 5] if results.shape[1] > 5 else None
            class_id = results[:, 6].astype(int) if results.shape[1] > 6 else None
            tracker_id = results[:, 4].astype(int)
            
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id
            )
            
            # 5. Filtrar solo personas (clase 0 en COCO)
            if len(detections) > 0 and detections.class_id is not None:
                detections = detections[detections.class_id == 0]
            
            if self.config.debug_mode and np.random.rand() < 0.033:
                print(f"  Personas filtradas: {len(detections)}")
            
            return detections, yolo_detections
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ Error en tracking: {e}")
                import traceback
                traceback.print_exc()
            return sv.Detections.empty(), None
