import cv2
import numpy as np
import time
from typing import Dict

from tracking.tracker import PersonTracker
from reid.person_reid import PersonReID
from detection.mediapipe_processor import MediaPipeProcessor
from anotation.annotator import Annotator
from smoothing.box_smoothing import BoundingBoxSmoother
from downloader.model_downloader import get_model_path

# --- Configuración ---
VIDEO_SOURCE = 0 # 0 para webcam, o ruta/URL de video
ENABLE_SMOOTHING = True # Activar/Desactivar suavizado de BBox

# Modelos
YOLO_MODEL_PATH = 'yolov8n.pt'
GESTURE_MODEL_PATH = 'gesture_recognizer.task'

def main():
    print("Iniciando sistema...")
    
    # 1. Inicializar componentes
    tracker = PersonTracker(YOLO_MODEL_PATH)
    reid = PersonReID()
    
    # Asegurarse que el modelo de gestos esté descargado
    gesture_model_path = get_model_path(GESTURE_MODEL_PATH.split('/')[-1])
    mp_processor = MediaPipeProcessor(gesture_model_path)
    
    annotator = Annotator()
    smoother = BoundingBoxSmoother(smoothing_factor=0.3, dead_zone=10)
    
    # Estructuras de datos para estado
    person_states: Dict[str, bool] = {} # person_id -> is_marked
    
    print("Iniciando captura de video...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video {VIDEO_SOURCE}")
        return

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Fin del stream o frame vacío.")
            break

        # Convertir a RGB (requerido por YOLO y MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. Tracking (YOLO + ByteTrack) ---
        # tracked_boxes = [x1, y1, x2, y2, tracker_id]
        tracked_boxes = tracker.update(rgb_frame)
        
        # Preparar datos para el anotador
        frame_data = []

        # --- 2. Re-ID y Procesamiento MediaPipe ---
        for box in tracked_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            tracker_id = int(box[4])

            # Asegurar que el ROI sea válido
            if x1 >= x2 or y1 >= y2:
                continue

            # --- 2a. Re-Identificación (ChromaDB + Kalman de Embedding) ---
            person_id, is_new = reid.identify_person(rgb_frame, (x1, y1, x2, y2), tracker_id)
            
            # Inicializar estado si es nueva persona
            if person_id not in person_states:
                person_states[person_id] = False # No marcado

            # --- 2b. Procesamiento MediaPipe (Solo Gestos) ---
            roi = rgb_frame[y1:y2, x1:x2]
            mp_results = mp_processor.process(roi)
            
            gesture_name = mp_results.get("gesture_name")
            hand_landmarks = mp_results.get("hand_landmarks")

            # --- 2c. Actualizar Estado (Marcado Permanente) ---
            if gesture_name == "Closed_Fist":
                person_states[person_id] = True
            
            is_marked = person_states[person_id]

            # --- 2d. Suavizado de BBox ---
            if ENABLE_SMOOTHING:
                smooth_box = smoother.smooth(person_id, (x1, y1, x2, y2))
            else:
                smooth_box = (x1, y1, x2, y2)

            # --- 3. Acumular datos para dibujar ---
            frame_data.append({
                "person_id": person_id,
                "bbox": smooth_box,
                "is_marked": is_marked,
                "gesture_name": gesture_name,
                "hand_landmarks": hand_landmarks,
                "roi_offset": (x1, y1),
                "roi_dims": (x2 - x1, y2 - y1)
            })

        # --- 4. Anotación (Dibujar todo) ---
        annotated_frame = annotator.draw_all(frame, frame_data)
        
        # Calcular y mostrar FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        annotator.draw_fps(annotated_frame, fps)

        # Mostrar frame
        cv2.imshow("Sistema de Re-ID y Gestos", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27: # Presionar ESC para salir
            break

    # --- Limpieza ---
    cap.release()
    cv2.destroyAllWindows()
    reid.shutdown() # Importante para ChromaDB
    print("Sistema finalizado.")

if __name__ == "__main__":
    main()

