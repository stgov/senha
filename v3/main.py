import cv2
import numpy as np
import time
from typing import Dict
import torch # Para detectar CUDA

from tracking.tracker import PersonTracker
from reid.person_reid import PersonReID
from detection.mediapipe_processor import MediaPipeProcessor
from anotation.annotator import Annotator
from smoothing.box_smoothing import BoundingBoxSmoother
from downloader.model_downloader import get_model_path

# --- Configuración ---
VIDEO_SOURCE = './media/video.mp4'
ENABLE_SMOOTHING = True # Activar/Desactivar suavizado de BBox
DEBUG_DRAW_HANDS = False # Iniciar con el dibujo de manos desactivado

# Modelos
GESTURE_MODEL_PATH = 'gesture_recognizer.task'

def main():
    print("Iniciando sistema...")
    
    # Detectar dispositivo (CUDA o CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo seleccionado: {device.upper()}")
    
    # 1. Inicializar componentes
    tracker = PersonTracker(device=device)
    reid = PersonReID()
    
    gesture_model_path = get_model_path(GESTURE_MODEL_PATH)
    mp_processor = MediaPipeProcessor(gesture_model_path, device=device)
    
    annotator = Annotator()
    smoother = BoundingBoxSmoother(smoothing_factor=0.3, dead_zone=10)
    
    # Estructuras de datos para estado
    person_states: Dict[str, bool] = {} # person_id -> is_marked
    
    # Variable de estado para depuración
    global DEBUG_DRAW_HANDS
    
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. Tracking (YOLO + ByteTrack) ---
        detections = tracker.track(rgb_frame)
        
        draw_data_list = []

        # --- 2. Re-ID y Procesamiento MediaPipe ---
        for detection_data in detections:
            tracker_id, (x1, y1, x2, y2) = detection_data
            
            if x1 >= x2 or y1 >= y2:
                continue

            # --- 2a. Re-Identificación ---
            person_id, is_new = reid.identify_person(rgb_frame, (x1, y1, x2, y2), tracker_id)
            
            if is_new:
                print(f"Nueva persona detectada: {person_id} (asignada a tracker {tracker_id})")
            
            if person_id not in person_states:
                person_states[person_id] = False

            # --- 2b. Procesamiento MediaPipe (Gestos) ---
            roi = rgb_frame[y1:y2, x1:x2]
            
            # mp_results es un objeto GestureRecognizerResult, NO un dict
            mp_results = mp_processor.process(roi)
            
            # --- 2c. Actualizar Estado (Marcado) ---
            # --- ¡CORRECCIÓN AQUÍ! ---
            # Accedemos a los atributos del objeto GestureRecognizerResult
            gesture_name = None
            if mp_results.gestures and mp_results.gestures[0]:
                gesture_name = mp_results.gestures[0][0].category_name

            if gesture_name == "Closed_Fist":
                if not person_states[person_id]:
                    print(f"Persona {person_id} marcada permanentemente.")
                person_states[person_id] = True
            
            is_marked = person_states[person_id]

            # --- 2d. Suavizado de BBox ---
            if ENABLE_SMOOTHING:
                smooth_box = smoother.smooth(person_id, (x1, y1, x2, y2))
            else:
                smooth_box = (x1, y1, x2, y2)

            # --- 3. Acumular datos para dibujar ---
            # --- ¡CORRECCIÓN AQUÍ! ---
            draw_data = {
                "person_id": person_id,
                "bbox": smooth_box,
                "is_marked": is_marked,
                "gesture_name": gesture_name, # Usamos la variable parseada
            }
            
            # Solo añadir datos de manos si existen
            # Accedemos al atributo .hand_landmarks
            if mp_results.hand_landmarks:
                draw_data["hand_landmarks"] = mp_results.hand_landmarks
                draw_data["roi_offset"] = (x1, y1)
                draw_data["roi_dims"] = (x2 - x1, y2 - y1)
            
            draw_data_list.append(draw_data)

        # --- 4. Limpieza de personas ausentes ---
        current_tracker_ids = {det[0] for det in detections}
        reid.cleanup_absent_persons(current_tracker_ids)

        # --- 5. Anotación (Dibujar todo) ---
        annotated_frame = annotator.draw_all(frame, draw_data_list, DEBUG_DRAW_HANDS)
        
        # --- 6. FPS y Controles ---
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            annotator.draw_fps(annotated_frame, fps)

        cv2.imshow("Sistema de Re-ID y Gestos", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Presionar ESC para salir
            break
        
        # --- NUEVOS CONTROLES DE DEPDEBUG ---
        elif key == ord('r'):
            print("\n[INFO] Reseteando el sistema...")
            tracker.reset()
            reid.reset()
            smoother.reset()
            person_states.clear()
            frame_count = 0
            start_time = time.time()
            print("[INFO] Sistema reseteado.\n")
            
        elif key == ord('d'):
            DEBUG_DRAW_HANDS = not DEBUG_DRAW_HANDS
            print(f"[INFO] Dibujo de manos: {'ON' if DEBUG_DRAW_HANDS else 'OFF'}")


    # --- Limpieza ---
    cap.release()
    cv2.destroyAllWindows()
    reid.shutdown()
    print("Sistema finalizado.")

if __name__ == "__main__":
    main()

