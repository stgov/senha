import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model_downloader import ModelDownloader


class PersonTracker:
    def __init__(self, pose_model="lite"):
        # Configuración del modelo de gestos
        self.GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
        
        # Configuración del modelo de pose
        if pose_model == "lite":
            self.POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
        elif pose_model == "full":
            self.POSE_MODEL_PATH = "models/pose_landmarker_full.task"
        else:
            raise ValueError("pose_model debe ser 'lite' o 'full'")
        
        self.pose_model_name = pose_model
        
        # Configuración de confianza
        self.MIN_HAND_DETECTION_CONFIDENCE = 0.5
        self.MIN_HAND_PRESENCE_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5
        self.MAX_NUM_HANDS = 2
        self.GESTURE_SCORE_THRESHOLD = 0.5
        
        self.MIN_POSE_DETECTION_CONFIDENCE = 0.5
        self.MIN_POSE_PRESENCE_CONFIDENCE = 0.5
        self.MIN_POSE_TRACKING_CONFIDENCE = 0.5
        
        # Configuración de cámara
        self.CAMERA_INDEX = 0
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        
        # Estado del sistema
        self.tracking_active = False
        self.target_person_position = None  # Posición de la mano que hizo el gesto
        self.target_person_id = None  # ID de la persona a seguir
        
        # Inicializar ambos modelos
        self.gesture_recognizer = self.create_gesture_recognizer()
        self.pose_landmarker = self.create_pose_landmarker()
        
        # Para dibujar
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
    
    def create_gesture_recognizer(self):
        """Crear el reconocedor de gestos"""
        base_options = python.BaseOptions(model_asset_path=self.GESTURE_MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            min_hand_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=self.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
            num_hands=self.MAX_NUM_HANDS,
        )
        return vision.GestureRecognizer.create_from_options(options)
    
    def create_pose_landmarker(self):
        """Crear el detector de pose"""
        base_options = python.BaseOptions(model_asset_path=self.POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=self.MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=self.MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.MIN_POSE_TRACKING_CONFIDENCE,
        )
        return vision.PoseLandmarker.create_from_options(options)
    
    def process_frame_parallel(self, frame):
        """Procesar frame con ambos modelos en paralelo"""
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear imagen de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Procesar con ambos modelos
        gesture_result = self.gesture_recognizer.recognize(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        
        return gesture_result, pose_result
    
    def detect_closed_fist(self, gesture_result):
        """Detectar si hay un gesto Closed_Fist y retornar la posición de la muñeca"""
        if gesture_result.gestures and gesture_result.hand_landmarks:
            for idx, gesture_list in enumerate(gesture_result.gestures):
                if gesture_list and idx < len(gesture_result.hand_landmarks):
                    top_gesture = gesture_list[0]
                    if (top_gesture.category_name == "Closed_Fist" and 
                        top_gesture.score >= self.GESTURE_SCORE_THRESHOLD):
                        # Obtener posición de la muñeca (landmark 0)
                        wrist = gesture_result.hand_landmarks[idx][0]
                        return True, top_gesture.score, (wrist.x, wrist.y)
        
        return False, 0.0, None
    
    def find_closest_pose(self, pose_result, target_position):
        """Encontrar la pose más cercana a la posición objetivo (muñeca que hizo el gesto)"""
        if not pose_result.pose_landmarks or target_position is None:
            return None
        
        min_distance = float('inf')
        closest_pose_idx = None
        
        for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
            # Usar las muñecas de la pose (landmarks 15 y 16)
            left_wrist = pose_landmarks[15]  # Left wrist
            right_wrist = pose_landmarks[16]  # Right wrist
            
            # Calcular distancia euclidiana a ambas muñecas
            dist_left = ((left_wrist.x - target_position[0])**2 + 
                        (left_wrist.y - target_position[1])**2)**0.5
            dist_right = ((right_wrist.x - target_position[0])**2 + 
                         (right_wrist.y - target_position[1])**2)**0.5
            
            # Tomar la distancia mínima
            min_dist_for_pose = min(dist_left, dist_right)
            
            if min_dist_for_pose < min_distance:
                min_distance = min_dist_for_pose
                closest_pose_idx = idx
        
        # Solo considerar si está suficientemente cerca (umbral de 0.15)
        if min_distance < 0.15:
            return closest_pose_idx
        
        return None
    
    def draw_hand_landmarks(self, frame, gesture_result):
        """Dibujar landmarks de mano y gestos"""
        annotated_frame = frame.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        if gesture_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(gesture_result.hand_landmarks):
                color = colors[idx % len(colors)]
                
                # Dibujar landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * annotated_frame.shape[1])
                    y = int(landmark.y * annotated_frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 5, color, -1)
                
                # Dibujar conexiones
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = (
                        int(hand_landmarks[start_idx].x * annotated_frame.shape[1]),
                        int(hand_landmarks[start_idx].y * annotated_frame.shape[0]),
                    )
                    end_point = (
                        int(hand_landmarks[end_idx].x * annotated_frame.shape[1]),
                        int(hand_landmarks[end_idx].y * annotated_frame.shape[0]),
                    )
                    cv2.line(annotated_frame, start_point, end_point, color, 2)
                
                # Mostrar gesto detectado
                if gesture_result.gestures and idx < len(gesture_result.gestures):
                    gesture_list = gesture_result.gestures[idx]
                    if gesture_list:
                        top_gesture = gesture_list[0]
                        if top_gesture.score >= self.GESTURE_SCORE_THRESHOLD:
                            wrist_x = int(hand_landmarks[0].x * annotated_frame.shape[1])
                            wrist_y = int(hand_landmarks[0].y * annotated_frame.shape[0])
                            label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (wrist_x, wrist_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )
        
        return annotated_frame
    
    def draw_pose_landmarks(self, frame, pose_result, target_pose_idx=None):
        """Dibujar landmarks de pose y bounding box solo para la persona objetivo"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        if pose_result.pose_landmarks:
            # Si hay un objetivo específico, solo dibujar esa pose
            poses_to_draw = [target_pose_idx] if target_pose_idx is not None else range(len(pose_result.pose_landmarks))
            
            for idx in poses_to_draw:
                if idx >= len(pose_result.pose_landmarks):
                    continue
                    
                pose_landmarks = pose_result.pose_landmarks[idx]
                # Convertir a formato compatible con mp_drawing
                landmarks_proto = mp.solutions.pose.PoseLandmark
                
                # Dibujar landmarks
                landmark_list = []
                for landmark in pose_landmarks:
                    landmark_list.append(landmark)
                
                # Usar drawing_utils de MediaPipe
                # Crear un landmark list compatible
                from mediapipe.framework.formats import landmark_pb2
                
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in pose_landmarks:
                    landmark_proto = pose_landmarks_proto.landmark.add()
                    landmark_proto.x = landmark.x
                    landmark_proto.y = landmark.y
                    landmark_proto.z = landmark.z
                    landmark_proto.visibility = landmark.visibility
                
                # Dibujar
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_landmarks_proto,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Calcular bounding box
                x_coords = [int(landmark.x * w) for landmark in pose_landmarks if landmark.visibility > 0.5]
                y_coords = [int(landmark.y * h) for landmark in pose_landmarks if landmark.visibility > 0.5]
                
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Agregar padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Dibujar bounding box
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        "TRACKING ACTIVE",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        
        return annotated_frame
    
    def run(self):
        """Ejecutar el sistema de detección y seguimiento"""
        cap = cv2.VideoCapture(self.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("🎬 Sistema de seguimiento de persona iniciado")
        print(f"📦 Modelo de pose: {self.pose_model_name.upper()}")
        print("📋 Instrucciones:")
        print("   - Ambos modelos funcionan en paralelo")
        print("   - Haz el gesto 'Closed_Fist' para activar el seguimiento")
        print("   - Presiona 'q' para salir")
        print("   - Presiona 'r' para resetear el seguimiento")
        print("   - Presiona 't' para alternar entre mostrar manos/pose")
        
        show_hands = True  # Alternar visualización
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame")
                    break
                
                # Procesar ambos modelos en paralelo
                gesture_result, pose_result = self.process_frame_parallel(frame)
                
                display_frame = frame.copy()
                
                # Modo 1: Detección de Closed_Fist
                if not self.tracking_active:
                    # Dibujar landmarks de mano
                    if show_hands:
                        display_frame = self.draw_hand_landmarks(display_frame, gesture_result)
                    
                    fist_detected, confidence, wrist_position = self.detect_closed_fist(gesture_result)
                    
                    if fist_detected:
                        # Encontrar la pose más cercana a la mano que hizo el gesto
                        target_idx = self.find_closest_pose(pose_result, wrist_position)
                        
                        if target_idx is not None:
                            self.tracking_active = True
                            self.target_person_position = wrist_position
                            self.target_person_id = target_idx
                            print(f"✊ Closed_Fist detectado ({confidence:.2f})")
                            print(f"🎯 Persona identificada - Iniciando seguimiento")
                        else:
                            print(f"⚠️ Closed_Fist detectado pero no se encontró pose cercana")
                    
                    # Mostrar estado
                    cv2.putText(
                        display_frame,
                        "Esperando gesto Closed_Fist...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                
                # Modo 2: Seguimiento activo
                else:
                    # Actualizar el ID de la persona objetivo (puede cambiar de índice entre frames)
                    if self.target_person_position is not None:
                        self.target_person_id = self.find_closest_pose(pose_result, self.target_person_position)
                    
                    # Dibujar pose tracking solo de la persona objetivo
                    if self.target_person_id is not None:
                        display_frame = self.draw_pose_landmarks(display_frame, pose_result, self.target_person_id)
                        
                        # Actualizar posición objetivo basándose en la pose actual
                        if pose_result.pose_landmarks and self.target_person_id < len(pose_result.pose_landmarks):
                            target_pose = pose_result.pose_landmarks[self.target_person_id]
                            # Actualizar con el promedio de las muñecas
                            left_wrist = target_pose[15]
                            right_wrist = target_pose[16]
                            self.target_person_position = (
                                (left_wrist.x + right_wrist.x) / 2,
                                (left_wrist.y + right_wrist.y) / 2
                            )
                    else:
                        # Si se pierde la persona, mostrar advertencia
                        cv2.putText(
                            display_frame,
                            "PERSONA PERDIDA - Presiona 'r' para resetear",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                    
                    # Opcionalmente también mostrar manos
                    if show_hands:
                        display_frame = self.draw_hand_landmarks(display_frame, gesture_result)
                    
                    # Mostrar indicador de seguimiento
                    cv2.putText(
                        display_frame,
                        f"MODO: SEGUIMIENTO ACTIVO ({self.pose_model_name.upper()})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        display_frame,
                        "Presiona 'r' para resetear | 't' para alternar vista",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                # Mostrar frame
                cv2.imshow("Person Tracker - Parallel Models", display_frame)
                
                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.tracking_active = False
                    self.target_person_position = None
                    self.target_person_id = None
                    print("🔄 Seguimiento resetado - Esperando nuevo gesto")
                elif key == ord("t"):
                    show_hands = not show_hands
                    status = "activada" if show_hands else "desactivada"
                    print(f"👋 Visualización de manos {status}")
        
        except KeyboardInterrupt:
            print("\nSistema interrumpido por el usuario")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Sistema cerrado correctamente")


def main():
    """Función principal"""
    import os
    import sys
    
    # Seleccionar modelo de pose
    pose_model = "lite"  # Por defecto lite
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["lite", "full"]:
            pose_model = sys.argv[1]
        else:
            print("Uso: python person_tracker.py [lite|full]")
            print("Usando modelo 'lite' por defecto")
    
    # Descargar modelos automáticamente si no existen
    try:
        downloader = ModelDownloader()
        downloader.download_required(pose_model)
    except Exception as e:
        print(f"Error descargando modelos: {e}")
        print("Verifica tu conexión a internet")
        return
    
    # Crear y ejecutar el tracker
    tracker = PersonTracker(pose_model=pose_model)
    tracker.run()


if __name__ == "__main__":
    main()
