"""
Script para grabar video con sistema de detección Re-ID
Graba el output con todas las detecciones en media/prototipo.mp4
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import os

from scripts import ModelDownloader
from src.core.person_reid import PersonReID


class VideoRecorder:
    def __init__(self, pose_model="lite", input_source=0, output_path="media/prototipo.mp4"):
        """
        Args:
            pose_model: 'lite' o 'full'
            input_source: índice de cámara (int) o ruta a video (str)
            output_path: ruta donde guardar el video grabado
        """
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
        self.input_source = input_source
        self.output_path = output_path

        # Configuración de confianza para gestos
        self.MIN_HAND_DETECTION_CONFIDENCE = 0.1
        self.MIN_HAND_PRESENCE_CONFIDENCE = 0.3
        self.MIN_TRACKING_CONFIDENCE = 0.3
        self.MAX_NUM_HANDS = 4
        self.GESTURE_SCORE_THRESHOLD = 0.1

        # Configuración de confianza para pose
        self.MIN_POSE_DETECTION_CONFIDENCE = 0.5
        self.MIN_POSE_PRESENCE_CONFIDENCE = 0.5
        self.MIN_POSE_TRACKING_CONFIDENCE = 0.5
        self.NUM_POSES = 2

        # Configuración de video
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.FPS = 30

        # Configuración de suavizado de bounding boxes
        self.BBOX_DEAD_ZONE = 15
        self.BBOX_SMOOTHING_FACTOR = 0.3
        self.previous_bboxes = {}

        # Sistema de Re-Identificación (Re-ID)
        self.reid_system = PersonReID(
            similarity_threshold=0.75,
            max_absent_frames=300,  # ~10 segundos a 30fps
        )
        
        # Mapeo de índices de frame actual a IDs persistentes
        self.frame_idx_to_person_id: Dict[int, str] = {}

        # Inicializar ambos modelos
        self.gesture_recognizer = self.create_gesture_recognizer()
        self.pose_landmarker = self.create_pose_landmarker()

        # Para dibujar
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else ".", exist_ok=True)

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
            num_poses=self.NUM_POSES,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def process_frame_parallel(self, frame):
        """Procesar frame con ambos modelos en paralelo"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        gesture_result = self.gesture_recognizer.recognize(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        
        return gesture_result, pose_result

    def calculate_bbox_centroid(self, x_min, y_min, x_max, y_max):
        """Calcular el centroide de un bounding box"""
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        return cx, cy

    def calculate_bbox_distance(self, bbox1, bbox2):
        """Calcular distancia entre centroides de dos bounding boxes"""
        cx1, cy1 = self.calculate_bbox_centroid(*bbox1)
        cx2, cy2 = self.calculate_bbox_centroid(*bbox2)
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def smooth_bbox(self, person_id, new_bbox):
        """Suavizar bounding box usando dead zone e interpolación"""
        if person_id not in self.previous_bboxes:
            self.previous_bboxes[person_id] = new_bbox
            return new_bbox

        prev_bbox = self.previous_bboxes[person_id]
        distance = self.calculate_bbox_distance(prev_bbox, new_bbox)

        if distance < self.BBOX_DEAD_ZONE:
            return prev_bbox

        prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_bbox
        new_x_min, new_y_min, new_x_max, new_y_max = new_bbox

        alpha = self.BBOX_SMOOTHING_FACTOR
        smooth_x_min = int(prev_x_min + alpha * (new_x_min - prev_x_min))
        smooth_y_min = int(prev_y_min + alpha * (new_y_min - prev_y_min))
        smooth_x_max = int(prev_x_max + alpha * (new_x_max - prev_x_max))
        smooth_y_max = int(prev_y_max + alpha * (new_y_max - prev_y_max))

        smoothed_bbox = (smooth_x_min, smooth_y_min, smooth_x_max, smooth_y_max)
        self.previous_bboxes[person_id] = smoothed_bbox

        return smoothed_bbox

    def extract_pose_bboxes_and_crops(
        self, frame, pose_result
    ) -> List[Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]]:
        """Obtener bounding boxes y crops de imagen para cada persona detectada."""
        h, w, _ = frame.shape
        results: List[Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]] = []

        if not pose_result.pose_landmarks:
            return results

        for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
            x_coords = [
                int(landmark.x * w)
                for landmark in pose_landmarks
                if landmark.visibility > 0.5
            ]
            y_coords = [
                int(landmark.y * h)
                for landmark in pose_landmarks
                if landmark.visibility > 0.5
            ]

            if not x_coords or not y_coords:
                results.append(None)
                continue

            padding = 20
            x_min_raw = max(0, min(x_coords) - padding)
            x_max_raw = min(w, max(x_coords) + padding)
            y_min_raw = max(0, min(y_coords) - padding)
            y_max_raw = min(h, max(y_coords) + padding)

            raw_bbox = (x_min_raw, y_min_raw, x_max_raw, y_max_raw)
            
            # Extraer crop de la persona
            crop = frame[y_min_raw:y_max_raw, x_min_raw:x_max_raw]
            
            if crop.size == 0:
                results.append(None)
                continue
                
            results.append((raw_bbox, crop))

        return results

    def get_hand_bounding_boxes(
        self, gesture_result, frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Obtener bounding boxes de manos con gesto Closed_Fist"""
        h, w, _ = frame_shape
        hand_bboxes: List[Tuple[int, int, int, int]] = []

        if not gesture_result.hand_landmarks or not gesture_result.gestures:
            return hand_bboxes

        for idx, hand_landmarks in enumerate(gesture_result.hand_landmarks):
            if idx >= len(gesture_result.gestures):
                continue

            gesture_list = gesture_result.gestures[idx]
            if not gesture_list:
                continue

            top_gesture = gesture_list[0]
            if (
                top_gesture.category_name != "Closed_Fist"
                or top_gesture.score < self.GESTURE_SCORE_THRESHOLD
            ):
                continue

            x_coords = [int(landmark.x * w) for landmark in hand_landmarks]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks]

            if not x_coords or not y_coords:
                continue

            padding = 10
            x_min = max(0, min(x_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_min = max(0, min(y_coords) - padding)
            y_max = min(h, max(y_coords) + padding)

            hand_bboxes.append((x_min, y_min, x_max, y_max))

        return hand_bboxes

    @staticmethod
    def rectangles_intersect(
        rect_a: Tuple[int, int, int, int], rect_b: Tuple[int, int, int, int]
    ) -> bool:
        """Determinar si dos rectángulos se intersectan."""
        ax_min, ay_min, ax_max, ay_max = rect_a
        bx_min, by_min, bx_max, by_max = rect_b

        return not (
            ax_max < bx_min
            or ax_min > bx_max
            or ay_max < by_min
            or ay_min > by_max
        )

    def draw_hand_landmarks(self, frame, gesture_result):
        """Dibujar landmarks de mano y gestos detectados"""
        annotated_frame = frame.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
            (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        ]

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
                                annotated_frame, label, (wrist_x, wrist_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            )

        return annotated_frame

    def draw_pose_landmarks(
        self,
        frame,
        pose_result,
        person_ids: List[Optional[str]],
        marked_persons: Set[str],
    ):
        """Dibujar landmarks de pose con IDs persistentes y estado de marca."""
        annotated_frame = frame.copy()

        if pose_result.pose_landmarks:
            for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
                # Convertir a formato compatible con mp_drawing
                from mediapipe.framework.formats import landmark_pb2

                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in pose_landmarks:
                    landmark_proto = pose_landmarks_proto.landmark.add()
                    landmark_proto.x = landmark.x
                    landmark_proto.y = landmark.y
                    landmark_proto.z = landmark.z
                    landmark_proto.visibility = landmark.visibility

                # Dibujar skeleton
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_landmarks_proto,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                person_id = person_ids[idx] if idx < len(person_ids) else None
                if person_id is None:
                    continue

                # Obtener bbox suavizada usando el ID persistente
                if person_id in self.previous_bboxes:
                    bbox = self.previous_bboxes[person_id]
                    x_min, y_min, x_max, y_max = bbox

                    # Color según si está marcada
                    is_marked = person_id in marked_persons
                    color = (0, 0, 255) if is_marked else (0, 255, 0)

                    # Dibujar bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        2,
                    )

                    # Dibujar centroide
                    cx, cy = self.calculate_bbox_centroid(x_min, y_min, x_max, y_max)
                    cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

                    # Etiqueta con ID persistente
                    label = f"ID: {person_id}"
                    if is_marked:
                        label += " [MARKED]"

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

        return annotated_frame

    def record(self, max_frames=None):
        """
        Ejecutar el sistema de detección y grabar video.
        
        Args:
            max_frames: número máximo de frames a grabar (None = ilimitado hasta 'q')
        """
        cap = cv2.VideoCapture(self.input_source)
        
        # Configurar resolución si es cámara
        if isinstance(self.input_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.FPS)

        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir la fuente de video: {self.input_source}")
            return

        # Obtener propiedades del video de entrada
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.FPS

        # Configurar writer de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            actual_fps,
            (actual_width, actual_height)
        )

        if not out.isOpened():
            print(f"❌ Error: No se pudo crear el archivo de salida: {self.output_path}")
            cap.release()
            return

        print("🎬 Sistema de grabación iniciado")
        print(f"📦 Modelo de pose: {self.pose_model_name.upper()}")
        print(f"📹 Entrada: {self.input_source}")
        print(f"💾 Salida: {self.output_path}")
        print(f"📐 Resolución: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        print(f"👥 Re-ID: Threshold {self.reid_system.similarity_threshold}")
        print("📋 Instrucciones:")
        print("   - Closed_Fist marca persona permanentemente (rojo)")
        print("   - Re-ID mantiene identidad al salir/entrar del frame")
        print("   - Presiona 'q' para detener y guardar")
        if max_frames:
            print(f"   - Máximo {max_frames} frames")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✓ Fin del video o error de lectura")
                    break

                if max_frames and frame_count >= max_frames:
                    print(f"✓ Alcanzado límite de {max_frames} frames")
                    break

                frame_count += 1

                # Procesar frame con ambos modelos
                gesture_result, pose_result = self.process_frame_parallel(frame)

                # Extraer bboxes y crops para Re-ID
                pose_data = self.extract_pose_bboxes_and_crops(frame, pose_result)

                # Re-identificar personas
                person_ids: List[Optional[str]] = []
                self.frame_idx_to_person_id.clear()

                for idx, data in enumerate(pose_data):
                    if data is None:
                        person_ids.append(None)
                        continue

                    raw_bbox, crop = data
                    person_id = self.reid_system.identify_person(crop, raw_bbox)
                    person_ids.append(person_id)
                    self.frame_idx_to_person_id[idx] = person_id

                    # Aplicar suavizado con ID persistente
                    smoothed_bbox = self.smooth_bbox(person_id, raw_bbox)
                    self.previous_bboxes[person_id] = smoothed_bbox

                # Detectar colisiones Closed_Fist y marcar personas
                hand_bboxes = self.get_hand_bounding_boxes(gesture_result, frame.shape)

                for idx, person_id in enumerate(person_ids):
                    if person_id is None:
                        continue

                    person_bbox = self.previous_bboxes.get(person_id)
                    if person_bbox is None:
                        continue

                    # Verificar colisión con algún Closed_Fist
                    for hand_bbox in hand_bboxes:
                        if self.rectangles_intersect(person_bbox, hand_bbox):
                            self.reid_system.mark_person(person_id)
                            break

                # Actualizar frame counter en Re-ID
                self.reid_system.update_frame_counter()

                # Obtener personas marcadas
                marked_persons = self.reid_system.get_marked_persons()

                # Dibujar detecciones
                display_frame = self.draw_pose_landmarks(
                    frame, pose_result, person_ids, marked_persons
                )
                display_frame = self.draw_hand_landmarks(display_frame, gesture_result)

                # Estadísticas en pantalla
                num_people = len([pid for pid in person_ids if pid is not None])
                num_hands = (
                    len(gesture_result.hand_landmarks)
                    if gesture_result.hand_landmarks
                    else 0
                )
                num_marked = len(marked_persons)
                total_known = len(self.reid_system.get_all_person_ids())

                y_offset = 30
                cv2.putText(
                    display_frame,
                    f"Personas: {num_people} | Total conocidas: {total_known} | Marcadas: {num_marked}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                y_offset += 30
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count} | Manos: {num_hands}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                y_offset += 25
                cv2.putText(
                    display_frame,
                    "REC | [Q]uit",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                # Escribir frame al video
                out.write(display_frame)

                # Mostrar preview
                cv2.imshow("Video Recorder - Re-ID System", display_frame)

                # Verificar tecla de salida
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✋ Grabación detenida por el usuario")
                    break

                # Mostrar progreso cada 30 frames
                if frame_count % 30 == 0:
                    print(f"📹 Grabados {frame_count} frames... (Personas: {num_people}, Marcadas: {num_marked})")

        except KeyboardInterrupt:
            print("\n✋ Grabación interrumpida por el usuario")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Video guardado exitosamente: {self.output_path}")
            print(f"📊 Total frames grabados: {frame_count}")
            print(f"👥 Total personas identificadas: {len(self.reid_system.get_all_person_ids())}")
            print(f"🔴 Total personas marcadas: {len(marked_persons)}")


def main():
    """Función principal"""
    import sys

    # Argumentos: pose_model, input_source, output_path
    pose_model = "lite"
    input_source = 0  # Cámara por defecto
    output_path = "media/prototipo.mp4"
    max_frames = None

    if len(sys.argv) > 1:
        if sys.argv[1] in ["lite", "full"]:
            pose_model = sys.argv[1]
        else:
            # Asumir que es la fuente de entrada
            try:
                input_source = int(sys.argv[1])
            except ValueError:
                input_source = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if len(sys.argv) > 3:
        try:
            max_frames = int(sys.argv[3])
        except ValueError:
            pass

    print("Configuración:")
    print(f"   Modelo: {pose_model}")
    print(f"   Entrada: {input_source}")
    print(f"   Salida: {output_path}")
    if max_frames:
        print(f"   Max frames: {max_frames}")

    # Descargar modelos automáticamente si no existen
    try:
        downloader = ModelDownloader()
        downloader.download_required(pose_model)
    except Exception as e:
        print(f"❌ Error descargando modelos: {e}")
        print("Verifica tu conexión a internet")
        return

    # Crear y ejecutar el grabador
    recorder = VideoRecorder(
        pose_model=pose_model,
        input_source=input_source,
        output_path=output_path
    )
    recorder.record(max_frames=max_frames)


if __name__ == "__main__":
    main()
