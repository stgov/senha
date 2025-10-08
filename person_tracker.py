import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Optional, Sequence, Set, Tuple

from scripts import ModelDownloader


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

        # Configuración de cámara
        self.CAMERA_INDEX = "media/video.mp4"
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480

        # Configuración de suavizado de bounding boxes
        self.BBOX_DEAD_ZONE = 15  # Píxeles de zona muerta
        self.BBOX_SMOOTHING_FACTOR = (
            0.3  # Factor de interpolación (0-1, menor = más suave)
        )
        self.previous_bboxes = {}  # Diccionario para almacenar bboxes previas por persona

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
            num_poses=self.NUM_POSES,  # Detectar hasta 6 personas
        )
        return vision.PoseLandmarker.create_from_options(options)

    def process_frame_parallel(self, frame):
        """Procesar frame con ambos modelos en paralelo"""
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear imagen de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Procesar con ambos modelos simultáneamente
        gesture_result = self.gesture_recognizer.recognize(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)

        return gesture_result, pose_result

    def draw_hand_landmarks(self, frame, gesture_result):
        """Dibujar landmarks de mano y gestos detectados"""
        annotated_frame = frame.copy()
        # Colores para hasta 12 manos
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (255, 128, 0),
            (0, 128, 255),
            (255, 0, 128),
            (128, 255, 0),
            (0, 255, 128),
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
                            wrist_x = int(
                                hand_landmarks[0].x * annotated_frame.shape[1]
                            )
                            wrist_y = int(
                                hand_landmarks[0].y * annotated_frame.shape[0]
                            )
                            label = (
                                f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                            )
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

    def smooth_bbox(self, person_idx, new_bbox):
        """
        Suavizar bounding box usando dead zone e interpolación

        Args:
            person_idx: Índice de la persona
            new_bbox: Nueva bbox (x_min, y_min, x_max, y_max)

        Returns:
            bbox suavizada (x_min, y_min, x_max, y_max)
        """
        # Si no hay bbox previa, usar la nueva directamente
        if person_idx not in self.previous_bboxes:
            self.previous_bboxes[person_idx] = new_bbox
            return new_bbox

        prev_bbox = self.previous_bboxes[person_idx]

        # Calcular distancia entre centroides
        distance = self.calculate_bbox_distance(prev_bbox, new_bbox)

        # Si está dentro de la dead zone, mantener bbox anterior
        if distance < self.BBOX_DEAD_ZONE:
            return prev_bbox

        # Si está fuera de la dead zone, interpolar suavemente
        prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_bbox
        new_x_min, new_y_min, new_x_max, new_y_max = new_bbox

        # Interpolación lineal
        alpha = self.BBOX_SMOOTHING_FACTOR
        smooth_x_min = int(prev_x_min + alpha * (new_x_min - prev_x_min))
        smooth_y_min = int(prev_y_min + alpha * (new_y_min - prev_y_min))
        smooth_x_max = int(prev_x_max + alpha * (new_x_max - prev_x_max))
        smooth_y_max = int(prev_y_max + alpha * (new_y_max - prev_y_max))

        smoothed_bbox = (smooth_x_min, smooth_y_min, smooth_x_max, smooth_y_max)

        # Actualizar bbox previa
        self.previous_bboxes[person_idx] = smoothed_bbox

        return smoothed_bbox

    def extract_pose_bboxes(
        self, pose_result, frame_shape: Tuple[int, int, int]
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """Obtener bounding boxes suavizadas para cada persona detectada."""
        h, w, _ = frame_shape
        pose_bboxes: List[Optional[Tuple[int, int, int, int]]] = []

        if not pose_result.pose_landmarks:
            return pose_bboxes

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
                pose_bboxes.append(None)
                continue

            padding = 20
            x_min_raw, x_max_raw = min(x_coords), max(x_coords)
            y_min_raw, y_max_raw = min(y_coords), max(y_coords)

            x_min_raw = max(0, x_min_raw - padding)
            y_min_raw = max(0, y_min_raw - padding)
            x_max_raw = min(w, x_max_raw + padding)
            y_max_raw = min(h, y_max_raw + padding)

            raw_bbox = (x_min_raw, y_min_raw, x_max_raw, y_max_raw)
            pose_bboxes.append(self.smooth_bbox(idx, raw_bbox))

        return pose_bboxes

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

    def find_highlight_indices(
        self,
        pose_bboxes: Sequence[Optional[Tuple[int, int, int, int]]],
        hand_bboxes: Sequence[Tuple[int, int, int, int]],
    ) -> Set[int]:
        """Encontrar índices de personas cuya bbox colisiona con un Closed_Fist."""
        highlight_indices: Set[int] = set()

        if not pose_bboxes or not hand_bboxes:
            return highlight_indices

        for pose_idx, pose_bbox in enumerate(pose_bboxes):
            if pose_bbox is None:
                continue

            for hand_bbox in hand_bboxes:
                if self.rectangles_intersect(pose_bbox, hand_bbox):
                    highlight_indices.add(pose_idx)
                    break

        return highlight_indices

    def draw_pose_landmarks(
        self,
        frame,
        pose_result,
        pose_bboxes: Sequence[Optional[Tuple[int, int, int, int]]],
        highlight_indices: Optional[Set[int]] = None,
    ):
        """Dibujar landmarks de pose para todas las personas detectadas."""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        highlight_indices = highlight_indices or set()

        if pose_result.pose_landmarks:
            # Dibujar todas las poses detectadas
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

                # Dibujar
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_landmarks_proto,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                bbox = pose_bboxes[idx] if idx < len(pose_bboxes) else None
                if bbox is None:
                    continue

                x_min, y_min, x_max, y_max = bbox
                color = (0, 0, 255) if idx in highlight_indices else (0, 255, 0)

                cv2.rectangle(
                    annotated_frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    color,
                    2,
                )

                cx, cy = self.calculate_bbox_centroid(x_min, y_min, x_max, y_max)
                cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

                label = f"Person {idx + 1}"
                if idx in highlight_indices:
                    label += " - Closed Fist"

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

    def run(self):
        """Ejecutar el sistema de detección y seguimiento"""
        cap = cv2.VideoCapture(self.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        print("🎬 Sistema de seguimiento iniciado")
        print(f"📦 Modelo de pose: {self.pose_model_name.upper()}")
        print(
            f"👥 Detección: Hasta {self.NUM_POSES} personas y {self.MAX_NUM_HANDS} manos"
        )
        print(
            f"🎯 Suavizado: Dead zone {self.BBOX_DEAD_ZONE}px, Factor {self.BBOX_SMOOTHING_FACTOR}"
        )
        print("📋 Instrucciones:")
        print("   - Ambos modelos (gestos + pose) corren en paralelo")
        print("   - Tracking de pose SIEMPRE activo")
        print("   - Detección de gestos SIEMPRE activa")
        print("   - Bounding boxes con suavizado inteligente")
        print("   - Presiona 'q' para salir")
        print("   - Presiona 'h' para ocultar/mostrar manos")
        print("   - Presiona 'p' para ocultar/mostrar poses")

        show_hands = True
        show_poses = True
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame")
                    break

                frame_count += 1

                # Procesar ambos modelos en paralelo
                gesture_result, pose_result = self.process_frame_parallel(frame)

                display_frame = frame.copy()

                pose_bboxes = self.extract_pose_bboxes(pose_result, frame.shape)
                hand_bboxes = self.get_hand_bounding_boxes(gesture_result, frame.shape)
                highlight_indices = self.find_highlight_indices(
                    pose_bboxes, hand_bboxes
                )

                # Dibujar poses (si está habilitado)
                if show_poses:
                    display_frame = self.draw_pose_landmarks(
                        display_frame, pose_result, pose_bboxes, highlight_indices
                    )

                # Dibujar manos y gestos (si está habilitado)
                if show_hands:
                    display_frame = self.draw_hand_landmarks(
                        display_frame, gesture_result
                    )

                # Estadísticas
                num_people = (
                    len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0
                )
                num_hands = (
                    len(gesture_result.hand_landmarks)
                    if gesture_result.hand_landmarks
                    else 0
                )
                num_gestures = (
                    sum(
                        1
                        for g_list in gesture_result.gestures
                        if g_list
                        for g in g_list
                        if g.score >= self.GESTURE_SCORE_THRESHOLD
                    )
                    if gesture_result.gestures
                    else 0
                )
                num_alerts = len(highlight_indices)

                # Información en pantalla
                y_offset = 30
                cv2.putText(
                    display_frame,
                    (
                        f"Personas: {num_people} | Manos: {num_hands} | "
                        f"Gestos: {num_gestures} | Alertas: {num_alerts}"
                    ),
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                y_offset += 30
                cv2.putText(
                    display_frame,
                    f"Modelo: {self.pose_model_name.upper()} | Frame: {frame_count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                y_offset += 25
                status_hands = "ON" if show_hands else "OFF"
                status_poses = "ON" if show_poses else "OFF"
                cv2.putText(
                    display_frame,
                    f"[H]ands: {status_hands} | [P]oses: {status_poses} | [Q]uit",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Mostrar frame
                cv2.imshow(
                    "Person Tracker - Parallel Models (Always Active)", display_frame
                )

                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("h"):
                    show_hands = not show_hands
                    print(f"👋 Visualización de manos: {'ON' if show_hands else 'OFF'}")
                elif key == ord("p"):
                    show_poses = not show_poses
                    print(f"🧍 Visualización de poses: {'ON' if show_poses else 'OFF'}")

        except KeyboardInterrupt:
            print("\nSistema interrumpido por el usuario")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Sistema cerrado correctamente")


def main():
    """Función principal"""
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
