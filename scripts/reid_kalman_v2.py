"""
Sistema de Tracking con Re-Identificación Avanzado.

Características:
- BotSort (YOLO + Kalman interno + OSNet)
- ChromaDB para búsqueda vectorial
- Embeddings mejorados (HSV + HOG + textura)
- Filtro de Kalman adicional para suavizado
- Sistema de marcado permanente con Closed_Fist
"""

# Smoothing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# ChromaDB y procesamiento
import chromadb
import cv2
import mediapipe as mp
import numpy as np
import supervision as sv
from boxmot import BotSort
from chromadb.config import Settings
from skimage.feature import hog

# YOLO
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.smoothing.bbox_smoother import BBoxSmoother

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

VIDEO_SOURCE = 0  # 0 para webcam, o ruta de archivo
SIMILARITY_THRESHOLD = 0.65  # Umbral para considerar misma persona (bajado para mejor matching)
MAX_ABSENT_FRAMES = 150  # ~5s a 30fps antes de olvidar persona
GESTURE_CONFIDENCE = 0.5  # Confianza mínima para gestos
KALMAN_SMOOTH_FACTOR = 0.3  # Factor de suavizado (0-1)
DEBUG_MODE = True  # Mostrar información de debug
SHOW_ALL_DETECTIONS = True  # Mostrar TODAS las detecciones de YOLO (no solo personas)
OPTIMIZE_REID = True  # Solo ejecutar Re-ID cuando sea necesario (entrada/salida de frame)

# BBoxSmoother config
BBOX_DEAD_ZONE = 15  # Píxeles de zona muerta para evitar jitter
BBOX_SMOOTHING_FACTOR = 0.3  # Factor de suavizado para interpolación

# Kalman Filter para Embeddings
USE_KALMAN_EMBEDDINGS = True  # Usar Kalman para embeddings más robustos temporalmente
KALMAN_EMBEDDING_PROCESS_NOISE = 1e-3  # Ruido de proceso (qué tan rápido cambia)
KALMAN_EMBEDDING_MEASUREMENT_NOISE = 1e-2  # Ruido de medición (confianza en observación)


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================


@dataclass
class PersonRecord:
    """Registro de una persona rastreada."""

    person_id: str
    marked: bool = False  # Si hizo Closed_Fist
    frames_absent: int = 0  # Frames desde última detección
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_frame: int = 0
    # BBoxSmoother index para suavizado de bbox
    smoother_idx: Optional[int] = None
    predicted_bbox: Optional[Tuple[int, int, int, int]] = None
    # Kalman Filter para embedding (estado temporal del embedding)
    embedding_kalman: Optional[cv2.KalmanFilter] = None
    kalman_embedding: Optional[np.ndarray] = None  # Embedding filtrado por Kalman


# ============================================================================
# SISTEMA DE RE-IDENTIFICACIÓN CON CHROMADB
# ============================================================================


class PersonReID:
    """Sistema de Re-ID con embeddings mejorados y búsqueda vectorial."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_absent_frames: int = 300,
        use_kalman_embeddings: bool = True,
        bbox_smoother: Optional[BBoxSmoother] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames
        self.use_kalman_embeddings = use_kalman_embeddings
        self.bbox_smoother = bbox_smoother
        self.next_smoother_idx = 0  # Contador para índices de smoother

        # ChromaDB en memoria
        self.chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=False,
            )
        )

        self.collection = self.chroma_client.create_collection(
            name="person_embeddings", metadata={"hnsw:space": "cosine"}
        )

        self.persons: Dict[str, PersonRecord] = {}
        self.marked_ids: Set[str] = set()
        self.next_id = 1
        self.current_frame = 0

    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Extraer embedding mejorado con múltiples características.

        Combina:
        - Histogramas HSV (color robusto a iluminación)
        - HOG features (forma y postura)
        - Estadísticas de textura (gradientes, bordes)
        """
        x_min, y_min, x_max, y_max = bbox

        # Validar bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Verificar que bbox sea válido
        if x_max <= x_min or y_max <= y_min:
            return np.zeros(320, dtype=np.float32)

        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return np.zeros(320, dtype=np.float32)

        # Redimensionar a tamaño fijo (más pequeño para más velocidad)
        roi_resized = cv2.resize(roi, (48, 96))

        features = []

        # 1. HISTOGRAMAS HSV (Color) - 96 dimensiones
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

        # Histograma Hue (color) - más bins para mejor discriminación
        hist_h = cv2.calcHist([hsv], [0], None, [48], [0, 180])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()

        # Histograma Saturation (intensidad color)
        hist_s = cv2.calcHist([hsv], [1], None, [24], [0, 256])
        hist_s = cv2.normalize(hist_s, hist_s).flatten()

        # Histograma Value (brillo)
        hist_v = cv2.calcHist([hsv], [2], None, [24], [0, 256])
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        features.extend([hist_h, hist_s, hist_v])

        # 2. HOG FEATURES simplificado - 128 dimensiones
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        # Extraer HOG con parámetros más simples
        hog_features = hog(
            gray,
            orientations=8,
            pixels_per_cell=(12, 12),
            cells_per_block=(1, 1),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        # Tomar primeros 64 componentes de HOG
        hog_features = (
            hog_features[:64]
            if len(hog_features) >= 64
            else np.pad(hog_features, (0, 64 - len(hog_features)))
        )
        features.append(hog_features)

        # 3. GRADIENTES - 32 dimensiones
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        hist_grad, _ = np.histogram(magnitude, bins=32, range=(0, 255))
        hist_grad = hist_grad.astype(np.float32)
        hist_grad = hist_grad / (hist_grad.sum() + 1e-6)
        features.append(hist_grad)

        # Concatenar y normalizar
        embedding = np.concatenate(features).astype(np.float32)

        # Normalización L2 para similitud coseno
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _compute_simple_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Calcular características LBP simplificadas."""
        # Dividir en regiones
        h, w = gray.shape
        regions = []

        for i in range(0, h - 8, 16):
            for j in range(0, w - 8, 16):
                region = gray[i : i + 8, j : j + 8]
                if region.size > 0:
                    regions.append(region.std())

        if len(regions) == 0:
            return np.zeros(16, dtype=np.float32)

        # Histograma de desviaciones estándar
        hist, _ = np.histogram(regions, bins=16, range=(0, 100))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)

        return hist

    def find_or_create_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Encontrar persona por similitud o crear nueva."""
        embedding = self.extract_embedding(frame, bbox)

        # Primera persona
        if len(self.persons) == 0:
            return self._create_person(embedding, bbox)

        # Buscar match en ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(3, len(self.persons)),  # Buscar top 3 matches
            )

            if results["ids"] and len(results["ids"][0]) > 0:
                # Revisar top matches
                for i, (matched_id, distance) in enumerate(
                    zip(results["ids"][0], results["distances"][0])
                ):
                    # Convertir distancia coseno a similitud
                    similarity = 1.0 - (distance / 2.0)

                    if DEBUG_MODE and i == 0:
                        print(
                            f"🔍 Match: {matched_id} | Similitud: {similarity:.3f} | Umbral: {self.similarity_threshold}"
                        )

                    # Si la persona está marcada, siempre mantener su ID
                    if matched_id in self.marked_ids and similarity >= (
                        self.similarity_threshold - 0.1
                    ):
                        if DEBUG_MODE:
                            print(f"✅ Re-identificada persona MARCADA: {matched_id}")
                        self._update_person(matched_id, bbox, embedding)
                        return matched_id

                    # Match normal
                    if similarity >= self.similarity_threshold:
                        if DEBUG_MODE:
                            print(f"✅ Re-identificada: {matched_id}")
                        self._update_person(matched_id, bbox, embedding)
                        return matched_id

        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ Error en búsqueda vectorial: {e}")

        # No match, crear nueva
        return self._create_person(embedding, bbox)

    def _create_person(
        self,
        embedding: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Crear nueva persona."""
        person_id = f"P{self.next_id:03d}"
        self.next_id += 1

        # Inicializar Kalman para embedding temporal
        embedding_kalman = None
        kalman_embedding = embedding.copy()

        if self.use_kalman_embeddings:
            embedding_kalman = self._init_kalman_embedding(embedding)
            kalman_embedding = embedding.copy()
            if DEBUG_MODE:
                print(f"  🔬 Kalman embedding inicializado para {person_id}")

        # Agregar a ChromaDB (usar embedding filtrado si Kalman está activo)
        self.collection.add(
            embeddings=[kalman_embedding.tolist()],
            ids=[person_id],
        )

        # Inicializar BBoxSmoother
        smoother_idx = self.next_smoother_idx
        self.next_smoother_idx += 1

        # Crear registro
        self.persons[person_id] = PersonRecord(
            person_id=person_id,
            marked=False,
            frames_absent=0,
            last_bbox=bbox,
            last_seen_frame=self.current_frame,
            smoother_idx=smoother_idx,
            predicted_bbox=bbox,
            embedding_kalman=embedding_kalman,
            kalman_embedding=kalman_embedding,
        )

        print(f"✅ Nueva persona: {person_id}")
        return person_id

    def _update_person(
        self,
        person_id: str,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
    ):
        """Actualizar persona existente."""
        if person_id not in self.persons:
            return

        record = self.persons[person_id]
        record.frames_absent = 0
        record.last_bbox = bbox
        record.last_seen_frame = self.current_frame

        # Actualizar bbox suavizado con BBoxSmoother
        if self.bbox_smoother is not None and record.smoother_idx is not None:
            record.predicted_bbox = self.bbox_smoother.smooth(record.smoother_idx, bbox)
        else:
            record.predicted_bbox = bbox

        # Actualizar embedding con Kalman temporal
        if self.use_kalman_embeddings and record.embedding_kalman is not None:
            # Filtrar nuevo embedding con Kalman para obtener estado temporal
            filtered_embedding = self._update_kalman_embedding(record.embedding_kalman, embedding)
            record.kalman_embedding = filtered_embedding

            if DEBUG_MODE and self.current_frame % 90 == 0:
                # Calcular qué tan diferente es el embedding filtrado del observado
                diff = np.linalg.norm(filtered_embedding - embedding)
                print(f"  🔬 {person_id}: Kalman diff={diff:.3f} (menor = más estable)")

            # Actualizar ChromaDB con embedding filtrado (más estable)
            try:
                self.collection.update(
                    embeddings=[filtered_embedding.tolist()],
                    ids=[person_id],
                )
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Error actualizando embedding filtrado: {e}")
        else:
            # Sin Kalman: usar promedio móvil simple (método anterior)
            try:
                current = self.collection.get(ids=[person_id])
                if current["embeddings"]:
                    old_embedding = np.array(current["embeddings"][0])
                    updated_embedding = 0.7 * old_embedding + 0.3 * embedding

                    # Re-normalizar
                    norm = np.linalg.norm(updated_embedding)
                    if norm > 0:
                        updated_embedding = updated_embedding / norm

                    self.collection.update(
                        embeddings=[updated_embedding.tolist()],
                        ids=[person_id],
                    )
                    record.kalman_embedding = updated_embedding
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Error actualizando embedding: {e}")

    def _init_kalman_embedding(
        self, initial_embedding: np.ndarray, embedding_dim: int = 320
    ) -> cv2.KalmanFilter:
        """
        Inicializar Filtro de Kalman para embedding temporal.

        El Kalman mantiene un estado del embedding que evoluciona suavemente
        en el tiempo, capturando características más estables de la persona.

        Estado: [embedding_features] (solo el embedding actual)
        Medición: [embedding_features] (observación del frame actual)
        """
        # Estados = dimensiones del embedding, mediciones = dimensiones del embedding
        kalman = cv2.KalmanFilter(embedding_dim, embedding_dim)

        # Matriz de transición: el embedding persiste (modelo de persistencia)
        kalman.transitionMatrix = np.eye(embedding_dim, dtype=np.float32)

        # Matriz de medición: observamos directamente el embedding
        kalman.measurementMatrix = np.eye(embedding_dim, dtype=np.float32)

        # Covarianza del ruido de proceso (qué tan rápido puede cambiar la apariencia)
        kalman.processNoiseCov = (
            np.eye(embedding_dim, dtype=np.float32) * KALMAN_EMBEDDING_PROCESS_NOISE
        )

        # Covarianza del ruido de medición (confianza en la observación actual)
        kalman.measurementNoiseCov = (
            np.eye(embedding_dim, dtype=np.float32) * KALMAN_EMBEDDING_MEASUREMENT_NOISE
        )

        # Estado inicial con el primer embedding
        kalman.statePost = initial_embedding.astype(np.float32).reshape(-1, 1)

        # Covarianza del error inicial
        kalman.errorCovPost = np.eye(embedding_dim, dtype=np.float32) * 1.0

        return kalman

    def _update_kalman_embedding(
        self, kalman: cv2.KalmanFilter, new_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Actualizar Kalman con nueva observación de embedding.

        Returns:
            embedding filtrado que combina historia temporal con observación actual
        """
        # Predicción (estado previo)
        kalman.predict()

        # Corrección con nueva medición
        measurement = new_embedding.astype(np.float32).reshape(-1, 1)
        corrected = kalman.correct(measurement)

        # Convertir de vuelta a vector 1D
        filtered_embedding = corrected.flatten()

        # Re-normalizar para mantener propiedades de similitud coseno
        norm = np.linalg.norm(filtered_embedding)
        if norm > 0:
            filtered_embedding = filtered_embedding / norm

        return filtered_embedding

    def mark_person(self, person_id: str):
        """Marcar persona permanentemente."""
        if person_id in self.persons:
            self.persons[person_id].marked = True
            self.marked_ids.add(person_id)
            print(f"🔴 Persona {person_id} MARCADA permanentemente")

    def is_marked(self, person_id: str) -> bool:
        """Verificar si persona está marcada."""
        return person_id in self.marked_ids

    def update_absent_persons(self):
        """Actualizar contadores de ausencia."""
        to_remove = []

        for person_id, record in self.persons.items():
            if record.last_seen_frame < self.current_frame:
                record.frames_absent += 1

                # No remover personas marcadas
                if record.marked:
                    continue

                # Remover personas ausentes mucho tiempo
                if record.frames_absent > self.max_absent_frames:
                    to_remove.append(person_id)

        for person_id in to_remove:
            print(
                f"🗑️ Removiendo persona {person_id} (ausente {self.persons[person_id].frames_absent} frames)"
            )
            del self.persons[person_id]
            try:
                self.collection.delete(ids=[person_id])
            except Exception:
                pass

    def get_stats(self) -> Dict[str, int]:
        """Obtener estadísticas."""
        return {
            "total": len(self.persons),
            "marked": len(self.marked_ids),
            "active": sum(1 for p in self.persons.values() if p.frames_absent < 30),
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def find_closest_person_to_hand(
    hand_landmarks, detections, frame_width: int, frame_height: int
) -> Optional[int]:
    """Encontrar índice de persona más cercana a una mano."""
    if not hand_landmarks or len(detections) == 0:
        return None

    # Centroide de la mano
    hand_x = np.mean([lm.x for lm in hand_landmarks]) * frame_width
    hand_y = np.mean([lm.y for lm in hand_landmarks]) * frame_height

    min_distance = float("inf")
    closest_idx = None

    for idx, detection in enumerate(detections):
        bbox = detection[0]  # xyxy
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        distance = np.sqrt((hand_x - cx) ** 2 + (hand_y - cy) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_idx = idx

    # Solo retornar si está cerca (< 250 px)
    return closest_idx if min_distance < 250 else None


def render_overlay(
    frame: np.ndarray, person_reid: PersonReID, frame_index: int, fps: float, num_detections: int
):
    """Renderizar overlay de estadísticas."""
    stats = person_reid.get_stats()

    y = 30
    # Línea 1: Detecciones actuales
    cv2.putText(
        frame,
        f"Detecciones: {num_detections} | FPS: {fps:.1f} | Frame: {frame_index}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    y += 30
    # Línea 2: Re-ID stats
    cv2.putText(
        frame,
        f"Re-ID: {stats['total']} total | {stats['active']} activas | {stats['marked']} marcadas",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    y += 30
    # Línea 3: Configuración
    reid_mode = "Optimizado (entrada/salida)" if OPTIMIZE_REID else "Completo (cada frame)"
    cv2.putText(
        frame,
        f"Umbral: {SIMILARITY_THRESHOLD:.2f} | Memoria: {MAX_ABSENT_FRAMES}f | Re-ID: {reid_mode}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    y += 25
    # Línea 4: Controles
    cv2.putText(
        frame,
        "[Q] Salir | [R] Reset | [D] Debug | [A] Toggle All Detections",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("  SISTEMA DE TRACKING CON RE-ID AVANZADO")
    print("=" * 70)
    print("\n📋 Características:")
    print("  • BotSort (YOLO + Kalman + OSNet)")
    print("  • ChromaDB para búsqueda vectorial")
    print("  • Embeddings mejorados (HSV + HOG + Textura)")
    print(
        f"  • BBoxSmoother para bbox (dead_zone={BBOX_DEAD_ZONE}px, factor={BBOX_SMOOTHING_FACTOR})"
    )
    if USE_KALMAN_EMBEDDINGS:
        print(
            f"  • Kalman Filter para embeddings temporales (proceso={KALMAN_EMBEDDING_PROCESS_NOISE}, medición={KALMAN_EMBEDDING_MEASUREMENT_NOISE})"
        )
    print("  • Marcado permanente con Closed_Fist")
    print("  • Re-ID optimizado (solo en entrada/salida de frame)")
    print("\n🎨 Visualización:")
    print("  • Bounding Box VERDE: Persona normal")
    print("  • Bounding Box ROJO (grueso): Persona MARCADA")
    print("  • Keypoints de mano con gesto detectado")
    print("\n" + "=" * 70 + "\n")

    # Inicializar componentes
    print("🔧 Inicializando componentes...")

    # MediaPipe Gestures
    model_path = "./models/gesture_recognizer.task"
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=GESTURE_CONFIDENCE,
    )
    recognizer = GestureRecognizer.create_from_options(options)

    # BotSort Tracker
    print("  ⏳ Cargando YOLO directamente...")
    try:
        yolo_model = YOLO("./models/yolov8n.pt")
        print("  ✅ YOLO cargado correctamente")
    except Exception as e:
        print(f"  ❌ Error cargando YOLO: {e}")
        return

    print("  ⏳ Cargando BotSort tracker...")
    try:
        tracker = BotSort(
            model_weights=Path("./models/yolov8n.pt"),
            device="cpu",
            reid_weights=Path("./models/osnet_x0_25_msmt17.pt"),
            half=False,
        )
        print("  ✅ BotSort cargado correctamente")
    except Exception as e:
        print(f"  ❌ Error cargando BotSort: {e}")
        print("  ℹ️ Usando solo YOLO sin tracker")
        tracker = None

    # BBoxSmoother para suavizar bounding boxes
    bbox_smoother = BBoxSmoother(dead_zone=BBOX_DEAD_ZONE, smoothing_factor=BBOX_SMOOTHING_FACTOR)

    # PersonReID con ChromaDB y Kalman de embeddings
    person_reid = PersonReID(
        similarity_threshold=SIMILARITY_THRESHOLD,
        max_absent_frames=MAX_ABSENT_FRAMES,
        use_kalman_embeddings=USE_KALMAN_EMBEDDINGS,
        bbox_smoother=bbox_smoother,
    )

    # Anotadores Supervision
    box_annotator_green = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    box_annotator_red = sv.BoxAnnotator(
        thickness=4, color=sv.Color.RED
    )  # ROJO para personas marcadas
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER,
        text_scale=0.6,
        text_thickness=2,
    )

    # Video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir video source {VIDEO_SOURCE}")
        return

    print("✅ Sistema iniciado correctamente\n")

    # Mapeo: índice de detección → person_id de Re-ID
    detection_to_reid: Dict[int, str] = {}

    # Tracking de tracker_ids previos para optimizar Re-ID
    previous_tracker_ids: Set[int] = set()
    tracker_to_reid: Dict[int, str] = {}  # Mapeo tracker_id → person_id

    frame_index = 0
    fps_start_time = time.time()
    fps = 0.0

    # Variables locales para toggle
    show_all_detections = SHOW_ALL_DETECTIONS
    debug_mode = DEBUG_MODE

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            person_reid.current_frame = frame_index

            # Calcular FPS
            if frame_index % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # ====== TRACKING CON BOTSORT O YOLO ======
            try:
                # Primero: Obtener detecciones de YOLO
                yolo_results = yolo_model(frame, verbose=False)

                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                    boxes = yolo_results[0].boxes

                    if DEBUG_MODE and frame_index % 30 == 0:
                        total_detections = len(boxes)
                        person_detections = sum(1 for box in boxes if box.cls[0].cpu().numpy() == 0)
                        print(
                            f"🔍 YOLO detectó: {total_detections} objetos ({person_detections} personas)"
                        )

                    # Convertir a formato [x1, y1, x2, y2, conf, class_id]
                    yolo_detections = []
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        yolo_detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
                    yolo_detections = np.array(yolo_detections)

                    # Segundo: Pasar detecciones a BotSort para tracking
                    if tracker is not None:
                        results = tracker.update(yolo_detections, frame)
                    else:
                        # Sin tracker, usar YOLO directamente con IDs temporales
                        results = []
                        for i, det in enumerate(yolo_detections):
                            results.append([det[0], det[1], det[2], det[3], i, det[4], det[5]])
                        results = np.array(results)
                else:
                    if DEBUG_MODE and frame_index % 30 == 0:
                        print("⚠️ YOLO no detectó ningún objeto")
                    results = None

            except Exception as e:
                if debug_mode:
                    print(f"❌ Error en tracker.update(): {e}")
                    import traceback

                    traceback.print_exc()
                results = None

            if DEBUG_MODE and frame_index % 30 == 0:
                if results is not None and len(results) > 0:
                    print(f"📦 BotSort detectó {len(results)} objetos")
                else:
                    print("⚠️ BotSort no detectó nada")
                    # Verificar que el frame sea válido
                    if frame is None or frame.size == 0:
                        print("  ⚠️ Frame inválido!")
                    else:
                        print(f"  ℹ️ Frame válido: {frame.shape}")

            # Convertir a Supervision Detections
            if results is not None and len(results) > 0:
                xyxy = results[:, :4]
                confidence = results[:, 5] if results.shape[1] > 5 else None
                class_id = results[:, 6].astype(int) if results.shape[1] > 6 else None
                tracker_id = results[:, 4].astype(int)

                if DEBUG_MODE and frame_index % 30 == 0:
                    print(f"  Clases detectadas: {np.unique(class_id)}")

                detections = sv.Detections(
                    xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id
                )
            else:
                detections = sv.Detections.empty()

            # Filtrar solo personas (clase 0 en COCO)
            original_count = len(detections)
            if len(detections) > 0 and detections.class_id is not None:
                detections = detections[detections.class_id == 0]

            if DEBUG_MODE and frame_index % 30 == 0:
                print(f"  Personas filtradas: {len(detections)} de {original_count} detecciones")

            # ====== DEBUG: Mostrar TODAS las detecciones si está habilitado ======
            if show_all_detections and original_count > 0:
                # Dibujar TODAS las detecciones en rojo (antes del filtro)
                debug_frame = frame.copy()
                if results is not None and len(results) > 0:
                    for result in results:
                        x1, y1, x2, y2 = map(int, result[:4])
                        class_id = int(result[6]) if len(result) > 6 else -1
                        conf = result[5] if len(result) > 5 else 0.0

                        # COCO class names (abreviado)
                        coco_names = [
                            "person",
                            "bicycle",
                            "car",
                            "motorcycle",
                            "airplane",
                            "bus",
                            "train",
                            "truck",
                            "boat",
                            "traffic light",
                            "fire hydrant",
                        ]
                        class_name = (
                            coco_names[class_id]
                            if class_id < len(coco_names)
                            else f"clase_{class_id}"
                        )

                        color = (
                            (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        )  # Verde para personas, rojo para otros
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            debug_frame,
                            f"{class_name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                cv2.imshow("DEBUG: Todas las detecciones YOLO", debug_frame)

            # ====== RE-IDENTIFICACIÓN OPTIMIZADA ======
            detection_to_reid.clear()
            current_tracker_ids = set()

            for idx, detection in enumerate(detections):
                bbox_xyxy = detection[0]
                bbox = (int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3]))

                tracker_id = int(detection[4]) if detection[4] is not None else None
                current_tracker_ids.add(tracker_id)

                # OPTIMIZACIÓN: Solo ejecutar Re-ID si es necesario
                if OPTIMIZE_REID and tracker_id is not None:
                    # Caso 1: Ya conocemos este tracker_id
                    if tracker_id in tracker_to_reid:
                        person_id = tracker_to_reid[tracker_id]
                        detection_to_reid[idx] = person_id

                        # Actualizar registro sin calcular embedding
                        if person_id in person_reid.persons:
                            record = person_reid.persons[person_id]
                            record.frames_absent = 0
                            record.last_bbox = bbox
                            record.last_seen_frame = person_reid.current_frame

                            # Actualizar sistema de suavizado de bbox
                            if (
                                person_reid.bbox_smoother is not None
                                and record.smoother_idx is not None
                            ):
                                record.predicted_bbox = person_reid.bbox_smoother.smooth(
                                    record.smoother_idx, bbox
                                )
                            else:
                                record.predicted_bbox = bbox

                        if DEBUG_MODE and frame_index % 90 == 0:
                            print(
                                f"⚡ Re-usando ID: tracker_{tracker_id} → {person_id} (sin Re-ID)"
                            )
                        continue

                    # Caso 2: Tracker_id nuevo - ejecutar Re-ID completo
                    if DEBUG_MODE and frame_index % 90 == 0:
                        print(f"🔍 Nuevo tracker_id {tracker_id} - ejecutando Re-ID completo")

                # Buscar o crear persona con Re-ID completo
                person_id = person_reid.find_or_create_person(frame, bbox)
                detection_to_reid[idx] = person_id

                # Guardar mapeo
                if tracker_id is not None:
                    tracker_to_reid[tracker_id] = person_id

            # Limpiar tracker_ids que ya no están presentes
            disappeared_ids = previous_tracker_ids - current_tracker_ids
            for tracker_id in disappeared_ids:
                if tracker_id in tracker_to_reid:
                    if DEBUG_MODE:
                        print(f"🗑️ Tracker_id {tracker_id} desapareció")
                    # No eliminar inmediatamente por si vuelve

            previous_tracker_ids = current_tracker_ids.copy()

            # Actualizar personas ausentes
            person_reid.update_absent_persons()

            # ====== RECONOCIMIENTO DE GESTOS Y VISUALIZACIÓN DE MANOS ======
            gesture_annotations = {}  # person_id → (hand_landmarks, gesture_name, confidence)

            for idx, detection in enumerate(detections):
                person_id = detection_to_reid.get(idx)
                if not person_id:
                    continue

                bbox = detection[0]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Recortar persona
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Reconocer gesto
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

                try:
                    gesture_result = recognizer.recognize(mp_image)

                    if gesture_result.gestures and gesture_result.hand_landmarks:
                        for gesture_list, hand_landmarks in zip(
                            gesture_result.gestures, gesture_result.hand_landmarks
                        ):
                            if not gesture_list:
                                continue

                            top_gesture = gesture_list[0]

                            # Guardar para visualización
                            gesture_annotations[person_id] = (
                                hand_landmarks,
                                top_gesture.category_name,
                                top_gesture.score,
                                (x1, y1),  # offset para coordenadas
                            )

                            # Detectar Closed_Fist
                            if (
                                top_gesture.category_name == "Closed_Fist"
                                and top_gesture.score >= GESTURE_CONFIDENCE
                            ):

                                person_reid.mark_person(person_id)
                except Exception:
                    pass  # Ignorar errores de reconocimiento

            # ====== VISUALIZACIÓN ======
            annotated_frame = frame.copy()

            # Separar detecciones por estado
            detections_normal = []
            detections_marked = []
            labels_normal = []
            labels_marked = []

            for idx, detection in enumerate(detections):
                person_id = detection_to_reid.get(idx)
                if not person_id:
                    continue

                is_marked = person_reid.is_marked(person_id)

                # Convertir detection a lista mutable
                detection = list(detection)

                # Usar bbox suavizado por Kalman si está disponible
                if person_id in person_reid.persons:
                    record = person_reid.persons[person_id]
                    if record.predicted_bbox:
                        # Actualizar bbox en detection con predicción Kalman
                        detection[0] = np.array(record.predicted_bbox, dtype=np.float32)

                label = f"{person_id}"
                if is_marked:
                    label += " [MARCADO]"
                    detections_marked.append(detection)
                    labels_marked.append(label)
                else:
                    detections_normal.append(detection)
                    labels_normal.append(label)

            # Dibujar personas normales (verde)
            if len(detections_normal) > 0:
                detections_normal_obj = sv.Detections(
                    xyxy=np.array([d[0] for d in detections_normal]),
                    confidence=(
                        np.array([d[1] for d in detections_normal])
                        if detections_normal[0][1] is not None
                        else None
                    ),
                    class_id=(
                        np.array([d[2] for d in detections_normal], dtype=np.int32)
                        if detections_normal[0][2] is not None
                        else None
                    ),
                    tracker_id=np.array([int(d[4]) for d in detections_normal], dtype=np.int32),
                )
                annotated_frame = box_annotator_green.annotate(
                    scene=annotated_frame, detections=detections_normal_obj
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections_normal_obj, labels=labels_normal
                )

            # Dibujar personas marcadas (ROJO con borde grueso)
            if len(detections_marked) > 0:
                detections_marked_obj = sv.Detections(
                    xyxy=np.array([d[0] for d in detections_marked]),
                    confidence=(
                        np.array([d[1] for d in detections_marked])
                        if detections_marked[0][1] is not None
                        else None
                    ),
                    class_id=(
                        np.array([d[2] for d in detections_marked], dtype=np.int32)
                        if detections_marked[0][2] is not None
                        else None
                    ),
                    tracker_id=np.array([int(d[4]) for d in detections_marked], dtype=np.int32),
                )
                annotated_frame = box_annotator_red.annotate(
                    scene=annotated_frame, detections=detections_marked_obj
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections_marked_obj, labels=labels_marked
                )

            # ====== DIBUJAR KEYPOINTS DE MANOS Y GESTOS ======
            for person_id, (
                hand_landmarks,
                gesture_name,
                confidence,
                offset,
            ) in gesture_annotations.items():
                x_offset, y_offset = offset

                # Dibujar conexiones de la mano
                HAND_CONNECTIONS = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),  # Pulgar
                    (0, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),  # Índice
                    (0, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),  # Medio
                    (0, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),  # Anular
                    (0, 17),
                    (17, 18),
                    (18, 19),
                    (19, 20),  # Meñique
                    (5, 9),
                    (9, 13),
                    (13, 17),  # Palma
                ]

                # Convertir landmarks a coordenadas pixel
                h, w = frame.shape[:2]
                bbox = (
                    person_reid.persons[person_id].last_bbox
                    if person_id in person_reid.persons
                    else None
                )
                if bbox:
                    crop_w = bbox[2] - bbox[0]
                    crop_h = bbox[3] - bbox[1]

                    # Dibujar conexiones
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]

                        start_x = int(start.x * crop_w + x_offset)
                        start_y = int(start.y * crop_h + y_offset)
                        end_x = int(end.x * crop_w + x_offset)
                        end_y = int(end.y * crop_h + y_offset)

                        # Color según si está marcada
                        is_marked = person_reid.is_marked(person_id)
                        line_color = (0, 0, 255) if is_marked else (255, 200, 0)  # Rojo o celeste

                        cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), line_color, 2)

                    # Dibujar keypoints
                    for landmark in hand_landmarks:
                        x = int(landmark.x * crop_w + x_offset)
                        y = int(landmark.y * crop_h + y_offset)
                        cv2.circle(annotated_frame, (x, y), 4, (255, 255, 255), -1)
                        cv2.circle(annotated_frame, (x, y), 3, line_color, -1)

                    # Anotar gesto detectado
                    text = f"{gesture_name} ({confidence:.2f})"
                    text_x = x_offset
                    text_y = y_offset - 10

                    # Fondo para el texto
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(
                        annotated_frame,
                        (text_x, text_y - text_h - 5),
                        (text_x + text_w + 5, text_y + 5),
                        (0, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        annotated_frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Overlay de estadísticas
            render_overlay(annotated_frame, person_reid, frame_index, fps, len(detections))

            # Mostrar
            cv2.imshow("Tracking con Re-ID Avanzado", annotated_frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                # Toggle mostrar todas las detecciones
                show_all_detections = not show_all_detections
                if not show_all_detections:
                    cv2.destroyWindow("DEBUG: Todas las detecciones YOLO")
                print(
                    f"\n{'✅ Mostrar todas las detecciones ON' if show_all_detections else '❌ Mostrar todas las detecciones OFF'}\n"
                )
            elif key == ord("d"):
                # Toggle debug mode
                debug_mode = not debug_mode
                print(f"\n{'✅ Debug ON' if debug_mode else '❌ Debug OFF'}\n")
            elif key == ord("r"):
                print("\n🔄 Reseteando sistema Re-ID...")
                person_reid.chroma_client.reset()
                person_reid.collection = person_reid.chroma_client.create_collection(
                    name="person_embeddings", metadata={"hnsw:space": "cosine"}
                )
                person_reid.persons.clear()
                person_reid.marked_ids.clear()
                person_reid.next_id = 1
                person_reid.next_smoother_idx = 0
                tracker_to_reid.clear()
                previous_tracker_ids.clear()
                if bbox_smoother is not None:
                    bbox_smoother.reset()
                print("✅ Re-ID reseteado\n")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por usuario")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Sistema finalizado\n")


if __name__ == "__main__":
    main()
