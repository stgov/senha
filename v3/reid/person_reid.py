import cv2
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import Dict, Tuple, Optional

# Importar la función con el nombre correcto
from reid.embedding_utils import extract_embedding, EMBEDDING_DIM

# Configuración del Filtro de Kalman para Embeddings
KALMAN_PROCESS_NOISE = 1e-4  # Qué tan rápido puede cambiar la apariencia
KALMAN_MEASUREMENT_NOISE = 1e-3 # Confianza en la observación actual

class PersonReID:
    """
    Gestiona la Re-Identificación (ReID) de personas.
    
    - Asigna IDs permanentes (person_id) a IDs temporales (tracker_id).
    - Utiliza ChromaDB para búsqueda de similitud de apariencia.
    - Aplica un Filtro de Kalman al *embedding* para una firma de apariencia
      más robusta y estable en el tiempo.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        similarity_threshold: float = 0.55,
        max_absent_frames: int = 150
    ):
        """
        Inicializa el sistema ReID.

        Args:
            db_path (Optional[str]): Ruta para ChromaDB persistente. 
                                     Si es None, se usa en memoria.
            similarity_threshold (float): Umbral de similitud (1.0 - distancia coseno)
                                          para considerar a dos personas como la misma.
            max_absent_frames (int): Nº de frames antes de olvidar a una persona.
        """
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames
        
        if db_path:
            self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        else:
            print("Iniciando ChromaDB en modo in-memoria.")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            
        # Usamos distancia Coseno (L2 normalizado)
        self.collection = self.client.get_or_create_collection(
            name="person_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        # --- Almacenes de Estado ---
        # Mapeo: tracker_id (temporal) -> person_id (permanente)
        self.tracker_to_person: Dict[int, str] = {}
        # Mapeo: person_id -> Filtro de Kalman para su embedding
        self.kalman_filters: Dict[str, cv2.KalmanFilter] = {}
        # Mapeo: person_id -> Nº de frames ausente
        self.frames_absent: Dict[str, int] = {}
        
        self.next_person_id = 1

    def _init_kalman_filter(self) -> cv2.KalmanFilter:
        """Inicializa un Filtro de Kalman para un vector de embedding."""
        kalman = cv2.KalmanFilter(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        # Matriz de transición (Estado = [embedding, delta_embedding])
        kalman.transitionMatrix = np.eye(EMBEDDING_DIM * 2, dtype=np.float32)
        kalman.transitionMatrix[0:EMBEDDING_DIM, EMBEDDING_DIM:] = np.eye(EMBEDDING_DIM, dtype=np.float32)
        
        # Matriz de medición (Solo medimos el embedding)
        kalman.measurementMatrix = np.zeros((EMBEDDING_DIM, EMBEDDING_DIM * 2), dtype=np.float32)
        kalman.measurementMatrix[0:EMBEDDING_DIM, 0:EMBEDDING_DIM] = np.eye(EMBEDDING_DIM, dtype=np.float32)
        
        # Ruidos
        kalman.processNoiseCov = np.eye(EMBEDDING_DIM * 2, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kalman.measurementNoiseCov = np.eye(EMBEDDING_DIM, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        
        return kalman

    def _update_kalman_embedding(self, kalman: cv2.KalmanFilter, embedding: np.ndarray) -> np.ndarray:
        """Predice y corrige el Kalman con un nuevo embedding."""
        predicted = kalman.predict().flatten()
        
        # Corregir con la nueva medición
        measurement = embedding.astype(np.float32).reshape(-1, 1)
        kalman.correct(measurement)
        
        # El estado filtrado es la primera mitad del vector de estado
        stable_embedding = kalman.statePost[0:EMBEDDING_DIM].flatten()
        
        # Re-normalizar (importante para coseno)
        norm = np.linalg.norm(stable_embedding)
        if norm > 0:
            stable_embedding /= norm
            
        return stable_embedding

    def identify_person(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int], tracker_id: int) -> Tuple[str, bool]:
        """
        Identifica a una persona, la rastrea y actualiza su embedding estable.
        """
        is_new_person = False
        
        # --- 1. Extraer embedding del frame actual ---
        
        # --- CORRECCIÓN DEL ERROR ---
        # Recortar el ROI del frame completo
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2: # Salvaguarda
            return "unknown", False
            
        roi = frame_rgb[y1:y2, x1:x2]
        
        # Llamar con un solo argumento (el ROI)
        current_embedding = extract_embedding(roi)
        # ---------------------------

        # --- 2. Verificar si ya conocemos este tracker_id ---
        if tracker_id in self.tracker_to_person:
            person_id = self.tracker_to_person[tracker_id]
            self.frames_absent[person_id] = 0 # Marcar como presente
            
            # Actualizar embedding con Kalman
            kalman = self.kalman_filters[person_id]
            stable_embedding = self._update_kalman_embedding(kalman, current_embedding)
            
            # Actualizar en ChromaDB
            self.collection.update(ids=[person_id], embeddings=[stable_embedding.tolist()])
            return person_id, False # No es nuevo

        # --- 3. Buscar por similitud de apariencia (ReID) ---
        if self.collection.count() > 0:
            results = self.collection.query(
                query_embeddings=[current_embedding.tolist()],
                n_results=1
            )
            
            if results['ids'] and results['distances']:
                best_match_id = results['ids'][0][0]
                best_match_distance = results['distances'][0][0]
                similarity = 1.0 - best_match_distance # Distancia Coseno -> Similitud
                
                if similarity >= self.similarity_threshold:
                    # ¡Re-identificado!
                    person_id = best_match_id
                    print(f"Re-identificado: Tracker {tracker_id} es {person_id} (Sim: {similarity:.2f})")
                    self.tracker_to_person[tracker_id] = person_id
                    self.frames_absent[person_id] = 0 # Marcar como presente
                    
                    # Actualizar Kalman
                    kalman = self.kalman_filters[person_id]
                    stable_embedding = self._update_kalman_embedding(kalman, current_embedding)
                    self.collection.update(ids=[person_id], embeddings=[stable_embedding.tolist()])
                    return person_id, False # No es nuevo

        # --- 4. Nueva persona detectada ---
        is_new_person = True
        person_id = f"P{self.next_person_id:03d}"
        self.next_person_id += 1
        print(f"Nueva persona detectada: {person_id} (asignada a tracker {tracker_id})")

        # Inicializar su Filtro de Kalman
        kalman = self._init_kalman_filter()
        # Inicializar estado del Kalman
        kalman.statePost[0:EMBEDDING_DIM] = current_embedding.reshape(-1, 1)
        self.kalman_filters[person_id] = kalman
        
        # Añadir a estados
        self.tracker_to_person[tracker_id] = person_id
        self.frames_absent[person_id] = 0
        
        # Añadir a ChromaDB (usamos el primer embedding como inicial)
        self.collection.add(
            ids=[person_id],
            embeddings=[current_embedding.tolist()]
        )
        
        return person_id, is_new_person

    def update_absent(self, active_tracker_ids: set):
        """Maneja la lógica de personas que salen del frame."""
        
        current_persons = set(self.tracker_to_person.values())
        active_persons = set(self.tracker_to_person[tid] for tid in active_tracker_ids)
        
        absent_persons = current_persons - active_persons
        
        for person_id in absent_persons:
            self.frames_absent[person_id] += 1
            
        # Limpiar trackers antiguos
        inactive_tracker_ids = set(self.tracker_to_person.keys()) - active_tracker_ids
        for tracker_id in inactive_tracker_ids:
            person_id = self.tracker_to_person[tracker_id]
            if self.frames_absent.get(person_id, 0) > self.max_absent_frames:
                # Olvidar a esta persona
                print(f"Olvidando a {person_id} (ausente por {self.frames_absent[person_id]} frames)")
                if person_id in self.kalman_filters:
                    del self.kalman_filters[person_id]
                if person_id in self.frames_absent:
                    del self.frames_absent[person_id]
                self.collection.delete(ids=[person_id])
                
            # Siempre eliminar el mapeo del tracker (podría reaparecer)
            del self.tracker_to_person[tracker_id]

    def shutdown(self):
        """Limpia la base de datos ChromaDB."""
        try:
            print("Limpiando colección de ChromaDB...")
            ids_to_delete = self.collection.get()['ids']
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            print(f"Se eliminaron {len(ids_to_delete)} entradas.")
        except Exception as e:
            print(f"Error limpiando ChromaDB: {e}")

