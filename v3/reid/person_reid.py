import chromadb
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Set
from .embedding_utils import extract_embedding

class PersonReID:
    def __init__(self, similarity_threshold: float = 0.5, max_absent_frames: int = 60, kalman_process_noise: float = 1e-4, kalman_measurement_noise: float = 1e-2):
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        
        # ChromaDB en memoria
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="person_embeddings", metadata={"hnsw:space": "cosine"})
        
        # Dimensiones del embedding (debe coincidir con embedding_utils.py)
        self.embedding_dim = 320 # 96 (HSV) + 192 (HOG) + 32 (Grad)
        
        # Estado interno
        self.persons: Dict[str, Dict] = {} # person_id -> { "kalman": cv2.KalmanFilter, "stable_embedding": np.ndarray, "last_seen_tracker": int, "frames_absent": int }
        self.tracker_to_person_map: Dict[int, str] = {} # tracker_id -> person_id
        self.marked_ids: Set[str] = set()
        self.next_person_id = 1

    def _init_kalman(self, initial_embedding: np.ndarray) -> cv2.KalmanFilter:
        """Inicializa un Filtro de Kalman para un vector de embedding."""
        kalman = cv2.KalmanFilter(self.embedding_dim, self.embedding_dim)
        kalman.transitionMatrix = np.eye(self.embedding_dim, dtype=np.float32)
        kalman.measurementMatrix = np.eye(self.embedding_dim, dtype=np.float32)
        kalman.processNoiseCov = np.eye(self.embedding_dim, dtype=np.float32) * self.kalman_process_noise
        kalman.measurementNoiseCov = np.eye(self.embedding_dim, dtype=np.float32) * self.kalman_measurement_noise
        kalman.statePost = initial_embedding.reshape(-1, 1).astype(np.float32)
        kalman.errorCovPost = np.eye(self.embedding_dim, dtype=np.float32) * 0.1
        return kalman

    def _update_kalman_embedding(self, kalman: cv2.KalmanFilter, new_embedding: np.ndarray) -> np.ndarray:
        """Actualiza el Kalman con un nuevo embedding y devuelve el estado filtrado."""
        predicted = kalman.predict().flatten()
        measurement = new_embedding.astype(np.float32).reshape(-1, 1)
        kalman.correct(measurement)
        stable_embedding = kalman.statePost.flatten()
        
        # Re-normalizar para similitud coseno
        norm = np.linalg.norm(stable_embedding)
        if norm > 0:
            stable_embedding /= norm
        return stable_embedding

    def _create_person(self, initial_embedding: np.ndarray, tracker_id: int) -> str:
        """Registra una nueva persona en el sistema."""
        person_id = f"P{self.next_person_id:03d}"
        self.next_person_id += 1
        
        kalman = self._init_kalman(initial_embedding)
        
        self.persons[person_id] = {
            "kalman": kalman,
            "stable_embedding": initial_embedding,
            "last_seen_tracker": tracker_id,
            "frames_absent": 0
        }
        
        self.collection.add(embeddings=[initial_embedding.tolist()], ids=[person_id])
        self.tracker_to_person_map[tracker_id] = person_id
        
        return person_id, True # (person_id, is_new)

    def identify_person(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int], tracker_id: int) -> Tuple[str, bool]:
        """Identifica o crea una persona basada en su apariencia y tracking."""
        x1, y1, x2, y2 = bbox
        roi = frame_rgb[y1:y2, x1:x2]
        
        # Validar ROI
        if roi.size == 0:
            # Si el ROI es inválido, re-usar el ID del tracker si existe
            if tracker_id in self.tracker_to_person_map:
                return self.tracker_to_person_map[tracker_id], False
            else:
                # Caso raro: ROI inválido y tracker nuevo. No podemos hacer nada.
                return "P_INVALID", False

        current_embedding = extract_embedding(roi)
        
        # --- Caso 1: El Tracker ya es conocido ---
        if tracker_id in self.tracker_to_person_map:
            person_id = self.tracker_to_person_map[tracker_id]
            
            # Actualizar estado
            record = self.persons[person_id]
            record["frames_absent"] = 0
            record["last_seen_tracker"] = tracker_id
            
            # Actualizar Kalman y embedding estable
            kalman = record["kalman"]
            stable_embedding = self._update_kalman_embedding(kalman, current_embedding)
            record["stable_embedding"] = stable_embedding
            
            # Actualizar en ChromaDB
            self.collection.update(ids=[person_id], embeddings=[stable_embedding.tolist()])
            
            return person_id, False # (person_id, is_new)
            
        # --- Caso 2: Tracker nuevo, buscar por apariencia (Re-ID) ---
        if self.collection.count() > 0:
            try:
                results = self.collection.query(
                    query_embeddings=[current_embedding.tolist()],
                    n_results=1
                )
                
                # Asegurarse de que results['ids'][0] no esté vacío
                if results['ids'] and results['ids'][0]:
                    best_match_id = results['ids'][0][0]
                    best_match_distance = results['distances'][0][0]
                    similarity = 1.0 - best_match_distance
                    
                    if similarity >= self.similarity_threshold:
                        # ¡Re-identificación exitosa!
                        person_id = best_match_id
                        self.tracker_to_person_map[tracker_id] = person_id
                        
                        record = self.persons[person_id]
                        record["frames_absent"] = 0
                        record["last_seen_tracker"] = tracker_id
                        
                        # Actualizar Kalman
                        kalman = record["kalman"]
                        stable_embedding = self._update_kalman_embedding(kalman, current_embedding)
                        record["stable_embedding"] = stable_embedding
                        self.collection.update(ids=[person_id], embeddings=[stable_embedding.tolist()])
                        
                        return person_id, False # (person_id, is_new)
                        
            except Exception as e:
                print(f"[Error ReID Query]: {e}")
                pass # Continuar para crear nueva persona

        # --- Caso 3: Tracker nuevo, sin match de apariencia -> Persona Nueva ---
        return self._create_person(current_embedding, tracker_id)

    def cleanup_absent_persons(self, active_tracker_ids: Set[int]):
        """Limpia personas que han estado ausentes por mucho tiempo."""
        absent_person_ids = []
        
        # Incrementar ausencia de personas no activas
        for person_id, record in self.persons.items():
            if record["last_seen_tracker"] not in active_tracker_ids:
                record["frames_absent"] += 1
                
                # Si la persona está marcada, nunca la borres
                if person_id in self.marked_ids:
                    continue
                    
                if record["frames_absent"] > self.max_absent_frames:
                    absent_person_ids.append(person_id)

        # Eliminar personas ausentes
        if absent_person_ids:
            for person_id in absent_person_ids:
                print(f"[INFO] Eliminando persona ausente: {person_id}")
                del self.persons[person_id]
                
            self.collection.delete(ids=absent_person_ids)
            
            # Limpiar el mapa de trackers
            self.tracker_to_person_map = {tid: pid for tid, pid in self.tracker_to_person_map.items() if pid not in absent_person_ids}

    def mark_person(self, person_id: str):
        """Marca una persona como permanente (ej. por gesto)."""
        self.marked_ids.add(person_id)

    def reset(self):
        """Resetea el estado completo del sistema Re-ID."""
        print("[ReID] Reseteando base de datos y estados...")
        self.client.delete_collection(name="person_embeddings")
        self.collection = self.client.create_collection(name="person_embeddings", metadata={"hnsw:space": "cosine"})
        self.persons.clear()
        self.tracker_to_person_map.clear()
        self.marked_ids.clear()
        self.next_person_id = 1

    def shutdown(self):
        """Limpia la base de datos en memoria (no es necesario para persistente)."""
        try:
            self.client.delete_collection(name="person_embeddings")
            print("[ReID] Base de datos en memoria limpiada.")
        except Exception as e:
            print(f"[Error shutdown ReID]: {e}")

