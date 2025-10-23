"""
Sistema de Re-Identificación de Personas con ChromaDB y Kalman Filter para embeddings.
"""
from __future__ import annotations

import cv2
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass

from .config import TrackingConfig
from .embedding_extractor import EmbeddingExtractor


@dataclass
class PersonRecord:
    """Registro de una persona rastreada."""
    person_id: str
    marked: bool = False
    frames_absent: int = 0
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_frame: int = 0
    embedding_kalman: Optional[cv2.KalmanFilter] = None
    kalman_embedding: Optional[np.ndarray] = None


class PersonReID:
    """
    Sistema de Re-Identificación con:
    - ChromaDB para búsqueda vectorial
    - Kalman Filter para embeddings temporales robustos
    - BBoxSmoother para suavizado de bounding boxes
    """
    
    def __init__(
        self,
        config: TrackingConfig,
        embedding_extractor: EmbeddingExtractor,
    ):
        self.config = config
        self.embedding_extractor = embedding_extractor

        self.chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=False,
            )
        )
        
        self.collection = self.chroma_client.create_collection(
            name="person_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.persons: Dict[str, PersonRecord] = {}
        self.marked_ids: Set[str] = set()
        self.next_id = 1
        self.current_frame = 0
    
    def find_or_create_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Encontrar persona por similitud o crear nueva."""
        embedding = self.embedding_extractor.extract(frame, bbox)
        
        if len(self.persons) == 0:
            return self._create_person(embedding, bbox)
        
        # Buscar match en ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(3, len(self.persons)),
            )
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, (matched_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    similarity = 1.0 - (distance / 2.0)
                    
                    if self.config.debug_mode and i == 0:
                        print(f"🔍 Match: {matched_id} | Similitud: {similarity:.3f} | Umbral: {self.config.similarity_threshold}")
                    
                    # Prioridad para personas marcadas
                    if matched_id in self.marked_ids and similarity >= (self.config.similarity_threshold - 0.1):
                        if self.config.debug_mode:
                            print(f"✅ Re-identificada persona MARCADA: {matched_id}")
                        self._update_person(matched_id, bbox, embedding)
                        return matched_id
                    
                    if similarity >= self.config.similarity_threshold:
                        if self.config.debug_mode:
                            print(f"✅ Re-identificada: {matched_id}")
                        self._update_person(matched_id, bbox, embedding)
                        return matched_id
        
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️ Error en búsqueda vectorial: {e}")
        
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
        
        if self.config.use_kalman_embeddings:
            embedding_kalman = self._init_kalman_embedding(embedding)
            kalman_embedding = embedding.copy()
            if self.config.debug_mode:
                print(f"  🔬 Kalman embedding inicializado para {person_id}")
        
        # Agregar a ChromaDB
        self.collection.add(
            embeddings=[kalman_embedding.tolist()],
            ids=[person_id],
        )
        
        # Crear registro
        self.persons[person_id] = PersonRecord(
            person_id=person_id,
            marked=False,
            frames_absent=0,
            last_bbox=bbox,
            last_seen_frame=self.current_frame,
            embedding_kalman=embedding_kalman,
            kalman_embedding=kalman_embedding
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
        
        # Actualizar embedding con Kalman temporal
        if self.config.use_kalman_embeddings and record.embedding_kalman is not None:
            filtered_embedding = self._update_kalman_embedding(record.embedding_kalman, embedding)
            record.kalman_embedding = filtered_embedding
            
            if self.config.debug_mode and self.current_frame % 90 == 0:
                diff = np.linalg.norm(filtered_embedding - embedding)
                print(f"  🔬 {person_id}: Kalman diff={diff:.3f} (menor = más estable)")
            
            # Actualizar ChromaDB con embedding filtrado (más estable)
            try:
                self.collection.update(
                    embeddings=[filtered_embedding.tolist()],
                    ids=[person_id],
                )
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️ Error actualizando embedding: {e}")
        else:
            # Sin Kalman: usar promedio móvil simple
            try:
                current = self.collection.get(ids=[person_id])
                if current['embeddings']:
                    old_embedding = np.array(current['embeddings'][0])
                    # Promedio móvil: 70% viejo, 30% nuevo
                    updated_embedding = 0.7 * old_embedding + 0.3 * embedding
                    
                    # Re-normalizar
                    norm = np.linalg.norm(updated_embedding)
                    if norm > 0:
                        updated_embedding = updated_embedding / norm
                    
                    self.collection.update(
                        embeddings=[updated_embedding.tolist()],
                        ids=[person_id],
                    )
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️ Error actualizando embedding: {e}")
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️ Error actualizando embedding: {e}")
    
    def _init_kalman_embedding(
        self,
        initial_embedding: np.ndarray,
    ) -> cv2.KalmanFilter:
        """Inicializar Filtro de Kalman para embedding temporal."""
        embedding_dim = len(initial_embedding)
        kalman = cv2.KalmanFilter(embedding_dim, embedding_dim)
        
        kalman.transitionMatrix = np.eye(embedding_dim, dtype=np.float32)
        kalman.measurementMatrix = np.eye(embedding_dim, dtype=np.float32)
        kalman.processNoiseCov = np.eye(embedding_dim, dtype=np.float32) * self.config.kalman_embedding_process_noise
        kalman.measurementNoiseCov = np.eye(embedding_dim, dtype=np.float32) * self.config.kalman_embedding_measurement_noise
        kalman.statePost = initial_embedding.astype(np.float32).reshape(-1, 1)
        kalman.errorCovPost = np.eye(embedding_dim, dtype=np.float32) * 1.0
        
        return kalman
    
    def _update_kalman_embedding(
        self,
        kalman: cv2.KalmanFilter,
        new_embedding: np.ndarray
    ) -> np.ndarray:
        """Actualizar Kalman con nueva observación de embedding."""
        kalman.predict()
        measurement = new_embedding.astype(np.float32).reshape(-1, 1)
        corrected = kalman.correct(measurement)
        filtered_embedding = corrected.flatten()
        
        # Re-normalizar
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
                
                if record.marked:
                    continue
                
                if record.frames_absent > self.config.max_absent_frames:
                    to_remove.append(person_id)
        
        for person_id in to_remove:
            print(f"🗑️ Removiendo persona {person_id} (ausente {self.persons[person_id].frames_absent} frames)")
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
    
    def reset(self):
        """Resetear sistema Re-ID."""
        self.chroma_client.reset()
        self.collection = self.chroma_client.create_collection(
            name="person_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.persons.clear()
        self.marked_ids.clear()
        self.next_id = 1
        print("🔄 Sistema Re-ID reseteado")
        self.bbox_smoother.reset()
