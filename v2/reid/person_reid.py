"""
Sistema de Re-Identificación de personas con ChromaDB y Kalman Filter.
"""

from typing import Dict, Optional, Set, Tuple

import chromadb
import cv2
import numpy as np
from chromadb.config import Settings

from ..core.config import TrackingConfig
from ..core.models import PersonRecord
from ..smoothing.bbox_smoother import BBoxSmoother
from .embedding import extract_person_embedding


class PersonReID:
    """
    Sistema de Re-ID con embeddings mejorados, búsqueda vectorial y Kalman temporal.

    El Filtro de Kalman NO se usa para tracking de bbox, sino para mantener
    un embedding temporal más estable que capture el "estado global" de la persona.
    """

    def __init__(
        self,
        config: TrackingConfig,
        bbox_smoother: Optional[BBoxSmoother] = None,
    ):
        self.config = config
        self.bbox_smoother = bbox_smoother
        self.next_smoother_idx = 0

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

    def find_or_create_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Encontrar persona por similitud o crear nueva."""
        embedding = extract_person_embedding(frame, bbox, debug=self.config.debug_mode)

        # Primera persona
        if len(self.persons) == 0:
            return self._create_person(embedding, bbox)

        # Buscar match en ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(3, len(self.persons)),
            )

            if results["ids"] and len(results["ids"][0]) > 0:
                for i, (matched_id, distance) in enumerate(
                    zip(results["ids"][0], results["distances"][0])
                ):
                    similarity = 1.0 - (distance / 2.0)

                    if self.config.debug_mode and i == 0:
                        print(
                            f"🔍 Match: {matched_id} | Similitud: {similarity:.3f} | Umbral: {self.config.similarity_threshold}"
                        )

                    # Personas marcadas tienen umbral más bajo
                    if matched_id in self.marked_ids and similarity >= (
                        self.config.similarity_threshold - 0.1
                    ):
                        if self.config.debug_mode:
                            print(f"✅ Re-identificada persona MARCADA: {matched_id}")
                        self._update_person(matched_id, bbox, embedding)
                        return matched_id

                    # Match normal
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
        """Crear nueva persona con Kalman embedding."""
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

        # Inicializar smoother
        smoother_idx = None
        if self.bbox_smoother is not None:
            smoother_idx = self.next_smoother_idx
            self.next_smoother_idx += 1

        # Crear registro
        self.persons[person_id] = PersonRecord(
            person_id=person_id,
            marked=False,
            smoother_idx=smoother_idx,
            embedding_kalman=embedding_kalman,
            kalman_embedding=kalman_embedding,
            last_seen_frame=self.current_frame,
            entrance_frame=self.current_frame,
        )

        print(f"✅ Nueva persona: {person_id}")
        return person_id

    def _update_person(
        self,
        person_id: str,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
    ):
        """Actualizar persona existente con Kalman embedding."""
        if person_id not in self.persons:
            return

        record = self.persons[person_id]
        record.last_seen_frame = self.current_frame

        # Actualizar embedding con Kalman temporal
        if self.config.use_kalman_embeddings and record.embedding_kalman is not None:
            filtered_embedding = self._update_kalman_embedding(record.embedding_kalman, embedding)
            record.kalman_embedding = filtered_embedding

            if self.config.debug_mode and self.current_frame % 90 == 0:
                diff = np.linalg.norm(filtered_embedding - embedding)
                print(f"  🔬 {person_id}: Kalman diff={diff:.3f}")

            # Actualizar ChromaDB con embedding filtrado
            try:
                self.collection.update(
                    embeddings=[filtered_embedding.tolist()],
                    ids=[person_id],
                )
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️ Error actualizando embedding: {e}")
        else:
            # Sin Kalman: promedio móvil simple
            try:
                current = self.collection.get(ids=[person_id])
                if current["embeddings"]:
                    old_embedding = np.array(current["embeddings"][0])
                    updated_embedding = 0.7 * old_embedding + 0.3 * embedding

                    norm = np.linalg.norm(updated_embedding)
                    if norm > 0:
                        updated_embedding = updated_embedding / norm

                    self.collection.update(
                        embeddings=[updated_embedding.tolist()],
                        ids=[person_id],
                    )
                    record.kalman_embedding = updated_embedding
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️ Error actualizando embedding: {e}")

    def _init_kalman_embedding(
        self, initial_embedding: np.ndarray, embedding_dim: int = 320
    ) -> cv2.KalmanFilter:
        """
        Inicializar Filtro de Kalman para embedding temporal.

        El Kalman mantiene un estado del embedding que evoluciona suavemente,
        capturando características más estables de la persona a lo largo del tiempo.
        """
        kalman = cv2.KalmanFilter(embedding_dim, embedding_dim)

        kalman.transitionMatrix = np.eye(embedding_dim, dtype=np.float32)
        kalman.measurementMatrix = np.eye(embedding_dim, dtype=np.float32)
        kalman.processNoiseCov = (
            np.eye(embedding_dim, dtype=np.float32) * self.config.kalman_embedding_process_noise
        )
        kalman.measurementNoiseCov = (
            np.eye(embedding_dim, dtype=np.float32) * self.config.kalman_embedding_measurement_noise
        )
        kalman.statePost = initial_embedding.astype(np.float32).reshape(-1, 1)
        kalman.errorCovPost = np.eye(embedding_dim, dtype=np.float32) * 1.0

        return kalman

    def _update_kalman_embedding(
        self, kalman: cv2.KalmanFilter, new_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Actualizar Kalman con nueva observación de embedding.

        Returns:
            Embedding filtrado que combina historia temporal con observación actual.
        """
        kalman.predict()
        measurement = new_embedding.astype(np.float32).reshape(-1, 1)
        corrected = kalman.correct(measurement)

        filtered_embedding = corrected.flatten()

        # Re-normalizar para similitud coseno
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
        """Remover personas ausentes (excepto marcadas)."""
        to_remove = []

        for person_id, record in self.persons.items():
            frames_absent = self.current_frame - record.last_seen_frame

            if record.marked:
                continue

            if frames_absent > self.config.max_absent_frames:
                to_remove.append(person_id)

        for person_id in to_remove:
            frames_absent = self.current_frame - self.persons[person_id].last_seen_frame
            print(f"🗑️ Removiendo {person_id} (ausente {frames_absent} frames)")
            del self.persons[person_id]
            try:
                self.collection.delete(ids=[person_id])
            except Exception:
                pass

    def get_stats(self) -> Dict[str, int]:
        """Obtener estadísticas del sistema."""
        active = sum(
            1 for p in self.persons.values() if (self.current_frame - p.last_seen_frame) < 30
        )
        return {
            "total": len(self.persons),
            "marked": len(self.marked_ids),
            "active": active,
        }
