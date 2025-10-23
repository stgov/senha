"""
Sistema de Re-Identificación de Personas (Re-ID) con memoria vectorial.

Este módulo maneja:
- Extracción de embeddings visuales de personas
- Búsqueda de similitud en base de datos vectorial (ChromaDB)
- Persistencia en memoria de IDs y estados de marcado
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import chromadb
import cv2
import numpy as np
from chromadb.config import Settings
from sklearn.preprocessing import normalize


@dataclass
class PersonRecord:
    """Registro de una persona identificada."""

    person_id: str
    marked: bool = False  # Si realizó el gesto Closed_Fist
    frame_count: int = 0  # Frames desde última detección
    last_bbox: Optional[Tuple[int, int, int, int]] = None


class PersonReID:
    """Sistema de re-identificación de personas usando embeddings visuales."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_absent_frames: int = 300,  # ~10 segundos a 30fps
    ):
        """
        Inicializar sistema Re-ID.

        Args:
            similarity_threshold: Umbral de similitud coseno para match (0-1)
            max_absent_frames: Frames máximos antes de olvidar persona ausente
        """
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames

        # Base de datos vectorial en memoria (no persiste)
        self.chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=False,
            )
        )

        # Colección para embeddings de personas
        self.collection = self.chroma_client.create_collection(
            name="person_embeddings", metadata={"hnsw:space": "cosine"}
        )

        # Registro de personas conocidas
        self.persons: Dict[str, PersonRecord] = {}

        # Contador para IDs únicos
        self.next_id = 1

        # Personas marcadas permanentemente
        self.marked_ids: Set[str] = set()

    def extract_person_embedding(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        use_color_histogram: bool = True,
    ) -> np.ndarray:
        """
        Extraer embedding visual de una persona desde su bounding box.

        Estrategia: Combinar histograma de color + textura simple.
        Para producción, considerar modelos pre-entrenados (ResNet, OSNet).

        Args:
            frame: Frame completo
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            use_color_histogram: Incluir histograma de color

        Returns:
            Vector de características normalizado
        """
        x_min, y_min, x_max, y_max = bbox

        # Extraer región de interés
        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            # ROI vacía, retornar embedding dummy
            return np.zeros(256, dtype=np.float32)

        # Redimensionar a tamaño fijo para consistencia
        roi_resized = cv2.resize(roi, (64, 128))

        features = []

        if use_color_histogram:
            # Histograma de color en HSV (más robusto a iluminación)
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

            # Histogramas por canal
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Normalizar histogramas
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            features.extend([hist_h, hist_s, hist_v])

        # Características de textura (gradientes)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        # Gradientes horizontales y verticales
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Histograma de magnitud de gradientes
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        hist_grad, _ = np.histogram(magnitude, bins=32, range=(0, 255))
        hist_grad = hist_grad.astype(np.float32)
        hist_grad = hist_grad / (hist_grad.sum() + 1e-6)

        features.append(hist_grad)

        # Histograma de bordes (Canny)
        edges = cv2.Canny(gray, 50, 150)
        hist_edges, _ = np.histogram(edges, bins=32, range=(0, 255))
        hist_edges = hist_edges.astype(np.float32)
        hist_edges = hist_edges / (hist_edges.sum() + 1e-6)

        features.append(hist_edges)

        # Concatenar todas las características
        embedding = np.concatenate(features).astype(np.float32)

        # Normalizar L2
        embedding = normalize(embedding.reshape(1, -1), norm="l2").flatten()

        return embedding

    def find_or_create_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """
        Encontrar persona conocida por similitud o crear nueva entrada.

        Args:
            frame: Frame completo
            bbox: Bounding box de la persona

        Returns:
            ID de la persona (existente o nueva)
        """
        embedding = self.extract_person_embedding(frame, bbox)

        # Si no hay personas registradas, crear primera entrada
        if len(self.persons) == 0:
            return self._create_new_person(embedding, bbox)

        # Buscar match en base de datos vectorial
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
            )

            if results["ids"] and len(results["ids"][0]) > 0:
                matched_id = results["ids"][0][0]
                distance = results["distances"][0][0]

                # Similitud coseno: 0 = idéntico, 2 = opuesto
                # Convertir a similitud: 1 - (distance / 2)
                similarity = 1.0 - (distance / 2.0)

                if similarity >= self.similarity_threshold:
                    # Match encontrado
                    self._update_person(matched_id, bbox, embedding)
                    return matched_id

        except Exception as e:
            print(f"Error en búsqueda vectorial: {e}")

        # No match, crear nueva persona
        return self._create_new_person(embedding, bbox)

    def _create_new_person(
        self,
        embedding: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Crear nueva entrada de persona."""
        person_id = f"P{self.next_id:03d}"
        self.next_id += 1

        # Agregar a base de datos vectorial
        self.collection.add(
            embeddings=[embedding.tolist()],
            ids=[person_id],
        )

        # Registrar persona
        self.persons[person_id] = PersonRecord(
            person_id=person_id,
            marked=False,
            frame_count=0,
            last_bbox=bbox,
        )

        return person_id

    def _update_person(
        self,
        person_id: str,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
    ):
        """Actualizar registro de persona existente."""
        if person_id in self.persons:
            record = self.persons[person_id]
            record.frame_count = 0
            record.last_bbox = bbox

            # Actualizar embedding (promedio móvil suave)
            try:
                # Obtener embedding actual
                current = self.collection.get(ids=[person_id])
                if current["embeddings"]:
                    old_embedding = np.array(current["embeddings"][0])
                    # Promedio ponderado: 70% anterior + 30% nuevo
                    updated_embedding = 0.7 * old_embedding + 0.3 * embedding
                    updated_embedding = normalize(
                        updated_embedding.reshape(1, -1), norm="l2"
                    ).flatten()

                    # Actualizar en base de datos
                    self.collection.update(
                        embeddings=[updated_embedding.tolist()],
                        ids=[person_id],
                    )
            except Exception as e:
                print(f"Error actualizando embedding: {e}")

    def mark_person(self, person_id: str):
        """Marcar persona permanentemente (realizó Closed_Fist)."""
        if person_id in self.persons:
            self.persons[person_id].marked = True
            self.marked_ids.add(person_id)

    def is_marked(self, person_id: str) -> bool:
        """Verificar si persona está marcada permanentemente."""
        return person_id in self.marked_ids

    def update_absent_persons(self):
        """Incrementar contador de frames ausentes y limpiar personas antiguas."""
        to_remove = []

        for person_id, record in self.persons.items():
            record.frame_count += 1

            # No remover personas marcadas
            if record.marked:
                continue

            # Remover personas no vistas por mucho tiempo
            if record.frame_count > self.max_absent_frames:
                to_remove.append(person_id)

        # Limpiar personas antiguas
        for person_id in to_remove:
            del self.persons[person_id]
            try:
                self.collection.delete(ids=[person_id])
            except:
                pass

    def get_stats(self) -> Dict[str, int]:
        """Obtener estadísticas del sistema Re-ID."""
        return {
            "total_persons": len(self.persons),
            "marked_persons": len(self.marked_ids),
            "active_persons": sum(1 for p in self.persons.values() if p.frame_count < 30),
        }

    def reset(self):
        """Reiniciar sistema Re-ID (limpiar memoria)."""
        self.chroma_client.reset()
        self.collection = self.chroma_client.create_collection(
            name="person_embeddings", metadata={"hnsw:space": "cosine"}
        )
        self.persons.clear()
        self.marked_ids.clear()
        self.next_id = 1
