"""Inicialización del módulo core."""

from .config import TrackingConfig
from .embedding_extractor import EmbeddingExtractor
from .model_manager import ModelManager
from .models import PersonRecord
from .person_reid import PersonReID

__all__ = [
    "TrackingConfig",
    "PersonRecord",
    "PersonReID",
    "EmbeddingExtractor",
    "ModelManager",
]
