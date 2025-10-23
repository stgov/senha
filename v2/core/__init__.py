"""Inicialización del módulo core."""

from .config import TrackingConfig
from .models import PersonRecord
from .person_reid import PersonReID
from .embedding_extractor import EmbeddingExtractor
from .model_manager import ModelManager

__all__ = [
    'TrackingConfig',
    'PersonRecord',
    'PersonReID',
    'EmbeddingExtractor',
    'ModelManager',
]
