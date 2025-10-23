"""Inicialización del módulo reid."""

from .embedding import extract_person_embedding
from .person_reid import PersonReID

__all__ = ["PersonReID", "extract_person_embedding"]
