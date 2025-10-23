"""Inicialización del módulo reid."""

from .person_reid import PersonReID
from .embedding import extract_person_embedding

__all__ = ['PersonReID', 'extract_person_embedding']
