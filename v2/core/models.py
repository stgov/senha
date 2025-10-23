"""
Clases de datos para el sistema de tracking.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PersonRecord:
    """Registro de una persona detectada."""
    person_id: str
    marked: bool = False
    smoother_idx: Optional[int] = None
    embedding_kalman: Optional[object] = field(default=None, repr=False)
    kalman_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    last_seen_frame: int = 0
    entrance_frame: int = 0
