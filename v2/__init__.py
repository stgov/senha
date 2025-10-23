"""
Sistema de Re-ID con tracking - Versión 2 (Modular)
"""

__version__ = "2.0.0"

from .core.config import TrackingConfig
from .core.embedding_extractor import EmbeddingExtractor
from .core.person_reid import PersonRecord, PersonReID
from .gestures.recognizer import GestureRecognizer
from .main_v2 import main as main_v2
from .smoothing.bbox_smoother import BBoxSmoother
from .tracking.tracker import PersonTracker
from .visualization.renderer import Visualizer

__all__ = [
    "TrackingConfig",
    "EmbeddingExtractor",
    "PersonReID",
    "PersonRecord",
    "PersonTracker",
    "GestureRecognizer",
    "Visualizer",
    "BBoxSmoother",
    "main_v2",
]
