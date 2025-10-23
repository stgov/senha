"""
Sistema de Re-ID con tracking - Versión 2 (Modular)
"""
__version__ = "2.0.0"

from .core.config import TrackingConfig
from .core.embedding_extractor import EmbeddingExtractor
from .core.person_reid import PersonReID, PersonRecord
from .tracking.tracker import PersonTracker
from .gestures.recognizer import GestureRecognizer
from .visualization.renderer import Visualizer
from .smoothing.bbox_smoother import BBoxSmoother
from .main_v2 import main as main_v2

__all__ = [
    'TrackingConfig',
    'EmbeddingExtractor',
    'PersonReID',
    'PersonRecord',
    'PersonTracker',
    'GestureRecognizer',
    'Visualizer',
    'BBoxSmoother',
    'main_v2',
]
