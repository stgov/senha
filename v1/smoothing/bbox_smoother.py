from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class BBoxSmoother:
    def __init__(self, dead_zone: int = 15, smoothing_factor: float = 0.3):
        self.dead_zone = dead_zone
        self.smoothing_factor = smoothing_factor
        self.previous_bboxes: Dict[int, np.ndarray] = {}
    
    def calculate_centroid(self, bbox: np.ndarray) -> np.ndarray:
        points = bbox.reshape(2, 2)
        return points.mean(axis=0)
    
    def calculate_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        centroid1 = self.calculate_centroid(bbox1)
        centroid2 = self.calculate_centroid(bbox2)
        return np.linalg.norm(centroid2 - centroid1)
    
    def smooth(self, person_idx: int, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        new_bbox_array = np.array(new_bbox, dtype=np.float32)
        
        if person_idx not in self.previous_bboxes:
            self.previous_bboxes[person_idx] = new_bbox_array
            return new_bbox
        
        prev_bbox = self.previous_bboxes[person_idx]
        distance = self.calculate_distance(prev_bbox, new_bbox_array)
        
        if distance < self.dead_zone:
            return tuple(prev_bbox.astype(int))
        
        smoothed_bbox = prev_bbox + self.smoothing_factor * (new_bbox_array - prev_bbox)
        self.previous_bboxes[person_idx] = smoothed_bbox
        
        return tuple(smoothed_bbox.astype(int))
    
    def get_centroid_from_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        bbox_array = np.array(bbox, dtype=np.float32)
        centroid = self.calculate_centroid(bbox_array)
        return int(centroid[0]), int(centroid[1])
    
    def reset(self) -> None:
        self.previous_bboxes.clear()
