from typing import Dict, Tuple, Optional


class BBoxSmoother:
    def __init__(self, dead_zone: int = 15, smoothing_factor: float = 0.3):
        self.dead_zone = dead_zone
        self.smoothing_factor = smoothing_factor
        self.previous_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    
    def calculate_centroid(self, x_min: int, y_min: int, x_max: int, y_max: int) -> Tuple[int, int]:
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        return cx, cy
    
    def calculate_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        cx1, cy1 = self.calculate_centroid(*bbox1)
        cx2, cy2 = self.calculate_centroid(*bbox2)
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
    
    def smooth(self, person_idx: int, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if person_idx not in self.previous_bboxes:
            self.previous_bboxes[person_idx] = new_bbox
            return new_bbox
        
        prev_bbox = self.previous_bboxes[person_idx]
        distance = self.calculate_distance(prev_bbox, new_bbox)
        
        if distance < self.dead_zone:
            return prev_bbox
        
        prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_bbox
        new_x_min, new_y_min, new_x_max, new_y_max = new_bbox
        
        alpha = self.smoothing_factor
        smooth_x_min = int(prev_x_min + alpha * (new_x_min - prev_x_min))
        smooth_y_min = int(prev_y_min + alpha * (new_y_min - prev_y_min))
        smooth_x_max = int(prev_x_max + alpha * (new_x_max - prev_x_max))
        smooth_y_max = int(prev_y_max + alpha * (new_y_max - prev_y_max))
        
        smoothed_bbox = (smooth_x_min, smooth_y_min, smooth_x_max, smooth_y_max)
        self.previous_bboxes[person_idx] = smoothed_bbox
        
        return smoothed_bbox
    
    def reset(self) -> None:
        self.previous_bboxes.clear()
