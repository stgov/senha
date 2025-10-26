import numpy as np
from typing import Dict, Tuple

class BoundingBoxSmoother:
    """
    Aplica suavizado a los bounding boxes para reducir el "jitter" (vibración).
    
    Utiliza una combinación de:
    1. Interpolación Lineal (Lerp) para un movimiento gradual.
    2. Una "zona muerta" (dead zone) para ignorar movimientos pequeños.
    """
    
    def __init__(self, smoothing_factor: float = 0.5, dead_zone: int = 10):
        """
        Inicializa el suavizador.

        Args:
            smoothing_factor (float): Factor de suavizado (alpha para Lerp).        
                                      0.0 = sin suavizado (salta), 
                                      1.0 = no se mueve. 
                                      Un valor bueno es 0.3-0.5.
            dead_zone (int): Píxeles. Si el centroide se mueve menos que esto, 
                             no se actualiza la posición (ignora jitter).
        """
        self.smoothing_factor = smoothing_factor
        self.dead_zone = dead_zone
        # Almacena el último bounding box suavizado [x1, y1, x2, y2]
        self.last_boxes: Dict[str, np.ndarray] = {}

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Calcula el centroide (cx, cy) de un bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def smooth(self, person_id: str, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Aplica suavizado a un nuevo bounding box detectado para una persona.
        """
        new_bbox_np = np.array(new_bbox)
        
        # --- Caso 1: Primera vez que vemos a esta persona ---
        if person_id not in self.last_boxes:
            self.last_boxes[person_id] = new_bbox_np
            return new_bbox

        last_bbox_np = self.last_boxes[person_id]

        # --- Caso 2: Aplicar Zona Muerta (Dead Zone) ---
        last_center = self._get_center(last_bbox_np.astype(int))
        new_center = self._get_center(new_bbox)
        
        distance = np.linalg.norm(new_center - last_center)
        
        if distance < self.dead_zone:
            # El movimiento es muy pequeño (jitter), no hacer nada
            return tuple(last_bbox_np.astype(int))

        # --- Caso 3: Aplicar Suavizado (Lerp) ---
        # Interpolamos linealmente entre el último box y el nuevo
        smoothed_box_np = (
            last_bbox_np * self.smoothing_factor +
            new_bbox_np * (1.0 - self.smoothing_factor)
        )
        
        # Actualizar el último box
        self.last_boxes[person_id] = smoothed_box_np
        
        return tuple(smoothed_box_np.astype(int))

    def reset(self):
        """Limpia el historial de bounding boxes."""
        self.last_boxes.clear()

