import numpy as np
from typing import Dict, Tuple

class BoundingBoxSmoother:
    def __init__(self, smoothing_factor: float = 0.5, dead_zone: int = 10):
        """
        Inicializa el suavizador de Bounding Boxes.
        
        Args:
            smoothing_factor (float): Factor de suavizado (alpha). 
                                     Valores más altos = más rápido responde (menos suave).
                                     Valores más bajos = más lento responde (más suave).
            dead_zone (int): Píxeles. Si el nuevo BBox está dentro de esta distancia 
                             del BBox suavizado, no se actualiza (evita jitter).
        """
        self.smoothing_factor = smoothing_factor
        self.dead_zone = dead_zone
        self.smoothers: Dict[str, np.ndarray] = {} # person_id -> [x1, y1, x2, y2]

    def reset(self):
        """Resetea todos los suavizadores almacenados."""
        print("[Smoother] Reseteando suavizadores...")
        self.smoothers.clear()

    def smooth(self, person_id: str, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Suaviza un Bounding Box usando interpolación exponencial y una zona muerta.
        
        Args:
            person_id (str): ID de la persona para seguimiento.
            new_bbox (Tuple): El nuevo BBox detectado (x1, y1, x2, y2).
            
        Returns:
            El BBox suavizado (x1, y1, x2, y2).
        """
        new_bbox_np = np.array(new_bbox, dtype=np.float32)

        if person_id not in self.smoothers:
            # Si es la primera vez, inicializar el smoother con el BBox actual
            self.smoothers[person_id] = new_bbox_np
            return new_bbox

        # Obtener el BBox suavizado anterior
        prev_smooth_bbox = self.smoothers[person_id]

        # --- Lógica de Zona Muerta (Dead Zone) ---
        # Calcular la diferencia en el centro y el tamaño
        center_diff = np.linalg.norm(
            (new_bbox_np[:2] + new_bbox_np[2:])/2 - (prev_smooth_bbox[:2] + prev_smooth_bbox[2:])/2
        )
        size_diff = np.linalg.norm(
            (new_bbox_np[2:] - new_bbox_np[:2]) - (prev_smooth_bbox[2:] - prev_smooth_bbox[:2])
        )
        
        # Si el cambio es muy pequeño (dentro de la zona muarta), no hacer nada
        if center_diff < self.dead_zone and size_diff < self.dead_zone:
            # Devolver el BBox suavizado anterior (convertido a int)
            return tuple(prev_smooth_bbox.astype(int))

        # --- Lógica de Suavizado (Exponential Moving Average) ---
        # Si el cambio es significativo, aplicar suavizado
        smoothed_bbox_np = (
            self.smoothing_factor * new_bbox_np +
            (1 - self.smoothing_factor) * prev_smooth_bbox
        )

        # Guardar el nuevo BBox suavizado
        self.smoothers[person_id] = smoothed_bbox_np

        # Devolver el BBox suavizado (convertido a int)
        return tuple(smoothed_bbox_np.astype(int))

